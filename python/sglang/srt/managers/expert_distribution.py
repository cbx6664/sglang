import logging
from abc import ABC
from contextlib import contextmanager
from copy import deepcopy
from typing import List, Type

import torch
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import Withable, get_bool_env_var

logger = logging.getLogger(__name__)


# --------------------------------------- Entrypoint -----------------------------------------

class _ExpertDistributionRecorder:
    """Global expert distribution recording"""

    def __init__(self):
        self._recording = False
        self._current_layer_idx = Withable()
        self._single_pass_gatherer = _SinglePassGatherer.init_new(server_args)
        self._accumulator = _Accumulator.init_new()

    def with_current_layer(self, layer_idx):
        return self._current_layer_idx.with_value(layer_idx)

    @contextmanager
    def with_forward_pass(self):
        try:
            yield
        finally:
            self._on_forward_pass_end()

    def _on_forward_pass_end(self):
        single_pass_physical_count = self._single_pass_gatherer.collect()
        self._accumulator.append(single_pass_physical_count)
        self._single_pass_gatherer.reset()

    def on_select_experts(self, topk_ids: torch.Tensor):
        if not self._recording:
            return
        self._single_pass_gatherer.on_select_experts(layer_idx=self._current_layer_idx.value, topk_ids=topk_ids)

    def on_deepep_dispatch_normal(self, num_recv_tokens_per_expert_list: List[int]):
        if not self._recording:
            return
        self._single_pass_gatherer.on_deepep_dispatch_normal(self._current_layer_idx.value,
                                                             num_recv_tokens_per_expert_list)

    def _reset(self):
        """Reset the expert distribution recorder."""
        logger.info("Resetting ExpertDistributionRecorder...")
        self._recording = False
        assert self._current_layer_idx.value is None
        self._single_pass_gatherer.reset()
        self._accumulator.reset()

    def start_record(self):
        """Start recording the expert distribution."""
        if self._recording:
            logger.warning(
                "SGLang server is already recording expert ids. Did you forget to dump the expert ids recorded so far by sending requests to the `/stop_expert_distribution_record` and `/dump_expert_distribution_record` endpoints?"
            )
        self._reset()
        self._recording = True

    def stop_record(self):
        """Stop recording the expert distribution."""
        if not self._recording:
            logger.warning(
                "SGLang server has not been recording expert ids. Did you forget to start recording by sending request to the `/start_expert_distribution_record` endpoint?"
            )
        self._recording = False

    def dump_record(self):
        """Dump the expert distribution record and reset the recorder after dumping."""
        output = self._accumulator.dump()
        self._reset()
        return output


expert_distribution_recorder = _ExpertDistributionRecorder()


# --------------------------------------- SinglePassGatherer -----------------------------------------

class _SinglePassGatherer(ABC):
    @staticmethod
    def init_new(server_args: ServerArgs) -> "_SinglePassGatherer":
        if server_args.enable_deepep_moe:
            # TODO DeepEP low latency
            return _DeepepNormalSinglePassGatherer()
        return _LayerBasedSinglePassGatherer()

    def on_select_experts(self, layer_idx: int, topk_ids: torch.Tensor):
        pass

    def on_deepep_dispatch_normal(self, layer_idx: int, num_recv_tokens_per_expert_list: List[int]):
        pass

    def reset(self):
        raise NotImplementedError

    def collect(self) -> torch.Tensor:
        raise NotImplementedError


class _LayerBasedSinglePassGatherer(_SinglePassGatherer):
    def __init__(self):
        self._num_recv_tokens_per_expert_list_of_layer = {}

    def _on_layer_data(self, layer_idx: int, num_recv_tokens_per_expert_list: List[int]):
        # TODO for TBO, we may need to relax this restriction
        assert layer_idx not in self._num_recv_tokens_per_expert_list_of_layer
        assert 0 <= layer_idx < num_layers
        self._num_recv_tokens_per_expert_list_of_layer[layer_idx] = num_recv_tokens_per_expert_list

    def reset(self):
        self._num_recv_tokens_per_expert_list_of_layer.clear()

    def collect(self) -> torch.Tensor:
        data = [
            self._num_recv_tokens_per_expert_list_of_layer.get(layer_index) or ([0] * num_local_physical_experts)
            for layer_index in range(num_layers)
        ]
        return torch.tensor(data)


class _SelectExpertsSinglePassGatherer(_LayerBasedSinglePassGatherer):
    # pretty slow, but we will use the DeepEP Gatherer in production
    def on_select_experts(self, layer_idx: int, topk_ids: torch.Tensor):
        topk_ids_list = topk_ids.to("cpu", non_blocking=True).numpy().tolist()
        torch.cuda.synchronize()

        num_recv_tokens_per_expert_list = [0] * num_local_physical_experts
        for token_record in topk_ids_list:
            for expert_idx in token_record:
                num_recv_tokens_per_expert_list[expert_idx] += 1

        self._on_layer_data(layer_idx, num_recv_tokens_per_expert_list)


class _DeepepNormalSinglePassGatherer(_LayerBasedSinglePassGatherer):
    def on_deepep_dispatch_normal(self, layer_idx: int, num_recv_tokens_per_expert_list: List[int]):
        assert isinstance(num_recv_tokens_per_expert_list, list)
        self._on_layer_data(layer_idx, num_recv_tokens_per_expert_list)


# TODO Wait for LowLatency DeepEP merging
# e.g. use naive tensor copying
class _DeepepLowLatencySinglePassGatherer(_SinglePassGatherer):
    pass


# --------------------------------------- Accumulator -----------------------------------------

class _Accumulator(ABC):
    @staticmethod
    def init_new() -> "_Accumulator":
        return _Accumulator.get_class()()

    @staticmethod
    def get_class() -> Type["_Accumulator"]:
        if get_bool_env_var("SGLANG_EXPERT_DISTRIBUTION_RECORDER_DETAIL"):
            return _DetailAccumulator
        return _StatAccumulator

    def append(self, single_pass_physical_count: torch.Tensor):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def dump(self):
        raise NotImplementedError


class _DetailAccumulator(_Accumulator):
    def __init__(self):
        self._records = []

    def append(self, single_pass_physical_count: torch.Tensor):
        self._records.append(dict(
            physical_count=single_pass_physical_count.tolist(),
        ))

    def reset(self):
        self._records.clear()

    def dump(self):
        return deepcopy(self._records)


class _StatAccumulator(_Accumulator):
    def __init__(self):
        self._physical_count = torch.zeros((num_layers, num_local_physical_experts))

    def append(self, single_pass_physical_count: torch.Tensor):
        self._physical_count += single_pass_physical_count

    def reset(self):
        self._physical_count[...] = 0

    def dump(self):
        return dict(
            physical_count=self._physical_count.tolist(),
        )
