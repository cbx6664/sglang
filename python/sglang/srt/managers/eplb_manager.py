from typing import TYPE_CHECKING

from sglang.srt.managers import deepseek_eplb
from sglang.srt.managers.expert_distribution_storage import ExpertDistributionStorage

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager


class EPLBManager:
    def __init__(self, tokenizer_manager: TokenizerManager):
        self._tokenizer_manager = tokenizer_manager
        self._expert_distribution_storage = ExpertDistributionStorage()

    async def rebalance_experts(self):
        TODO_may_or_may_not_save_current
        logical_count = self._expert_distribution_storage.get_last_snapshot()
        expert_location_metadata = _compute_expert_location_metadata(logical_count)
        await self._tokenizer_manager.update_expert_location_metadata(expert_location_metadata)


def _compute_expert_location_metadata(logical_count):
    physical_to_logical_map, logical_to_physical_map, expert_count = deepseek_eplb.rebalance_experts(
        weight=logical_count,
        num_replicas=TODO,
        num_groups=TODO,
        num_nodes=TODO,
        num_gpus=TODO,
    )
    return TODO
