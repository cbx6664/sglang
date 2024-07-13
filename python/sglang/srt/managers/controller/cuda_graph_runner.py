"""Run the model with cuda graph."""

import bisect

import torch
from vllm.distributed.parallel_state import graph_capture

from sglang.global_config import global_config
from sglang.srt.layers.logits_processor import LogitProcessorOutput
from sglang.srt.managers.controller.infer_batch import (
    Batch, ForwardMode, InputMetadata, init_flashinfer_args
)


class CudaGraphRunner:
    def __init__(self, model_runner, max_batch_size_to_capture):
        self.model_runner = model_runner
        self.graphs = {}
        self.input_buffers = {}
        self.output_buffers = {}
        self.flashinfer_handlers = {}
        self.graph_memory_pool = None

        # Common inputs
        self.max_bs = max_batch_size_to_capture
        self.input_ids = torch.zeros((self.max_bs,), dtype=torch.int32, device="cuda")
        self.req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int32, device="cuda")
        self.seq_lens = torch.ones((self.max_bs,), dtype=torch.int32, device="cuda")
        self.position_ids_offsets = torch.zeros((self.max_bs,), dtype=torch.int32, device="cuda")
        self.out_cache_loc = torch.zeros((self.max_bs,), dtype=torch.int32, device="cuda")

        # FlashInfer inputs
        self.flashinfer_workspace_buffer = self.model_runner.flashinfer_workspace_buffers[0]
        self.flashinfer_kv_indptr = torch.zeros(
            (self.max_bs + 1,), dtype=torch.int32, device="cuda"
        )
        self.flashinfer_kv_indices = torch.zeros(
            (self.max_bs * model_runner.model_config.context_len,), dtype=torch.int32, device="cuda"
        )
        self.flashinfer_kv_last_page_len = torch.ones(
            (self.max_bs,), dtype=torch.int32, device="cuda"
        )

    def can_run(self, batch_size):
        return batch_size < self.max_bs

    def capture(self, batch_size_list):
        self.batch_size_list = batch_size_list
        with graph_capture() as graph_capture_context:
            self.stream = graph_capture_context.stream
            for bs in batch_size_list:
                graph, input_buffers, output_buffers, flashinfer_handler = self.capture_one_batch_size(bs)
                self.graphs[bs] = graph
                self.input_buffers[bs] = input_buffers
                self.output_buffers[bs] = output_buffers
                self.flashinfer_handlers[bs] = flashinfer_handler

    def capture_one_batch_size(self, bs):
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper
        from flashinfer.decode import _grouped_size_compiled_for_decode_kernels

        graph = torch.cuda.CUDAGraph()
        stream = self.stream

        # Common inputs
        input_ids = self.input_ids[:bs]
        req_pool_indices = self.req_pool_indices[:bs]
        seq_lens = self.seq_lens[:bs]
        position_ids_offsets = self.position_ids_offsets[:bs]
        out_cache_loc = self.out_cache_loc[:bs]

        # FlashInfer inputs
        if not _grouped_size_compiled_for_decode_kernels(
            self.model_runner.model_config.num_attention_heads // self.model_runner.tp_size,
            self.model_runner.model_config.get_num_kv_heads(self.model_runner.tp_size),
        ):
            use_tensor_cores = True
        else:
            use_tensor_cores = False
        flashinfer_decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.flashinfer_workspace_buffer, "NHD",
            use_cuda_graph=True,
            use_tensor_cores=use_tensor_cores,
            paged_kv_indptr_buffer=self.flashinfer_kv_indptr[:bs+1],
            paged_kv_indices_buffer=self.flashinfer_kv_indices,
            paged_kv_last_page_len_buffer=self.flashinfer_kv_last_page_len[:bs],
        )
        init_flashinfer_args(
            ForwardMode.DECODE,
            self.model_runner,
            req_pool_indices,
            seq_lens,
            None,
            flashinfer_decode_wrapper,
        )

        # Run and capture
        def run_once():
            input_metadata = InputMetadata.create(
                self.model_runner,
                forward_mode=ForwardMode.DECODE,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                prefix_lens=None,
                position_ids_offsets=position_ids_offsets,
                out_cache_loc=out_cache_loc,
                out_cache_cont_start=None,
                out_cache_cont_end=None,
                return_logprob=False,
                top_logprobs_nums=0,
                skip_flashinfer_init=True,
            )
            input_metadata.flashinfer_decode_wrapper = flashinfer_decode_wrapper
            return self.model_runner.model.forward(
                input_ids, input_metadata.positions, input_metadata
            )

        for _ in range(2):
            run_once()

        torch.cuda.synchronize()
        with torch.cuda.graph(graph, pool=self.graph_memory_pool, stream=stream):
            out = run_once()
        torch.cuda.synchronize()
        self.graph_memory_pool = graph.pool()
        return graph, None, out, flashinfer_decode_wrapper

    def replay(self, batch: Batch):
        assert batch.out_cache_loc is not None
        assert not batch.return_logprob
        raw_bs = len(batch.reqs)

        # Pad
        index = bisect.bisect_left(self.batch_size_list, raw_bs)
        bs = self.batch_size_list[index]
        if bs != raw_bs:
            self.seq_lens.fill_(1)
            self.out_cache_loc.zero_()

        # Common inputs
        self.input_ids[:raw_bs] = batch.input_ids
        self.req_pool_indices[:raw_bs] = batch.req_pool_indices
        self.seq_lens[:raw_bs] = batch.seq_lens
        self.position_ids_offsets[:raw_bs] = batch.position_ids_offsets
        self.out_cache_loc[:raw_bs] = batch.out_cache_loc

        # FlashInfer inputs
        init_flashinfer_args(
            ForwardMode.DECODE,
            self.model_runner,
            self.req_pool_indices[:bs],
            self.seq_lens[:bs],
            None,
            self.flashinfer_handlers[bs],
        )

        # Replay
        self.graphs[bs].replay()
        output = self.output_buffers[bs]

        # Unpad
        if bs == raw_bs:
            return output
        else:
            output = LogitProcessorOutput(
                next_token_logits=output.next_token_logits[:raw_bs],
                next_token_logprobs=output.next_token_logprobs[:raw_bs] if output.next_token_logprobs is not None else None,
                normalized_prompt_logprobs=None,
                prefill_token_logprobs=None,
                prefill_top_logprobs=None,
                decode_top_logprobs=output.decode_top_logprobs[:raw_bs] if output.decode_top_logprobs is not None else None,
            )
        return output