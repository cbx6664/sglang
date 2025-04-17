"""Benchmark offline inference throughput using SGLang."""
import time
import dataclasses
import pandas as pd
import torch
from typing import List, Tuple, Dict
import sglang as sgl
from sglang.srt.server_args import ServerArgs
from sglang.srt.sampling.sampling_params import SamplingParams
from pathlib import Path
import os
import logging
import json
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_engine_instance():
    server_args = ServerArgs(
        # model_path="/scratch/bingxche/Mixtral-8x7B-Instruct-v0.1",
        model_path="/home/bingxche/Mixtral-8x7B-Instruct-v0.1",
        # model_path="/home/bingxche/deepseek-v3",
        # model_path="/scratch/bingxche/deepseek-v3",
        tp_size=4,
        # dp_size=8,
        ep_size=4,
        # using "enable_ep_moe" will cause error: Unsupported conversion from 'f8E4M3FN' to 'f16'
        enable_ep_moe=True,
        trust_remote_code=True,
        disable_cuda_graph=True,
    )
    
    os.environ["CUSTOM_EXPERT_ALLOCATION"] = "False"
    os.environ["MODEL_PATH"] = f"{server_args.model_path}"
    os.environ["LOG_ALL"] = "True"
    os.environ["LOG_DIR"] = "/home/bingxche/log/mixtral8x7b_ep4_vanilla_expert_allocation"
    os.environ["NUM_EXPERTS"] = '8'
    # os.environ["EXPERT_ALLOCATION"] = "[[5, 1, 2, 0, 4, 3, 6, 7], [3, 1, 6, 5, 2, 7, 0, 4],[4, 1, 0, 6, 2, 3, 5, 7], [4, 1, 0, 5, 2, 6, 7, 3],[3, 6, 1, 0, 5, 7, 4, 2], [6, 7, 2, 4, 5, 0, 1, 3],[4, 0, 6, 1, 3, 2, 5, 7], [2, 5, 7, 6, 0, 4, 3, 1],[0, 2, 3, 4, 7, 6, 5, 1], [1, 2, 3, 5, 7, 6, 4, 0],[2, 6, 1, 3, 7, 5, 4, 0], [1, 3, 2, 5, 0, 7, 4, 6],[1, 7, 0, 6, 5, 4, 3, 2], [1, 6, 0, 5, 2, 3, 7, 4],[4, 5, 0, 2, 3, 7, 1, 6], [0, 5, 2, 4, 1, 3, 7, 6],[6, 0, 5, 1, 3, 7, 2, 4], [4, 6, 5, 3, 7, 1, 2, 0],[1, 5, 7, 3, 0, 6, 4, 2], [5, 4, 7, 6, 3, 1, 2, 0],[6, 5, 0, 4, 1, 7, 2, 3], [3, 1, 7, 4, 0, 5, 6, 2],[6, 4, 1, 5, 2, 0, 7, 3], [1, 4, 6, 2, 0, 5, 3, 7],[5, 0, 2, 6, 1, 3, 7, 4], [6, 4, 5, 1, 3, 2, 0, 7],[6, 4, 2, 3, 0, 1, 5, 7], [1, 2, 4, 7, 0, 5, 6, 3],[3, 0, 4, 1, 6, 2, 7, 5], [7, 6, 5, 3, 4, 0, 1, 2],[4, 1, 7, 5, 6, 3, 0, 2], [7, 3, 2, 5, 4, 1, 0, 6]]"
    return sgl.Engine(**dataclasses.asdict(server_args))

def sample_requests_moe():
    processed_data = pd.read_pickle("/home/bingxche/data/09292024_mixtral_15k_mintoken2_v1.pkl").head(100)
    # processed_data = pd.read_pickle("/scratch/bingxche/data/09292024_mixtral_15k_mintoken2_v1.pkl").head(100)
    
    prompts: List[Tuple[int, int, int]] = []
    for idx, request in processed_data.iterrows():
        prompts.append((request['tok_input'], request['tok_input_len'], request['tok_ref_output_len']))
        # prompts.append((request['tok_input'], request['tok_input_len'], request['tok_ref_output_len']))
   
    # sort prompts by descending length
    sorted_prompts = sorted(prompts, key=lambda x: x[1], reverse=True)
    return sorted_prompts

# todo Not Supported Yet
def profile_run_sglang(prompts, sampling_params):
    engine = get_engine_instance()
    input_ids = [prompt[0] for prompt in prompts]
    profile_dir = os.path.join(os.environ.get("LOG_DIR"), "trace")
    Path(profile_dir).mkdir(parents=True, exist_ok=True)
    os.environ["SGLANG_TORCH_PROFILER_DIR"] = profile_dir
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir))
    ) as p:
        start = time.perf_counter()
        end = time.perf_counter()
    
    # Try both table formats since different GPU backends use different fields
    print("==== CPU PROFILE ====")
    print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
    print("\n==== GPU PROFILE (CUDA style) ====")
    try:
        print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
    except:
        print("Could not sort by self_cuda_time_total")
    
    print(f"\nTotal execution time: {end - start:.4f} seconds")
    
    outputs = engine.generate(input_ids=input_ids, sampling_params=sampling_params)
    save_outputs_to_json(outputs)
    
    engine.shutdown()

def run_sglang(prompts, sampling_params):
    engine = get_engine_instance()
    start = time.perf_counter()
    # Convert token IDs to input_ids format for SGLang
    input_ids = [prompt[0] for prompt in prompts]
    
    # Generate responses in batch
    outputs = engine.generate(input_ids=input_ids, sampling_params=sampling_params)
    
    end = time.perf_counter()
    
    # Extract output tokens from responses
    output_prompts = []
    if isinstance(outputs, list):
        # Batch generation case
        for output in outputs:
            output_prompts.append(output.get("text", output.get("meta_info", [])))
    else:
        # Single generation case
        output_prompts.append(outputs.get("text", outputs.get("meta_info", [])))
    
    logger.info(f"time: {end - start}")
    out = [len(a) for a in output_prompts]
    logger.info(f"output prompt lens: {sum(out) / len(out)}")
    logger.info(f"input and output prompts: {output_prompts[0]}")
    
    save_outputs_to_json(outputs)
    
    # Clean up
    engine.shutdown()
    
    return (prompts, output_prompts, end - start)

def save_outputs_to_json(outputs):
    result = []
    for i, output in enumerate(outputs):
        item = {}

        # text
        item["text"] = output.get("text", "")

        # meta info
        meta = output.get("meta_info", {})
        item["id"] = meta.get("id", f"unknown_{i}")
        item["prompt_tokens"] = meta.get("prompt_tokens", None)
        item["completion_tokens"] = meta.get("completion_tokens", None)
        item["cached_tokens"] = meta.get("cached_tokens", None)
        item["e2e_latency"] = meta.get("e2e_latency", None)

        result.append(item)

    path = os.path.join(os.environ.get("LOG_DIR"), "model_output.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(result)} entries to {path}")


def main(enable_profiling: bool = False):
    requests = sample_requests_moe()
    prompts = requests
    
    sampling_params = {
        "temperature": 1.0,
        "top_k": 1,
        "top_p": 0.001,
        "max_new_tokens": 1024,
        "ignore_eos": False,
    }
    
    if enable_profiling:
        # Use PyTorch profiler directly instead of SGLang's problematic profiling interface
        profile_run_sglang(prompts, sampling_params)
    else:
        prompts, output_prompts, elapsed_time = run_sglang(requests, sampling_params)
        assert len(prompts) == len(output_prompts), "prompt input and output lengths are different"
        total_num_tokens = sum(len(prompts[idx][0]) + len(output_prompts[idx]) for idx in range(0, len(prompts)))
        total_output_tokens = sum(len(output_prompt) for output_prompt in output_prompts)
        logger.info(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
              f"{total_num_tokens / elapsed_time:.2f} total tokens/s, "
              f"{total_output_tokens / elapsed_time:.2f} output tokens/s")
        
if __name__ == "__main__":
    main(False)  # Set to True to enable profiling