"""Benchmark offline inference throughput using SGLang."""
import time
import dataclasses
import pandas as pd
import torch
from typing import List, Tuple
import sglang as sgl
from sglang.srt.server_args import ServerArgs

def get_engine_instance():
    server_args = ServerArgs(
        model_path="/scratch/bingxche/deepseek-v3",
        tp_size=8,
        enable_ep_moe=True,
        trust_remote_code=True, 
    )
    return sgl.Engine(**dataclasses.asdict(server_args))

def sample_requests_moe():
    processed_data = pd.read_pickle("/scratch/bingxche/data/09292024_mixtral_15k_mintoken2_v1.pkl").iloc[0:100]
    
    prompts: List[Tuple[int, int, int]] = []
    for idx, request in processed_data.iterrows():
        prompts.append((request['tok_input'], request['tok_input_len'], request['tok_ref_output_len']))
        prompts.append((request['tok_input'], request['tok_input_len'], request['tok_ref_output_len']))
        prompts.append((request['tok_input'], request['tok_input_len'], request['tok_ref_output_len']))
        prompts.append((request['tok_input'], request['tok_input_len'], request['tok_ref_output_len']))
   
    # sort prompts by descending length
    sorted_prompts = sorted(prompts, key=lambda x: x[1], reverse=True)
    return sorted_prompts

def run_sglang(prompts):
    engine = get_engine_instance()
    start = time.perf_counter()
    
    # Convert token IDs to input_ids format for SGLang
    input_ids = [prompt[0] for prompt in prompts]
    
    # Configure sampling parameters for SGLang
    sgl_sampling_params = {
        "temperature": 1.0,
        "top_k": 1,
        "top_p": 0.001,
        "max_new_tokens": 1024,
        "ignore_eos": False,
    }
    
    # Generate responses in batch
    outputs = engine.generate(
        input_ids=input_ids,
        sampling_params=sgl_sampling_params
    )
    
    end = time.perf_counter()
    
    # Extract output tokens from responses
    output_prompts = []
    if isinstance(outputs, list):
        # Batch generation case
        for output in outputs:
            output_prompts.append(output.get("output_ids", output.get("token_ids", [])))
    else:
        # Single generation case
        output_prompts.append(outputs.get("output_ids", outputs.get("token_ids", [])))
    
    print("time:", end - start)
    out = [len(a) for a in output_prompts]
    print("output prompt lens:", sum(out) / len(out))
    print("input and output prompts:", output_prompts[0])
    
    
    # # Add this for debugging
    # print("------------------Output structure sample:", outputs[0] if isinstance(outputs, list) else outputs)
    # print("------------------Available keys:", outputs[0].keys() if isinstance(outputs, list) else outputs.keys())
    
    # Clean up
    engine.shutdown()
    
    return (prompts, output_prompts, end - start)

def main(enable_profiling: bool = False):
    requests = sample_requests_moe()
    
    if enable_profiling:
        # SGLang profiling
        engine = get_engine_instance()
        engine.start_profile()
        prompts = [prompt[0] for prompt in requests]
        sgl_sampling_params = {
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 0.001,
            "max_new_tokens": 1024,
            "ignore_eos": False,
        }
        engine.generate(input_ids=prompts, sampling_params=sgl_sampling_params)
        engine.stop_profile()
        engine.shutdown()
    else:
        prompts, output_prompts, elapsed_time = run_sglang(requests)
        assert len(prompts) == len(output_prompts), "prompt input and output lengths are different"
        total_num_tokens = sum(len(prompts[idx][0]) + len(output_prompts[idx]) for idx in range(0, len(prompts)))
        total_output_tokens = sum(len(output_prompt) for output_prompt in output_prompts)
        print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
              f"{total_num_tokens / elapsed_time:.2f} total tokens/s, "
              f"{total_output_tokens / elapsed_time:.2f} output tokens/s")

if __name__ == "__main__":
    main(True)
