# SGLang Internal

## Background

This repo is based on SGLang v0.4.5 https://github.com/sgl-project/sglang/tree/v0.4.5



## Features

- [x] Records token distribution on experts of Mixtral 8x7b, and DeepSeek-v3, and outputs as csv files
- [x]  Records experts-gpu allocation in each layer on Mixtral 8x7b, and outputs to csv files
- [x]  Supports custom experts allocation on Mixtral 8x7b
- [x] Supports profiling run



## Installation

Please use it on MI300X nodes.

### Clone Repo

clone this repo.

### Run Container 

```bash
docker run -it --ipc=host --cap-add=SYS_PTRACE --network=host --device=/dev/kfd --device=/dev/dri -v /home/username/:/home/username/ -v /destination/of/your/sglang/:/sgl-workspace/sglang --security-opt seccomp=unconfined --group-add video --privileged lmsysorg/sglang:v0.4.5-rocm630
```

**Remember to bind this repo and map it to the SGLang's runtime folder in `/sgl-workspace/sglang` **

Using this command

`-v /destination/of/your/sglang/:/sgl-workspace/sglang`



## Usage

1. Start container

2. Run entry scripts

   Enter the entry scripts and make configurations and run. `misc\entry.py`

### How to record token distribution and experts allocation?

```python
ENV = {
    "custom_expert_allocation": "True",
    "num_experts": "8", # this should match the number of physical experts if we use custom expert allocation
    "log_all": "True", # whether to log expert allocation, token distribution info...
    "log_dir": "/home/bingxche/log/mixtral8x7b_ep4_mixtral_dataset_15_prompts_8_custom_experts", # directory of log files
    # file path of custom_expert_allocation.csv
    "expert_allocation_file_path": "/home/bingxche/log/mixtral8x7b_ep4_mixtral_dataset_15_prompts_vanilla/moe_token_dist_eplb_8replicas_1groups_1nodes_4gpus/phy2log_8replicas_1groups_1nodes_4gpus.json",
}
```

Make sure in `entry.py`, `log_all` is set to "True" and `log_dir` is set to the path you want to store the log files. 



### How to use custom expert allocation?

1. Calculate load balanced experts allocation using DeepSeek-EPLB through `misc\use_eplb.py`, you should already have the token distribution log folder(`log_dir` set in `entry.py`). Just change the `input_folder` to your token distribution log folder, and set parameters in `main()`

   ```python
   def main():
       """Main entry point for the application."""
       # Configuration parameters
       input_folder = r"/home/bingxche/log/mixtral8x7b_ep4_mixtral_dataset_15_prompts_vanilla/moe_token_dist"
       config = {
           'num_replicas': 12,  # Number of physical experts
           'num_groups': 1,    # Number of expert groups
           'num_nodes': 1,     # Number of server nodes
           'num_gpus': 4       # Number of GPUs
       }
       
       # Run the EPLB algorithm
       phy2log, log2phy, logcnt = run_eplb(input_folder, config)
       
       # Export results to JSON files
       output_folder = f"{input_folder}_eplb_{config['num_replicas']}replicas_{config['num_groups']}groups_{config['num_nodes']}nodes_{config['num_gpus']}gpus"
       export_results_to_json(output_folder, phy2log, log2phy, logcnt, config)
   
   ```

2. In `entry.py`, set `expert_allocation_file_path`  and set `"custom_expert_allocation": "True"`,  remember to set `num_experts` to number of physical experts accordingly. 

   For example, although Mixtral 8x7b has 8 experts in each MoE layer, we can replicate some hotter experts to make load more balanced, let's say replicate it to 12 experts. So custom expert allocation list will be in size of [32, 12]  instead of [32, 8]