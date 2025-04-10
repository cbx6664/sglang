# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Callable, Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist
import atexit

from sglang.srt.utils import get_compiler_backend, get_model_name, print_expert_token_dist, use_eplb_to_calculate_experts_gpu_placement
import os
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# # TODO: pass these two variables to the MainProcess, consider the reason of subprocess, sprawning
# _manager = None
# _shared_dict = None
# # Local variables as fallback
_select_experts_call_count = 0
_token_distribution_dict = {}

def get_expert_token_distribution_dict():
    return _token_distribution_dict

def get_select_experts_call_count():
    return _select_experts_call_count

# def initialize_manager():
#     """Initialize the multiprocessing manager only when needed - lazy initialization"""
#     global _manager, _shared_dict
#     if _manager is None:
#         try:
#             _manager = Manager()
#             _shared_dict = _manager.dict({
#                 'select_experts_call_count': 0,
#                 'token_distribution_dict': _manager.dict()
#             })
#         except RuntimeError as e:
#             logger.warning(f"Could not initialize Manager: {e}")
#             logger.warning("Falling back to local variables (will not be shared across processes)")


# def get_token_distribution_dict():
#     global _shared_dict
#     if _shared_dict is not None:
#         return dict(_shared_dict['token_distribution_dict'])
#     return None


# def reset_token_distribution_dict():
#     global _shared_dict, token_distribution_dict
#     token_distribution_dict = {}
#     if _shared_dict is not None:
#         _shared_dict['token_distribution_dict'] = _manager.dict()


# def get_select_experts_call_count():
#     global _shared_dict
#     if _shared_dict is not None:
#         return _shared_dict['select_experts_call_count']
#     return None


# def reset_select_experts_call_count():
#     global _shared_dict, select_experts_call_count
#     select_experts_call_count = 0
#     if _shared_dict is not None:
#         _shared_dict['select_experts_call_count'] = 0


def fused_topk_native(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    assert (
        hidden_states.shape[0] == gating_output.shape[0]
    ), f"Number of tokens mismatch, {hidden_states.shape=} vs {gating_output.shape=}"
    M, _ = hidden_states.shape
    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    topk_weights = F.softmax(gating_output.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(topk_weights, topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    from vllm import _custom_ops as ops

    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    M, _ = hidden_states.shape

    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    token_expert_indicies = torch.empty(
        M, topk, dtype=torch.int32, device=hidden_states.device
    )

    ops.topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indicies,
        gating_output.float(),
    )
    del token_expert_indicies

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids


@torch.compile(dynamic=True, backend=get_compiler_backend())
def grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Scoring function '{scoring_func}' is not supported.")

    num_token = scores.shape[0]
    group_scores = (
        scores.view(num_token, num_expert_group, -1).max(dim=-1).values
    )  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
    topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


# DeepSeek V2/V3/R1 uses biased_grouped_top
@torch.compile(dynamic=True, backend=get_compiler_backend())
def biased_grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    scores = gating_output.sigmoid()
    num_token = scores.shape[0]
    scores_for_choice = scores.view(num_token, -1) + correction_bias.unsqueeze(0)
    group_scores = (
        scores_for_choice.view(num_token, num_expert_group, -1)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores_for_choice.masked_fill(
        ~score_mask.bool(), float("-inf")
    )  # [n, e]
    _, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
    topk_weights = scores.gather(1, topk_ids)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    topk_group: Optional[int] = None,
    num_expert_group: Optional[int] = None,
    custom_routing_function: Optional[Callable] = None,
    correction_bias: Optional[torch.Tensor] = None,
    torch_native: bool = False,
):
    # DeepSeek V2/V3/R1 uses biased_grouped_top
    if use_grouped_topk:
        assert topk_group is not None
        assert num_expert_group is not None
        if correction_bias is None:
            topk_weights, topk_ids = grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
            )
        else:
            topk_weights, topk_ids = biased_grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                correction_bias=correction_bias,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
            )
    elif torch_native and custom_routing_function is None:
        topk_weights, topk_ids = fused_topk_native(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
        )
    elif custom_routing_function is None:
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
        )
    else:
        topk_weights, topk_ids = custom_routing_function(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
        )

    # if use_eplb_to_calculate_experts_gpu_placement:
    #     if dist.get_rank() == 0:
    #         logger.info(f"Using DeepSeek-EPLB to calculate experts-gpu placement.")
    #         if "mixtral" in get_model_name():
    #             flatten_topk_ids = topk_ids.view(-1)
    #             output_dir = "/home/bingxche/trace_dir/moe_token_distribution/mixtral_8x7b_ep8_3"
    #             os.makedirs(output_dir, exist_ok=True)
    #             # from cbx.load_balancer import eplb 
    #             from sglang.srt.models.mixtral import MixtralModel
    #             layer_id_mixtral = MixtralModel.layer_id_print
    #             # Mixtral 8x7B has 8 experts in each layer, totally 32 layers, all are MoE layers
    #             token_dist_per_expert = torch.bincount(flatten_topk_ids, minlength=8)
    #             token_dist_for_each_layer = []
    #             for i in range(32):
    #                 if i == layer_id_mixtral:
    #                     # save to list
    #                     token_dist_for_each_layer.append(token_dist_per_expert.cpu().tolist())
                        
    #                     # save to csv
    #                     csv_path = os.path.join(output_dir, f"layer_{layer_id_mixtral}_token_distribution_rank_{dist.get_rank()}.csv")
    #                     with open(csv_path, "a") as f:
    #                         token_dist = token_dist_per_expert.cpu().tolist()
    #                         f.write(",".join(map(str, token_dist)) + "\n")
        
    if print_expert_token_dist:
        if dist.get_rank() == 0:
            if "mixtral" in get_model_name():
                flatten_topk_ids = topk_ids.view(-1)
                output_dir = f"/home/bingxche/trace_dir/moe_token_distribution/mixtral_8x7b_ep4_2"
                os.makedirs(output_dir, exist_ok=True) 
                from sglang.srt.models.mixtral import MixtralModel
                layer_id_mixtral = MixtralModel.layer_id_print
                
                # Mixtral 8x7B has 8 experts in each layer, totally 32 layers, all are MoE layers
                token_dist_per_expert = torch.bincount(flatten_topk_ids, minlength=8)
                
                global _token_distribution_dict
                if layer_id_mixtral not in _token_distribution_dict:
                    _token_distribution_dict[layer_id_mixtral] = []
                    
                if len(_token_distribution_dict[layer_id_mixtral]) == 0:
                    _token_distribution_dict[layer_id_mixtral].append(token_dist_per_expert.cpu().tolist())
                else:
                    current_sum = _token_distribution_dict[layer_id_mixtral][0]
                    new_dist = token_dist_per_expert.cpu().tolist()
                    _token_distribution_dict[layer_id_mixtral][0] = [current_sum[i] + new_dist[i] for i in range(len(new_dist))]

                global _select_experts_call_count
                _select_experts_call_count += 1
                    
                # # Accumulate the current distribution
                # # If there's already data, add to it element-wise, otherwise store the current distribution
                # if _shared_dict is not None:
                #     shared_token_dict = _shared_dict['token_distribution_dict']
                #     layer_id_str = str(layer_id_mixtral)
                    
                #     if layer_id_str not in shared_token_dict:
                #         shared_token_dict[layer_id_str] = []
                        
                #     # Convert to list for storage
                #     dist_list = token_dist_per_expert.cpu().tolist()
                    
                #     if not shared_token_dict.get(layer_id_str, []):
                #         shared_token_dict[layer_id_str] = [dist_list]
                #     else:
                #         current_sum = shared_token_dict[layer_id_str][0]
                #         shared_token_dict[layer_id_str][0] = [
                #             current_sum[i] + dist_list[i] for i in range(len(dist_list))
                #         ]
                
                
                # local variable 
                
                # Still save to CSV as before
                # csv_path = os.path.join(output_dir, f"layer_{layer_id_mixtral}_token_distribution_rank_{dist.get_rank()}.csv")
                # with open(csv_path, "a") as f:
                #     token_dist = token_dist_per_expert.cpu().tolist()
                #     f.write(",".join(map(str, token_dist)) + "\n")
            
            elif "deepseek-v3" in get_model_name():
                logger.info(f"printing deepseek-v3 token dist")
                flatten_topk_ids = topk_ids.view(-1)
                output_dir = "/home/bingxche/trace_dir/moe_token_distribution/deepseek-v3_tp8_0"
                os.makedirs(output_dir, exist_ok=True) 
                from sglang.srt.models.deepseek_v2 import DeepseekV2Model
                layer_id_deepseek = DeepseekV2Model.layer_id_print  
                # DeepSeek-V3 has 266 shared experts in each layer, totally 61 layers with 58 MoE layers(layer4 - layer61)
                token_dist_per_expert = torch.bincount(flatten_topk_ids, minlength=256)     
                csv_path = os.path.join(output_dir, f"layer_{layer_id_deepseek}_token_distribution.csv")
                with open(csv_path, "a") as f:
                    token_dist = token_dist_per_expert.cpu().tolist()
                    f.write(",".join(map(str, token_dist)) + "\n")

    
    
        
    return topk_weights, topk_ids
