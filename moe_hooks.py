#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple
import sys
import inspect

class MoERoutingMonitor:
    """监控DeepSeek-V3 MoE层的路由信息"""
    
    def __init__(self):
        self.routing_stats = defaultdict(list)
        self.hooks = []
        self.model = None
    
    def register_hooks(self, model):
        """注册钩子到模型的MoE层
        
        Args:
            model: 加载的DeepSeek-V3模型
        """
        self.model = model
        self.hooks = []
        
        # 清除现有统计信息
        self.routing_stats = defaultdict(list)
        
        # 遍历模型的所有模块，查找MoE相关层
        for name, module in model.named_modules():
            if "moe" in name.lower() and hasattr(module, "router"):
                # 为路由器前向传播添加钩子
                hook = module.router.register_forward_hook(
                    lambda _, inputs, outputs, layer_name=name:
                    self._router_hook(inputs, outputs, layer_name)
                )
                self.hooks.append(hook)
                
                # 为专家选择函数添加钩子
                if hasattr(module, "experts"):
                    # 假设select_experts是在MoE模块内调用的
                    # 为每个专家添加钩子来跟踪哪些token被路由到该专家
                    for expert_idx, expert in enumerate(module.experts):
                        expert_hook = expert.register_forward_hook(
                            lambda _, inputs, outputs, layer_name=name, expert_id=expert_idx:
                            self._expert_hook(inputs, outputs, layer_name, expert_id)
                        )
                        self.hooks.append(expert_hook)
    
    def _router_hook(self, inputs, outputs, layer_name):
        """路由器钩子，捕获topk_weights和topk_ids"""
        # 假设输出是一个包含topk_weights和topk_ids的元组
        # 适应实际的输出格式
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            topk_weights, topk_ids = outputs[0], outputs[1]
            
            # 保存路由信息
            self.routing_stats[layer_name].append({
                "topk_weights": topk_weights.detach().cpu().numpy(),
                "topk_ids": topk_ids.detach().cpu().numpy()
            })
    
    def _expert_hook(self, inputs, outputs, layer_name, expert_id):
        """专家钩子，捕获传递给每个专家的tokens"""
        # 这里仅记录有哪些输入被路由到这个专家
        # 对于输入，我们只关心数量而不是具体内容
        if isinstance(inputs, tuple) and len(inputs) > 0:
            # 假设第一个输入是hidden_states
            num_tokens = inputs[0].shape[0] if inputs[0].dim() >= 2 else 0
            
            if f"{layer_name}_expert_load" not in self.routing_stats:
                self.routing_stats[f"{layer_name}_expert_load"] = []
            
            self.routing_stats[f"{layer_name}_expert_load"].append({
                "expert_id": expert_id,
                "num_tokens": num_tokens
            })
    
    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_stats(self):
        """获取收集的统计信息"""
        return dict(self.routing_stats)
    
    def summarize_routing(self):
        """汇总路由统计信息"""
        summary = {}
        
        for layer_name, stats_list in self.routing_stats.items():
            if "_expert_load" in layer_name:
                continue  # 单独处理专家负载
                
            # 汇总topk_ids分布
            expert_counts = defaultdict(int)
            token_count = 0
            
            for stats in stats_list:
                topk_ids = stats["topk_ids"]
                token_count += topk_ids.shape[0]
                
                for token_idx in range(topk_ids.shape[0]):
                    for expert_idx in topk_ids[token_idx]:
                        expert_counts[int(expert_idx)] += 1
            
            summary[layer_name] = {
                "expert_distribution": dict(expert_counts),
                "total_tokens": token_count
            }
        
        # 汇总专家负载
        expert_load = {}
        for layer_name, stats_list in self.routing_stats.items():
            if not "_expert_load" in layer_name:
                continue
                
            original_layer = layer_name.replace("_expert_load", "")
            expert_load[original_layer] = defaultdict(int)
            
            for stats in stats_list:
                expert_id = stats["expert_id"]
                num_tokens = stats["num_tokens"]
                expert_load[original_layer][expert_id] += num_tokens
        
        summary["expert_load"] = dict(expert_load)
        return summary

# 修改DeepSeek-V3的select_experts函数的辅助函数
def patch_deepseek_select_experts(model):
    """
    修补DeepSeek-V3的select_experts函数以收集路由信息
    这是一个示例函数，实际实现可能需要根据模型结构调整
    
    Args:
        model: 加载的DeepSeek-V3模型
    """
    try:
        # 修正导入路径
        from sglang.srt.layers.moe.topk import select_experts as original_select_experts
    except ImportError:
        print("无法导入select_experts函数，请确认SGLang版本或查看源码获取正确路径")
        print("可能需要使用以下命令查找正确路径:")
        print("  find /path/to/sglang -name '*.py' | xargs grep -l 'select_experts'")
        return {}
    
    # 统计数据存储
    routing_stats = {}
    
    # 修补后的函数
    def patched_select_experts(
        hidden_states,
        router_logits,
        top_k,
        use_grouped_topk,
        renormalize,
        topk_group=None,
        num_expert_group=None,
        custom_routing_function=None,
        correction_bias=None,
        torch_native=False
    ):
        # 调用原始函数
        topk_weights, topk_ids = original_select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            correction_bias=correction_bias,
            torch_native=torch_native
        )
        
        # 收集统计信息
        # 注意：这里需要一种方法来标识当前调用对应的层
        frame = inspect.currentframe().f_back
        caller_info = inspect.getframeinfo(frame)
        caller_module = caller_info.function
        
        if caller_module not in routing_stats:
            routing_stats[caller_module] = []
        
        routing_stats[caller_module].append({
            "topk_weights": topk_weights.detach().cpu().numpy(),
            "topk_ids": topk_ids.detach().cpu().numpy()
        })
        
        return topk_weights, topk_ids
    
    # 替换函数
    try:
        from sglang.srt.layers.moe import topk
        
        # 保存原始函数
        topk.original_select_experts = original_select_experts
        
        # 替换为修补后的函数
        topk.select_experts = patched_select_experts
    except ImportError:
        print("无法导入topk模块，路径可能已改变")
        return routing_stats
    
    # 返回统计数据存储
    return routing_stats

def restore_select_experts():
    """恢复原始的select_experts函数"""
    try:
        from sglang.srt.layers.moe import topk
        
        if hasattr(topk, "original_select_experts"):
            topk.select_experts = topk.original_select_experts
            delattr(topk, "original_select_experts")
            print("已恢复原始select_experts函数")
    except ImportError:
        print("无法导入topk模块，无法恢复原始函数") 