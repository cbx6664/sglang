#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeepSeek-V3 MoE Token分布分析脚本

此脚本用于分析DeepSeek-V3模型中每层和每个专家的token分布情况。
使用SGLang框架加载和运行模型，并收集MoE层的路由信息。
"""

import os
import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict, Any
import time
import sys

# 添加当前目录到模块搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入SGLang相关模块
try:
    # 修正导入路径
    from sglang.srt.model_executor.model_runner import ModelRunner
    from moe_hooks import MoERoutingMonitor, patch_deepseek_select_experts, restore_select_experts
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装sglang: pip install sglang")
    print("请确保moe_hooks.py在当前目录下")
    sys.exit(1)

class DeepSeekMoEAnalyzer:
    def __init__(
        self, 
        model_path: str, 
        output_dir: str = "moe_analysis_results",
        topk: int = 2,
        topk_group: int = 2,
        num_expert_group: int = 8
    ):
        """
        初始化DeepSeek-V3 MoE分析器
        
        Args:
            model_path: 模型路径或Huggingface模型ID
            output_dir: 输出目录
            topk: 模型使用的专家数量 (通常为2)
            topk_group: 模型使用的专家组数量
            num_expert_group: 每组专家数量
        """
        self.model_path = model_path
        self.output_dir = output_dir
        self.topk = topk
        self.topk_group = topk_group
        self.num_expert_group = num_expert_group
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化路由监控器
        self.routing_monitor = MoERoutingMonitor()
        
        # 加载模型
        print(f"正在加载模型: {model_path}")
        self.model_runner = ModelRunner.from_pretrained(
            model_path,
            tp_size=1,  # 单GPU
            max_model_len=4096,
            dtype="auto",
        )
        
        # 注册钩子
        print("注册MoE监控钩子...")
        self.routing_monitor.register_hooks(self.model_runner.model)
        
        # 或者使用函数替换方法
        self.routing_stats = patch_deepseek_select_experts(self.model_runner.model)
        
        print("MoE分析器初始化完成")
    
    def run_analysis(self, prompts: List[str], batch_size: int = 1):
        """
        运行分析
        
        Args:
            prompts: 提示列表
            batch_size: 批处理大小
        """
        print(f"开始分析 {len(prompts)} 个提示...")
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            print(f"处理批次 {i//batch_size + 1}/{(len(prompts)-1)//batch_size + 1}")
            
            # 运行推理
            with torch.inference_mode():
                for prompt in batch_prompts:
                    # 只生成1个token，我们主要关注编码器部分的路由
                    outputs = self.model_runner.generate(
                        prompt=prompt,
                        sampling_params={
                            "max_tokens": 1,
                            "temperature": 0,
                        }
                    )
        
        # 汇总统计信息
        print("生成汇总统计信息...")
        summary = self.routing_monitor.summarize_routing()
        
        # 保存统计信息
        summary_path = os.path.join(self.output_dir, "routing_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"统计信息已保存到: {summary_path}")
        
        # 可视化结果
        self._visualize_results(summary)
        
        # 清理钩子
        self.routing_monitor.remove_hooks()
        restore_select_experts()
        
        return summary
    
    def _visualize_results(self, summary: Dict[str, Any]):
        """可视化路由统计信息"""
        print("生成可视化结果...")
        
        # 1. 为每层创建专家分布饼图
        for layer_name, layer_data in summary.items():
            if layer_name == "expert_load" or not isinstance(layer_data, dict):
                continue
                
            expert_dist = layer_data.get("expert_distribution", {})
            if not expert_dist:
                continue
                
            # 创建饼图
            plt.figure(figsize=(10, 8))
            experts = list(expert_dist.keys())
            counts = list(expert_dist.values())
            
            plt.pie(
                counts, 
                labels=[f"专家{e}" for e in experts],
                autopct='%1.1f%%',
                shadow=True,
                startangle=90
            )
            plt.axis('equal')
            plt.title(f"{layer_name} 层专家分布")
            
            pie_path = os.path.join(self.output_dir, f"{layer_name}_pie.png")
            plt.savefig(pie_path)
            plt.close()
        
        # 2. 创建专家负载热图
        expert_load = summary.get("expert_load", {})
        if expert_load:
            # 提取所有层和专家
            all_layers = list(expert_load.keys())
            all_experts = set()
            for layer_experts in expert_load.values():
                all_experts.update(layer_experts.keys())
            all_experts = sorted(list(all_experts), key=lambda x: int(x))
            
            # 创建热图数据
            heatmap_data = np.zeros((len(all_layers), len(all_experts)))
            for i, layer in enumerate(all_layers):
                for j, expert in enumerate(all_experts):
                    heatmap_data[i, j] = expert_load[layer].get(str(expert), 0)
            
            plt.figure(figsize=(15, 10))
            plt.imshow(heatmap_data, cmap='viridis')
            plt.colorbar(label='Token数量')
            plt.title('DeepSeek-V3 MoE负载热图')
            plt.xlabel('专家ID')
            plt.ylabel('层')
            plt.xticks(np.arange(len(all_experts)), all_experts)
            plt.yticks(np.arange(len(all_layers)), all_layers)
            
            heat_path = os.path.join(self.output_dir, "expert_load_heatmap.png")
            plt.savefig(heat_path)
            plt.close()
        
        # 3. 创建专家偏好条形图 - 按层聚合
        if expert_load:
            expert_totals = defaultdict(int)
            for layer_experts in expert_load.values():
                for expert_id, count in layer_experts.items():
                    expert_totals[expert_id] += count
            
            plt.figure(figsize=(12, 8))
            experts = sorted(list(expert_totals.keys()), key=lambda x: int(x))
            counts = [expert_totals[e] for e in experts]
            
            plt.bar(range(len(experts)), counts)
            plt.title('DeepSeek-V3 全局专家使用频率')
            plt.xlabel('专家ID')
            plt.ylabel('总Token数量')
            plt.xticks(range(len(experts)), experts)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            bar_path = os.path.join(self.output_dir, "expert_total_distribution.png")
            plt.savefig(bar_path)
            plt.close()
            
        print(f"可视化结果已保存到: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="DeepSeek-V3 MoE Token分布分析工具")
    parser.add_argument(
        "--model", 
        type=str, 
        default="deepseek-ai/deepseek-v3-chat",
        help="模型路径或Huggingface模型ID"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="moe_analysis_results",
        help="分析结果输出目录"
    )
    parser.add_argument(
        "--prompts-file", 
        type=str, 
        help="包含提示的文本文件，每行一个提示"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=1,
        help="批处理大小"
    )
    parser.add_argument(
        "--topk", 
        type=int, 
        default=2,
        help="模型使用的专家数量"
    )
    parser.add_argument(
        "--topk-group", 
        type=int, 
        default=2,
        help="模型使用的专家组数量"
    )
    parser.add_argument(
        "--num-expert-group", 
        type=int, 
        default=8,
        help="每组专家数量"
    )
    
    args = parser.parse_args()
    
    # 准备提示
    prompts = []
    if args.prompts_file and os.path.exists(args.prompts_file):
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        # 默认提示
        prompts = [
            "请解释一下混合专家模型(MoE)的工作原理。",
            "人工智能的未来发展趋势是什么？",
            "写一篇关于气候变化的短文。",
            "如何使用Python实现快速排序算法？",
            "解释量子计算的基本原理。"
        ]
    
    # 创建分析器
    analyzer = DeepSeekMoEAnalyzer(
        model_path=args.model,
        output_dir=args.output_dir,
        topk=args.topk,
        topk_group=args.topk_group,
        num_expert_group=args.num_expert_group
    )
    
    # 运行分析
    start_time = time.time()
    analyzer.run_analysis(prompts, batch_size=args.batch_size)
    end_time = time.time()
    
    print(f"分析完成！总耗时: {end_time - start_time:.2f}秒")
    print(f"结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main() 