#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple
import json

# 添加当前目录到模块搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入SGLang相关模块
try:
    from sglang.srt.model_executor.model_runner import ModelRunner
    # 我们将直接使用ModelRunner而不是InferenceEngine
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.models import get_model
    from sglang.srt.model_config import ModelConfig
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装sglang: pip install sglang")
    sys.exit(1)

# 设置环境变量以启用MoE分析
os.environ["MOE_COLLECT_ROUTING_STATS"] = "1"

class MoEAnalyzer:
    def __init__(self, model_path: str):
        """
        初始化MoE分析器
        
        Args:
            model_path: DeepSeek-V3模型路径
        """
        self.model_path = model_path
        self.stats = {}
        
        # 初始化模型 - 使用ModelRunner代替InferenceEngine
        try:
            print(f"正在加载模型: {model_path}")
            
            # 创建ServerArgs实例
            server_args = ServerArgs()
            server_args.model = model_path
            server_args.device = "cuda"
            server_args.max_model_len = 4096
            server_args.dtype = "auto"
            server_args.tp_size = 8  # 使用单GPU
            
            # 获取模型配置
            model_config = get_model(server_args)
            
            # 初始化ModelRunner
            self.model_runner = ModelRunner(
                model_config=model_config,
                mem_fraction_static=0.9,  # 内存分配比例
                gpu_id=0,  # 使用第一个GPU
                tp_rank=0,  # 张量并行rank
                tp_size=8,  # 单GPU运行
                nccl_port=29500,  # NCCL端口
                server_args=server_args,  # 服务器参数
            )
            
            # 加载模型权重
            self.model_runner.load_model()
            
            print("模型加载完成")
        except Exception as e:
            print(f"初始化ModelRunner失败: {e}")
            print("请检查SGLang文档获取当前版本的正确API用法")
            sys.exit(1)
    
    def analyze_routing(self, prompts: List[str], output_file: str = "moe_stats.json"):
        """
        分析一组提示的token路由情况
        
        Args:
            prompts: 要分析的提示列表
            output_file: 输出统计信息的JSON文件
        """
        routing_stats = defaultdict(lambda: defaultdict(int))
        expert_load_stats = defaultdict(lambda: defaultdict(int))
        
        for prompt in prompts:
            # 运行推理
            with torch.inference_mode():
                # 使用ModelRunner进行生成
                # 注意：API可能需要根据实际ModelRunner的接口调整
                # 这里假设ModelRunner有类似的generate方法
                outputs = self.model_runner.generate(
                    prompt=prompt,
                    max_tokens=1,  # 只生成一个token，我们只关注路由
                    temperature=0,
                )
                
                # 从输出中提取路由信息
                # 注意：这部分需要根据SGLang的实际API调整
                if hasattr(outputs, "routing_info"):
                    routing_info = outputs.routing_info
                    self._process_routing_info(routing_info, routing_stats, expert_load_stats)
        
        # 保存统计信息
        stats = {
            "routing": dict(routing_stats),
            "expert_load": dict(expert_load_stats)
        }
        with open(output_file, "w") as f:
            json.dump(stats, f, indent=2)
        
        self.stats = stats
        return stats
    
    def _process_routing_info(self, routing_info, routing_stats, expert_load_stats):
        """处理路由信息"""
        # 这部分实现取决于routing_info的具体结构
        # 假设routing_info是一个字典，键是层名，值是token路由信息
        for layer_name, layer_info in routing_info.items():
            if "topk_ids" in layer_info:
                topk_ids = layer_info["topk_ids"]
                for token_idx, expert_ids in enumerate(topk_ids):
                    for expert_id in expert_ids:
                        routing_stats[layer_name][int(expert_id)] += 1
                        expert_load_stats[layer_name][int(expert_id)] += 1
    
    def visualize(self, output_dir: str = "moe_visualization"):
        """可视化MoE统计信息"""
        if not self.stats:
            raise ValueError("请先运行analyze_routing收集统计信息")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 每层专家使用频率图
        self._plot_expert_distribution(output_dir)
        
        # 2. 专家负载热图
        self._plot_expert_load_heatmap(output_dir)
    
    def _plot_expert_distribution(self, output_dir: str):
        """绘制每层专家分布图"""
        routing_stats = self.stats["routing"]
        
        for layer_name, expert_counts in routing_stats.items():
            plt.figure(figsize=(12, 6))
            experts = list(expert_counts.keys())
            counts = list(expert_counts.values())
            
            plt.bar(experts, counts)
            plt.title(f"层 {layer_name} 的专家使用分布")
            plt.xlabel("专家ID")
            plt.ylabel("被路由的token数量")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            output_path = os.path.join(output_dir, f"{layer_name}_distribution.png")
            plt.savefig(output_path)
            plt.close()
    
    def _plot_expert_load_heatmap(self, output_dir: str):
        """绘制专家负载热图"""
        expert_load = self.stats["expert_load"]
        
        # 创建热图数据
        layers = list(expert_load.keys())
        all_experts = set()
        for layer_experts in expert_load.values():
            all_experts.update(layer_experts.keys())
        all_experts = sorted(list(all_experts))
        
        # 创建热图矩阵
        heatmap_data = np.zeros((len(layers), len(all_experts)))
        for i, layer in enumerate(layers):
            for j, expert in enumerate(all_experts):
                heatmap_data[i, j] = expert_load[layer].get(str(expert), 0)
        
        plt.figure(figsize=(15, 10))
        plt.imshow(heatmap_data, cmap='viridis')
        plt.colorbar(label='Token 数量')
        plt.title('DeepSeek-V3 MoE负载热图')
        plt.xlabel('专家ID')
        plt.ylabel('层')
        plt.xticks(np.arange(len(all_experts)), all_experts)
        plt.yticks(np.arange(len(layers)), layers)
        
        output_path = os.path.join(output_dir, "expert_load_heatmap.png")
        plt.savefig(output_path)
        plt.close()

def main():
    # 替换为实际的DeepSeek-V3模型路径
    model_path = "deepseek-ai/deepseek-v3-chat"
    
    # 创建分析器
    analyzer = MoEAnalyzer(model_path)
    
    # 准备一些测试提示
    prompts = [
        "请解释一下混合专家模型(MoE)的工作原理。",
        "人工智能的未来发展趋势是什么？",
        "写一篇关于气候变化的短文。",
        "如何使用Python实现快速排序算法？",
        "解释量子计算的基本原理。"
    ]
    
    # 分析路由情况
    analyzer.analyze_routing(prompts)
    
    # 可视化结果
    analyzer.visualize()
    
    print("分析完成！结果保存在moe_visualization目录中")

if __name__ == "__main__":
    main() 