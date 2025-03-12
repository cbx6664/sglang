import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import seaborn as sns

class MoEAnalyzer:
    def __init__(self):
        self.layer_distributions: Dict[str, List[torch.Tensor]] = {}
        self.total_tokens = 0
        self.total_batches = 0
    
    def hook_fn(self, layer_name: str, tokens_per_expert: torch.Tensor):
        """记录每个MoE层的token分布"""
        if layer_name not in self.layer_distributions:
            self.layer_distributions[layer_name] = []
        
        # 保存当前batch的分布数据
        self.layer_distributions[layer_name].append(tokens_per_expert.detach().cpu())
        self.total_tokens += tokens_per_expert.sum().item()
        self.total_batches += 1
    
    def plot_distribution(self, save_dir: str = None):
        """绘制所有层的分布情况"""
        num_layers = len(self.layer_distributions)
        fig, axes = plt.subplots(num_layers, 1, figsize=(15, 5*num_layers))
        
        for idx, (layer_name, distributions) in enumerate(self.layer_distributions.items()):
            # 计算平均分布
            avg_dist = torch.stack(distributions).float().mean(0)
            std_dist = torch.stack(distributions).float().std(0)
            
            ax = axes[idx] if num_layers > 1 else axes
            
            # 绘制条形图
            sns.barplot(x=range(len(avg_dist)), y=avg_dist, yerr=std_dist, ax=ax)
            ax.set_title(f"{layer_name} Token Distribution")
            ax.set_xlabel("Expert ID")
            ax.set_ylabel("Average Tokens")
            
            # 添加统计信息
            stats = f"mean={avg_dist.mean():.1f}, std={std_dist.mean():.1f}\n"
            stats += f"min={avg_dist.min():.1f}, max={avg_dist.max():.1f}"
            ax.text(0.95, 0.95, stats, transform=ax.transAxes, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/moe_distribution.png")
        plt.show()
    
    def print_statistics(self):
        """打印详细的统计信息"""
        print(f"\n总处理tokens: {self.total_tokens}")
        print(f"总batch数: {self.total_batches}")
        
        for layer_name, distributions in self.layer_distributions.items():
            avg_dist = torch.stack(distributions).float().mean(0)
            std_dist = torch.stack(distributions).float().std(0)
            
            print(f"\n{layer_name} 统计信息:")
            print(f"平均每个expert处理的token数: {avg_dist.mean():.2f} ± {std_dist.mean():.2f}")
            print(f"最大值: {avg_dist.max():.2f}")
            print(f"最小值: {avg_dist.min():.2f}")
            
            # 计算负载均衡指标
            n = len(avg_dist)
            fairness = avg_dist.sum()**2 / (n * (avg_dist**2).sum())
            print(f"负载均衡指标(Jain's fairness): {fairness:.4f}")