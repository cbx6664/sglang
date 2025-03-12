<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="logo" width="400" margin="10px"></img>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://img.shields.io/pypi/dm/sglang)
[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![](https://img.shields.io/badge/Gurubase-(experimental)-006BFF)](https://gurubase.io/g/sglang)

</div>

--------------------------------------------------------------------------------

| [**Blog**](https://lmsys.org/blog/2024-07-25-sglang-llama3/)
| [**Documentation**](https://docs.sglang.ai/)
| [**Join Slack**](https://slack.sglang.ai/)
| [**Join Bi-Weekly Development Meeting**](https://meeting.sglang.ai/)
| [**Roadmap**](https://github.com/sgl-project/sglang/issues/4042)
| [**Slides**](https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#slides) |

## News
- [2025/01] 🔥 SGLang provides day one support for DeepSeek V3/R1 models on NVIDIA and AMD GPUs with DeepSeek-specific optimizations. ([instructions](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3), [AMD blog](https://www.amd.com/en/developer/resources/technical-articles/amd-instinct-gpus-power-deepseek-v3-revolutionizing-ai-development-with-sglang.html), [10+ other companies](https://x.com/lmsysorg/status/1887262321636221412))
- [2024/12] 🔥 v0.4 Release: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs ([blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)).
- [2024/09] v0.3 Release: 7x Faster DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image/Video LLaVA-OneVision ([blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)).
- [2024/07] v0.2 Release: Faster Llama3 Serving with SGLang Runtime (vs. TensorRT-LLM, vLLM) ([blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)).

<details>
<summary>More</summary>

- [2024/10] The First SGLang Online Meetup ([slides](https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#the-first-sglang-online-meetup)).
- [2024/02] SGLang enables **3x faster JSON decoding** with compressed finite state machine ([blog](https://lmsys.org/blog/2024-02-05-compressed-fsm/)).
- [2024/01] SGLang provides up to **5x faster inference** with RadixAttention ([blog](https://lmsys.org/blog/2024-01-17-sglang/)).
- [2024/01] SGLang powers the serving of the official **LLaVA v1.6** release demo ([usage](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#demo)).

</details>

## About
SGLang is a fast serving framework for large language models and vision language models.
It makes your interaction with models faster and more controllable by co-designing the backend runtime and frontend language.
The core features include:

- **Fast Backend Runtime**: Provides efficient serving with RadixAttention for prefix caching, jump-forward constrained decoding, overhead-free CPU scheduler, continuous batching, token attention (paged attention), tensor parallelism, FlashInfer kernels, chunked prefill, and quantization (FP8/INT4/AWQ/GPTQ).
- **Flexible Frontend Language**: Offers an intuitive interface for programming LLM applications, including chained generation calls, advanced prompting, control flow, multi-modal inputs, parallelism, and external interactions.
- **Extensive Model Support**: Supports a wide range of generative models (Llama, Gemma, Mistral, QWen, DeepSeek, LLaVA, etc.), embedding models (e5-mistral, gte, mcdse) and reward models (Skywork), with easy extensibility for integrating new models.
- **Active Community**: SGLang is open-source and backed by an active community with industry adoption.

## Getting Started
- [Install SGLang](https://docs.sglang.ai/start/install.html)
- [Quick Start](https://docs.sglang.ai/backend/send_request.html)
- [Backend Tutorial](https://docs.sglang.ai/backend/openai_api_completions.html)
- [Frontend Tutorial](https://docs.sglang.ai/frontend/frontend.html)
- [Contribution Guide](https://docs.sglang.ai/references/contribution_guide.html)

## Benchmark and Performance
Learn more in the release blogs: [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/), [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/), [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)

## Roadmap
[Development Roadmap (2025 H1)](https://github.com/sgl-project/sglang/issues/4042)

## Adoption and Sponsorship
The project has been deployed to large-scale production, generating trillions of tokens every day.
It is supported by the following institutions: AMD, Atlas Cloud, Baseten, Cursor, DataCrunch, Etched, Hyperbolic, Iflytek, Jam & Tea Studios, LinkedIn, LMSYS, Meituan, Nebius, Novita AI, NVIDIA, RunPod, Stanford, UC Berkeley, UCLA, xAI, and 01.AI.

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/main/slides/adoption.png" alt="logo" width="800" margin="10px"></img>

## Contact Us

For enterprises interested in adopting or deploying SGLang at scale, including technical consulting, sponsorship opportunities, or partnership inquiries, please contact us at contact@sglang.ai.

## Acknowledgment and Citation
We learned the design and reused code from the following projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql). Please cite the paper, [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104), if you find the project useful.

# DeepSeek-V3 MoE Token分布分析工具

这个工具集用于分析DeepSeek-V3模型中每层和每个专家(Experts)的token分布情况。通过SGLang框架加载和运行模型，收集MoE层的路由信息，并生成可视化结果。

## 功能特点

- 分析DeepSeek-V3模型中的Mixture of Experts (MoE)路由模式
- 收集每层每个专家接收的token数量
- 生成多种可视化结果：
  - 每层专家使用分布饼图
  - 专家负载热图
  - 全局专家使用频率条形图
- 支持批量处理多个提示语句
- 支持自定义分析参数

## 安装依赖

```bash
pip install sglang torch matplotlib numpy
```

## 文件说明

- `moe_hooks.py`: 包含MoE监控钩子的实现，用于捕获路由信息
- `analyze_deepseek_v3_moe.py`: 主分析脚本，用于加载模型、收集数据和生成可视化结果
- `analyze_deepseek_moe.py`: 简化版分析脚本

## 使用方法

### 基本用法

```bash
python analyze_deepseek_v3_moe.py --model deepseek-ai/deepseek-v3-chat
```

### 高级用法

```bash
python analyze_deepseek_v3_moe.py \
  --model deepseek-ai/deepseek-v3-chat \
  --output-dir my_analysis_results \
  --prompts-file my_prompts.txt \
  --batch-size 4 \
  --topk 2 \
  --topk-group 2 \
  --num-expert-group 8
```

### 参数说明

- `--model`: 模型路径或Huggingface模型ID (默认: deepseek-ai/deepseek-v3-chat)
- `--output-dir`: 分析结果输出目录 (默认: moe_analysis_results)
- `--prompts-file`: 包含提示的文本文件，每行一个提示
- `--batch-size`: 批处理大小 (默认: 1)
- `--topk`: 模型使用的专家数量 (默认: 2)
- `--topk-group`: 模型使用的专家组数量 (默认: 2)
- `--num-expert-group`: 每组专家数量 (默认: 8)

## 准备提示文件

创建一个文本文件，每行包含一个提示语句：

```text
请解释一下混合专家模型(MoE)的工作原理。
人工智能的未来发展趋势是什么？
写一篇关于气候变化的短文。
如何使用Python实现快速排序算法？
解释量子计算的基本原理。
```

## 输出结果

分析完成后，输出目录将包含以下文件：

1. `routing_summary.json`: 包含所有路由统计信息的JSON文件
2. 每层专家分布饼图: `{layer_name}_pie.png`
3. 专家负载热图: `expert_load_heatmap.png`
4. 全局专家使用频率图: `expert_total_distribution.png`

## DeepSeek-V3 MoE结构说明

DeepSeek-V3采用了分组双向路由策略(Grouped Bi-directional Routing)的MoE结构：

- 使用`biased_grouped_topk`函数进行路由
- 每个token选择topk个专家
- 专家被分成多个组，每个token只能从topk_group个组中选择专家
- 每组包含num_expert_group个专家

## 技术细节

DeepSeek-V3的路由机制使用了SGLang中的`biased_grouped_topk`函数，该函数特点：

1. 使用sigmoid激活而非softmax
2. 添加了correction_bias以平衡专家负载
3. 采用分组选择策略，先选择专家组，再在组内选择专家

## 注意事项

- 分析大量提示可能需要较长时间
- 需要足够的GPU内存加载完整的DeepSeek-V3模型
- 结果可能因模型版本而略有不同

## 参考资料

- [DeepSeek-V3论文](https://arxiv.org/abs/2407.25833)
- [SGLang框架](https://github.com/sgl-project/sglang)
- [Mixture of Experts简介](https://arxiv.org/abs/2101.03961)
