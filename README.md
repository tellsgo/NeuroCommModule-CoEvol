# NeuroCommModule-CoEvol

[![许可证](https://img.shields.io/badge/许可证-MIT-blue.svg)](LICENSE)
[![Python版本](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)

## 项目简介

NeuroCommModule-CoEvol（神经通信模块协同进化）是一个探索大型语言模型中模块化协同进化机制的研究项目。本项目通过在模型特定层添加可学习的通信模块，观察其在交互过程中的演化，模拟脑区功能分化现象，为理解AI系统的协作智能提供微观视角。

## 研究背景与动机

传统的大模型协同训练需要巨大计算资源，本项目转向"局部功能层演化观察"，通过模块化设计实现：
- 资源高效：仅微调极小参数量（约0.1%），适合单卡实验
- 可解释性强：聚焦单一模块变化，更易追踪"语言"能力的涌现过程
- 生物启发：模拟人脑语言区等功能区的演化形成过程

## 核心概念

- **模块化改造**：在模型特定层插入可学习的通信模块（如Adapter或LoRA）
- **进化观察**：冻结模型主干，仅训练通信模块，分析其演化过程
- **通信协议**：定义模块间信息传递的形式与交互规则
- **功能涌现**：观测简单通信如何逐步发展出复杂的语言能力

## 技术实现路径

### 1. 通信模块设计
- 在Transformer中间层（第6-12层）插入轻量级模块
- 支持多种结构：Adapter式、符号生成器、向量通信器

### 2. 通信协议
- 支持离散符号、连续向量、概率分布等多种信号类型
- 实现回合制与流式两种交互方式
- 包含外部奖励与内部一致性的反馈机制

### 3. 观测指标
- 模块参数变化：权重分布、奇异值变化
- 通信内容分析：符号/向量聚类与可视化
- 行为涌现：协作任务表现提升度量

## 实验设计

### 阶段1：最小验证实验
- 简单协作任务（对话生成、合作解题）
- 使用小型模型（TinyLlama-1.1B或Phi-3-mini）
- 对照组设计验证通信模块的有效性

### 阶段2：通信进化分析
- 通信信号可视化和聚类分析
- 干预实验测试系统稳健性
- 跨任务迁移评估泛化能力

### 阶段3：理论扩展
- 多模块协同实验
- 进化动力学数学建模

## 安装指南

```bash
# 克隆仓库
git clone https://github.com/yourusername/NeuroCommModule-CoEvol.git

# 进入项目目录
cd NeuroCommModule-CoEvol

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

```python
# 基础实验示例
from neurocommmodule import ExperimentRunner

# 配置实验参数
config = {
    "base_model": "TinyLlama-1.1B",
    "module_type": "adapter",
    "module_position": 8,  # 第8层
    "signal_type": "vector",
    "task": "dialogue_completion"
}

# 运行实验
runner = ExperimentRunner(config)
results = runner.train(epochs=10)
runner.visualize_evolution(results)
```

更多示例请查看 [examples](examples/) 目录。

## 项目结构

```
NeuroCommModule-CoEvol/
├── src/                      # 源代码
│   ├── models/               # 模型定义
│   │   ├── model_wrapper.py  # 模型抽象基类
│   │   ├── tinyllama_wrapper.py # TinyLlama模型适配器
│   │   └── __init__.py
│   ├── modules/              # 通信模块实现
│   │   ├── adapter_module.py # Adapter式通信模块
│   │   ├── vector_module.py  # 向量通信模块
│   │   ├── symbol_module.py  # 符号通信模块
│   │   └── __init__.py
│   ├── protocols/            # 通信协议
│   ├── experiments/          # 实验设计
│   │   ├── experiment_runner.py # 实验运行框架
│   │   ├── dialogue_task.py  # 对话完成任务
│   │   └── __init__.py
│   └── visualization/        # 结果可视化
│       ├── vector_visualizer.py  # 向量可视化工具
│       ├── symbol_visualizer.py  # 符号可视化工具
│       ├── parameter_visualizer.py # 参数变化可视化工具
│       └── __init__.py
├── configs/                  # 配置文件
├── examples/                 # 示例代码
│   └── basic_experiment.py   # 基础实验示例
├── tests/                    # 测试代码
├── docs/                     # 文档
├── requirements.txt          # 项目依赖
├── setup.py                  # 安装脚本
└── README.md                 # 项目说明
```

## 如何贡献

我们欢迎各种形式的贡献！如果您对神经网络通信与协同进化感兴趣，可以：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m '添加了新的通信模块'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

这个项目提供了一个灵活的框架，用于研究大型语言模型中的模块化通信协同进化现象。 
主要组件包括： 
1. 模型包装器：为不同模型提供统一接口，支持插入通信模块。
2. 通信模块：三种不同类型的通信机制（Adapter、向量、符号）。 
3. 实验框架：用于设置、运行和记录实验的完整流程。 
4. 可视化工具：用于分析和展示通信模块的演化过程。
5. 示例任务：提供对话完成任务的实现，展示如何应用该框架。

这个框架的设计理念是灵活和可扩展的，
研究者可以： 
• 添加新的通信模块类型  
• 实现不同的协作任务  
• 支持其他模型架构  
• 设计新的观测指标和可视化方法   
用户可以通过examples/basic_experiment.py快速开始使用该框架，并根据自己的研究需求进行定制。

## 参考文献

- [Modular Meta-Learning (NeurIPS 2018)](https://arxiv.org/abs/1806.10166)
- [The Sparsity of Interaction in Neural Networks (ICLR 2023)](https://arxiv.org/abs/2210.14202)
- [Emergent Communication through Meta-Learning (ICLR 2022)](https://arxiv.org/abs/2110.05208)
- [Symbol Emergence in Neural Networks (Frontiers in Robotics and AI 2023)](https://www.frontiersin.org/articles/10.3389/frobt.2023.1205524)
- [Language Processing in Brains and Machines (Nature Neuroscience 2022)](https://www.nature.com/articles/s41593-022-01114-5)

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 实现详情

### 通信模块类型

1. **Adapter模块** (`adapter_module.py`)
   - 轻量级瓶颈结构，包含降维层、激活函数、升维层和残差连接
   - 低资源消耗，适合大规模模型微调

2. **向量通信模块** (`vector_module.py`)
   - 将隐藏状态压缩为低维通信向量
   - 支持可选的噪声添加和向量量化，模拟真实通信信道

3. **符号通信模块** (`symbol_module.py`)
   - 生成离散符号序列作为通信媒介
   - 使用Gumbel-Softmax技巧实现可微分离散通信
   - 跟踪符号使用频率以分析语言涌现

### 实验设计

实验框架支持使用不同通信模块在各种任务上进行测试：

1. **对话完成任务** (`dialogue_task.py`)
   - 模型A和模型B协作完成对话生成
   - 测试通信模块在语言任务中的协同进化

2. **可视化工具** (`visualization/`)
   - 支持跟踪参数变化、向量分布和符号使用统计
   - 生成直观可解释的进化过程可视化结果

### 使用示例

目录 `examples/` 中提供了示例代码，演示如何配置和运行实验：

```python
# examples/basic_experiment.py 示例
config = {
    "base_model": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "module_type": "vector",  # 'adapter', 'vector', 或 'symbol'
    "module_position": 8,     # 在第8层插入通信模块
    "task": "dialogue_completion",
    # ...其他参数
}

runner = ExperimentRunner(config)
results = runner.train(epochs=10)
```

