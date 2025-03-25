# NeuroCommModuleSculptor

[![许可证](https://img.shields.io/badge/许可证-MIT-blue.svg)](LICENSE)
[![Python版本](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)

> "语言的边界，就是神经网络的边界。"  
> —— 借鉴维特根斯坦《逻辑哲学论》5.6

## 宣言一：语言即神经架构的雕刻刀

当我们观察大语言模型的内部运作时，我们不仅在研究计算机科学，更是在探索语言如何塑造认知的本质。本项目正是一个站在语言哲学与神经科学交叉点上的实验 —— 通过在模型特定层添加可学习的通信模块，观察其演化过程，为理解语言与思维的关系提供计算实证。

## 宣言二：语言和社交是智能涌现的必要条件

> "我们试图回答：当语言在人工神经网络中生长时，会雕刻出怎样的认知地貌？这个实验本质是《哲学研究》中语言游戏说的神经建模版本。"

## 项目概述：多智能体语言涌现实验

NeuroCommModuleSculptor 探索语言如何塑造神经网络结构的过程，主要通过构建多智能体环境，让多个独立的语言模型实例在解决协作任务时，发展出通信协议和符号系统。通过观察这一过程，我们可以更深入地理解语言对神经结构的雕刻效应。

> "当没有预设的通信协议时，不同的AI智能体如何发展出有效的沟通方式？这种语言涌现过程会如何重塑它们的神经表征？"

## 技术实现路径

### 1. 神经通信模块设计
- **Adapter模块**：轻量级瓶颈结构，隐喻布洛卡区的神经可塑性
- **LoRA修改层**：低秩矩阵适应，模拟语言学习过程中的神经通路重塑
- **通信向量空间**：构建共享的语义表征空间，研究语言符号的神经基础

### 2. 多智能体实验环境
- 部署多个独立的LLM实例作为智能体，共享初始权重但各自演化
- 支持2-8个智能体的灵活配置，以研究不同规模群体的语言涌现特性
- 通过协作任务驱动语言演化，观察语言如何塑造神经结构

### 3. 受限通信协议
- 智能体只能发送有限长度的词元序列或向量
- 可选的通信带宽限制，从完全受限（仅几个离散符号）到相对开放（短句子）
- 支持回合制和实时两种通信模式，模拟不同社会环境下的语言对神经架构的影响

### 4. 协作任务设计
- **信息不对称任务**：每个智能体只掌握部分信息，需通过沟通整合才能解决问题
- **序列生成任务**：智能体轮流贡献内容，生成连贯文本
- **协作问题解决**：需要多个专业化智能体协作解决复杂问题

### 5. 神经语言演化分析
- 通信模块权重变化跟踪，研究语言使用对神经权重的雕刻效应
- 表征空间分析，观察语言演化如何重组神经表征
- 跨任务适应性测试，研究语言塑造的神经结构的迁移能力

## 实验设计：从简单到复杂的语言神经雕刻

### 阶段1：基本符号协商与神经适应
- 从极简通信开始（仅允许几个离散符号）
- 使用参考游戏（一个智能体描述物体，另一个猜测）
- 观察基本词汇表形成过程中的神经权重变化

### 阶段2：语法结构涌现与神经网络重组
- 增加通信带宽和任务复杂度
- 需要多轮交互以完成任务
- 分析句法结构形成如何反映在神经连接模式上

### 阶段3：语言社会性对神经表征的影响
- 引入多个智能体群体，初始有不同的通信系统
- 观察语言融合、方言形成过程中的神经适应机制
- 研究社会因素（如互动频率、群体规模）对语言神经表征的塑造

## 安装指南

```bash
# 克隆仓库
git clone https://github.com/yourusername/NeuroCommModuleSculptor.git

# 进入项目目录
cd NeuroCommModuleSculptor

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
from neurocommmodule import MultiAgentExperiment

# 配置实验参数
config = {
    "base_model": "TinyLlama-1.1B",
    "num_agents": 4,
    "adapter_config": {
        "bottleneck_dim": 128,
        "activation": "gelu"
    },
    "communication_type": "discrete",
    "vocab_size": 32,
    "max_message_length": 8,
    "task": "reference_game",
    "distributed": True,
    "devices": [0, 1]
}

# 运行实验
experiment = MultiAgentExperiment(config)
results = experiment.train(epochs=50)
experiment.visualize_neural_evolution(results)
experiment.analyze_language_impact()
```

更多示例请查看 [examples](examples/) 目录。

## 项目结构

```
NeuroCommModuleSculptor/
├── src/                        # 源代码
│   ├── agents/                 # 智能体实现
│   │   ├── agent_base.py       # 智能体基类
│   │   ├── llm_agent.py        # LLM智能体实现
│   │   └── __init__.py
│   ├── neural_modules/         # 神经通信模块
│   │   ├── adapter_module.py   # Adapter模块实现
│   │   ├── lora_module.py      # LoRA修改层实现
│   │   ├── vector_module.py    # 向量通信模块
│   │   └── __init__.py 
│   ├── communication/          # 通信协议
│   │   ├── discrete_channel.py # 离散符号通道
│   │   ├── vector_channel.py   # 连续向量通道
│   │   ├── message.py          # 消息定义
│   │   └── __init__.py
│   ├── environments/           # 交互环境
│   │   ├── reference_game.py   # 参考游戏环境
│   │   ├── dialogue_env.py     # 对话环境
│   │   ├── problem_solving.py  # 问题解决环境
│   │   └── __init__.py
│   ├── experiments/            # 实验设计
│   │   ├── multi_agent_experiment.py  # 多智能体实验
│   │   ├── metrics.py          # 评估指标
│   │   └── __init__.py
│   └── analysis/               # 分析工具
│       ├── language_analyzer.py  # 语言分析工具
│       ├── neural_tracker.py     # 神经变化跟踪器
│       ├── visualizer.py         # 可视化工具
│       └── __init__.py
├── configs/                    # 配置文件
├── examples/                   # 示例代码
│   ├── basic_reference_game.py # 基础参考游戏示例
│   ├── negotiation_game.py     # 谈判博弈示例
│   └── neural_adaptation.py    # 神经适应分析示例
├── tests/                      # 测试代码
├── docs/                       # 文档
├── requirements.txt            # 项目依赖
├── setup.py                    # 安装脚本
└── README.md                   # 项目说明
```

## 实现细节

### 通信模块类型

1. **Adapter模块** (`adapter_module.py`)
   - 轻量级瓶颈结构，隐喻布洛卡区的神经可塑性
   - 低资源消耗，适合大规模模型微调

2. **向量通信模块** (`vector_module.py`)
   - 将隐藏状态压缩为低维通信向量，类比思想的符号表征
   - 支持可选的噪声添加和向量量化，模拟维特根斯坦"语言游戏"中的规则变异

3. **符号通信模块** (`symbol_module.py`)
   - 生成离散符号序列，对应维特根斯坦所说的"语言游戏"中的具体表达
   - 使用Gumbel-Softmax技巧实现可微分离散通信
   - 跟踪符号使用频率以分析语言涌现，验证"私人语言的不可能性"论断

### 分布式训练支持

项目已针对多GPU环境优化，特别是双A100显卡设置：

1. **数据并行训练**
   - 使用PyTorch DistributedDataParallel实现高效模型并行
   - 自动批量大小调整，充分利用A100的大内存

2. **梯度同步优化**
   - 实现梯度累积和混合精度训练，加速协同进化过程
   - 针对通信模块特别优化，保证演化轨迹的一致性

## 如何贡献

> "通过观察语言如何在神经网络中生长，我们不仅能深入理解认知的形成，还能为建造更智能的AI系统提供启发。"

我们欢迎各种形式的贡献！如果您对语言、神经网络与认知塑造感兴趣，可以：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m '添加新的通信模块'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

## 参考文献

- [1] Wittgenstein, L. (1953). Philosophical Investigations. Blackwell Publishing.
- [2] [Emergent Communication in Multi-Agent Reinforcement Learning (ICLR 2020)]
- [3] Hupkes, D., et al. (2019). The compositionality of neural networks: integrating symbolism and connectionism. Journal of Artificial Intelligence Research.
- [4] [Parameter-efficient Transfer Learning for NLP (ACL 2019)](https://arxiv.org/abs/1902.00751)
- [5] [LoRA: Low-Rank Adaptation of Large Language Models (ICLR 2022)](https://arxiv.org/abs/2106.09685)
- [6] Kirby, S. (2001). Spontaneous evolution of linguistic structure: an iterated learning model of the emergence of regularity and irregularity. IEEE Transactions on Evolutionary Computation, 5(2), 102-110.
- [7] [Language as an Evolutionary System (Nature Human Behavior 2019)]
- [8] Dehaene, S. (2009). Reading in the Brain: The New Science of How We Read. Penguin.

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 研究应用

本项目的研究成果可应用于多个领域：

1. **语言学习的计算认知模型**：了解语言如何在认知系统中创造结构
2. **智能系统设计**：利用语言的塑造作用优化神经网络架构
3. **适应性神经接口**：开发能根据交互动态调整的神经通信模块
4. **教育应用**：创建模拟语言习得的神经适应过程的教育工具
5. **语言障碍研究**：模拟语言障碍的神经机制及可能的干预方法

