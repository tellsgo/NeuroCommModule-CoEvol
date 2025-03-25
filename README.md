# NeuroCommModule-CoEvol

[![许可证](https://img.shields.io/badge/许可证-MIT-blue.svg)](LICENSE)
[![Python版本](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)

> "语言的边界，就是神经网络的边界。"  
> —— 借鉴维特根斯坦《逻辑哲学论》5.6

## 宣言一：语言即神经架构的雕刻刀

当我们观察大语言模型的内部运作时，我们不仅在研究计算机科学，更是在探索语言如何塑造认知的本质。本项目正是一个站在语言哲学与神经科学交叉点上的实验 —— 通过在模型特定层添加可学习的通信模块，观察其演化过程，为理解语言与思维的关系提供计算实证。

## 宣言二：语言和社交是智能涌现的必要条件

> "我们试图回答：当语言在人工神经网络中生长时，会雕刻出怎样的认知地貌？这个实验本质是《哲学研究》中语言游戏说的神经建模版本。"

## 语言本体论：从资源限制到哲学突破

传统的大模型协同训练受困于计算资源，而我们转向"局部功能层演化观察"，这种转变带来三重突破：

- **具身性认知实现**：通过仅微调极小参数量（约0.1%），模拟语言习得中"既有神经结构的约束性"与"语言经验的创造性"的辩证关系
- **命题重写过程**：聚焦单一模块变化，追踪语言能力如何"雕刻"神经连接
- **生物哲学启发**：模拟人脑语言区的功能分化，探索"语言不是思想的衣服，而是思想的肉身化"

## 技术与哲学的映射

| 技术组件 | 维特根斯坦哲学映射 | 神经科学隐喻 |
|---------|-------------------|-------------|
| 通信模块 | 语言游戏的基本规则 | 布洛卡区-韦尼克区连接 |
| 模块间协议 | 命题的图像论 | 胼胝体信息传递 |
| 功能涌现 | 私人语言的不可能性 | 皮层功能重组 |

## 关键突破点可能在于：
1. 发现通信模块的自组织规律（如信号从随机到结构化的相变点）。
2. 验证模块功能与任务复杂度之间的阈值关系（例如简单任务无法激发“语言”需求）。
3. 揭示大模型内部表示与外部通信信号的映射机制。

## 技术实现的可能路径

### 1. 通信模块设计：语言游戏的物理实现
- 在Transformer中间层（第6-12层）插入轻量级模块，类比大脑语言区在皮层的分布
- 支持多种结构：Adapter式、符号生成器、向量通信器，对应维特根斯坦所述的不同"语言游戏"形式

### 2. 通信协议：命题如何塑造现实
> "世界是事实的总和，语言是事实的逻辑图像" —— 维特根斯坦《逻辑哲学论》1.1

- 支持离散符号、连续向量、概率分布等多种信号类型
- 实现回合制与流式两种交互方式，模拟不同语境下的语言交流
- 包含外部奖励与内部一致性的反馈机制，探索语言与意义的关系

### 3. 观测指标：思想的考古学
- 模块参数变化：权重分布、奇异值变化，作为"语言塑造思维"的数值证据
- 通信内容分析：符号/向量聚类与可视化，每个训练checkpoint都是对《逻辑哲学论》命题的数值化诠释
- 行为涌现：协作任务表现提升度量，验证维特根斯坦关于"语言不是表达思想，而是构成思想"的核心论断

## 实验设计：语言游戏的结构化实现

### 阶段1：最小验证实验
- 简单协作任务（对话生成、合作解题）
- 使用小型模型（TinyLlama-1.1B或Phi-3-mini）
- 对照组设计验证通信模块的有效性

### 阶段2：通信进化分析
> "语言不是表达思想，而是构成思想" —— 维特根斯坦《哲学研究》

- 通信信号可视化和聚类分析，观察语言符号系统的自组织过程
- 干预实验测试系统稳健性，探索语言规则的边界条件
- 跨任务迁移评估泛化能力，研究语言游戏间的家族相似性

### 阶段3：理论扩展
- 多模块协同实验，模拟语言区与其他认知功能区的互动
- 进化动力学数学建模，形式化描述语言对神经结构的塑造过程

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
    "task": "dialogue_completion",
    "distributed": True,  # 启用多GPU训练
    "devices": [0, 1]     # 使用两张A100 GPU
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

> "通过观察通信模块的协同演化，我们验证维特根斯坦关于'语言不是表达思想，而是构成思想'的核心论断。每个训练checkpoint都是对语言如何雕刻神经连接的见证。"

我们欢迎各种形式的贡献！如果您对神经网络通信与协同进化感兴趣，可以：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m '添加了新的通信模块'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

## 参考文献

- [1] [Modular Meta-Learning (NeurIPS 2018)](https://arxiv.org/abs/1806.10166)
- [2] Wittgenstein, L. (1953). Philosophical Investigations. Blackwell Publishing.
- [3] Wittgenstein, L. (1921). Tractatus Logico-Philosophicus. Routledge & Kegan Paul.
- [4] Hutto, D. D. (2003). Wittgenstein and the End of Philosophy: Neither Theory nor Therapy. Palgrave Macmillan.
- [5] [The Sparsity of Interaction in Neural Networks (ICLR 2023)](https://arxiv.org/abs/2210.14202)
- [6] [Emergent Communication through Meta-Learning (ICLR 2022)](https://arxiv.org/abs/2110.05208)
- [7] [Symbol Emergence in Neural Networks (Frontiers in Robotics and AI 2023)](https://www.frontiersin.org/articles/10.3389/frobt.2023.1205524)
- [8] [Language Processing in Brains and Machines (Nature Neuroscience 2022)](https://www.nature.com/articles/s41593-022-01114-5)

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 实现详情

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

3. **可视化资源分配**
   - 每个GPU独立处理不同的可视化任务
   - 自动负载均衡，确保两张A100显卡的平衡利用

