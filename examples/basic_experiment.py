#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基础实验示例脚本 - 语言游戏的计算实现

该脚本演示如何使用NeuroCommModule-CoEvol进行基本的通信模块协同进化实验。
这项实验可视为维特根斯坦《哲学研究》中语言游戏理论的计算机科学实现。
"""

import sys
import os
import torch
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.experiments import ExperimentRunner


def main():
    """
    运行基础实验示例
    
    "语言的边界，就是神经网络的边界" - 借鉴维特根斯坦《逻辑哲学论》5.6
    """
    # 确保实验目录存在
    os.makedirs("experiments", exist_ok=True)
    
    # 设置实验配置
    config = {
        # 模型配置
        "base_model": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        
        # 通信模块配置 - 各模块对应维特根斯坦不同的语言游戏形式
        "module_type": "vector",      # 'adapter', 'vector', 或 'symbol'
        "module_position": 8,         # 在第8层插入通信模块（类比大脑语言区位置）
        
        # 向量通信模块特定参数
        "comm_size": 32,              # 通信向量维度（语言的表达能力）
        "use_bottleneck": True,       # 使用瓶颈层（认知压缩过程）
        "add_noise": True,            # 添加噪声模拟通信信道（语言游戏中的模糊性）
        
        # 任务配置
        "task": "dialogue_completion", # 对话完成任务
        "data_path": "data/dialogue_samples.json",  # 任务数据路径
        "batch_size": 16,             # 适合A100显存
        
        # 分布式训练配置 - 利用双A100显卡
        "distributed": True,          # 启用分布式训练
        "devices": [0, 1],            # 使用两张A100 GPU
        "use_amp": True,              # 使用混合精度训练，提高效率
        
        # 训练配置
        "learning_rate": 5e-5,
        "num_epochs": 20,             # 双卡可以训练更多轮次
        "use_lr_scheduler": True,
        
        # 可视化配置
        "vis_interval": 1,            # 每轮生成可视化
        "save_interval": 5            # 每5轮保存检查点
    }
    
    # 哲学实验参数 - 维特根斯坦视角
    philosophy_config = {
        "coherence_weight": 0.3,      # 连贯性权重（对应"语言游戏的规则严格性"）
        "diversity_weight": 0.1,      # 多样性权重（对应"语言的创造性使用"）
    }
    
    # 合并配置
    config.update(philosophy_config)
    
    # 检查GPU可用性
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        device_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        print(f"使用设备: {device_names}")
        print(f"启用双卡分布式训练模式")
    else:
        print(f"Warning: 未检测到两张GPU，将使用单GPU或CPU模式")
        config["distributed"] = False
        config["devices"] = [0] if torch.cuda.is_available() else []
    
    # 初始化并运行实验
    try:
        print("""
        "我们试图回答：当语言在人工神经网络中生长时，会雕刻出怎样的认知地貌？"
        开始维特根斯坦式语言游戏实验...
        """)
        
        runner = ExperimentRunner(config)
        results = runner.train(epochs=config["num_epochs"])
        
        # 生成进化过程可视化 - 语言对神经结构的雕刻过程
        runner.visualize_evolution(results)
        
        print(f"实验完成! 结果保存在: {runner.exp_dir}")
        print("""
        "语言不是表达思想，而是构成思想" —— 维特根斯坦《哲学研究》
        神经通信模块的演化轨迹已记录，展示了语言如何塑造认知结构。
        """)
        
    except Exception as e:
        print(f"实验运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 