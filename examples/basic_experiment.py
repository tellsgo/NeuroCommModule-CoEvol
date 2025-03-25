#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基础实验示例脚本

该脚本演示如何使用NeuroCommModule-CoEvol进行基本的通信模块协同进化实验。
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
    """运行基础实验示例"""
    # 确保实验目录存在
    os.makedirs("experiments", exist_ok=True)
    
    # 设置实验配置
    config = {
        # 模型配置
        "base_model": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        
        # 通信模块配置
        "module_type": "vector",  # 'adapter', 'vector', 或 'symbol'
        "module_position": 8,     # 在第8层插入通信模块
        
        # 向量通信模块特定参数
        "comm_size": 32,          # 通信向量维度
        "use_bottleneck": True,   # 使用瓶颈层
        "add_noise": True,        # 添加噪声模拟通信信道
        
        # 任务配置
        "task": "dialogue_completion",  # 对话补全任务
        "data_path": "data/dialogue_samples.json",  # 任务数据路径
        "batch_size": 8,
        
        # 训练配置
        "learning_rate": 5e-5,
        "num_epochs": 10,
        "use_lr_scheduler": True,
        
        # 可视化配置
        "vis_interval": 1,   # 每轮生成可视化
        "save_interval": 5   # 每5轮保存检查点
    }
    
    # 检查GPU可用性
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 初始化并运行实验
    try:
        runner = ExperimentRunner(config)
        results = runner.train(epochs=config["num_epochs"])
        
        # 生成进化过程可视化
        runner.visualize_evolution(results)
        
        print(f"实验完成! 结果保存在: {runner.exp_dir}")
        
    except Exception as e:
        print(f"实验运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 