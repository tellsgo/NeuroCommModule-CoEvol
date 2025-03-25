import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple


def plot_parameter_changes(param_changes_history: List[Dict[str, Any]],
                          save_path: str,
                          title: Optional[str] = None,
                          figsize: Tuple[int, int] = (12, 8)):
    """
    绘制参数变化曲线
    
    参数:
        param_changes_history: 包含参数变化历史的列表
        save_path: 图像保存路径
        title: 图像标题
        figsize: 图像尺寸
    """
    if not param_changes_history:
        # 创建空白图像
        plt.figure(figsize=figsize)
        plt.title("无参数变化数据" if title is None else title)
        plt.text(0.5, 0.5, "没有参数变化历史可供可视化", 
                 horizontalalignment='center', verticalalignment='center')
        plt.savefig(save_path)
        plt.close()
        return
    
    # 提取模型A和模型B的总体变化
    epochs = range(1, len(param_changes_history) + 1)
    model_a_changes = [record["model_a"]["total"] for record in param_changes_history]
    model_b_changes = [record["model_b"]["total"] for record in param_changes_history]
    
    # 创建图像
    plt.figure(figsize=figsize)
    
    # 绘制总体变化曲线
    plt.subplot(2, 1, 1)
    plt.plot(epochs, model_a_changes, 'b-', marker='o', label='模型A')
    plt.plot(epochs, model_b_changes, 'r-', marker='x', label='模型B')
    
    if title:
        plt.title(title)
    else:
        plt.title("通信模块参数变化")
    
    plt.xlabel("训练轮次")
    plt.ylabel("参数变化量")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 提取各层变化（仅关注第一个模型）
    if len(param_changes_history) > 0 and "model_a" in param_changes_history[0]:
        layer_names = [k for k in param_changes_history[0]["model_a"].keys() if k != "total"]
        
        if layer_names:  # 如果有层级变化数据
            plt.subplot(2, 1, 2)
            
            for layer in layer_names:
                layer_changes = [record["model_a"][layer] for record in param_changes_history]
                plt.plot(epochs, layer_changes, marker='.', label=f'层 {layer}')
            
            plt.title("各层参数变化趋势 (模型A)")
            plt.xlabel("训练轮次")
            plt.ylabel("参数变化量")
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_parameter_distribution(initial_params: Dict[str, np.ndarray],
                               current_params: Dict[str, np.ndarray],
                               save_path: str,
                               n_samples: int = 1000,
                               title: Optional[str] = None):
    """
    绘制参数分布变化对比
    
    参数:
        initial_params: 初始参数字典
        current_params: 当前参数字典
        save_path: 图像保存路径
        n_samples: 随机采样的参数数量
        title: 图像标题
    """
    if not initial_params or not current_params:
        # 创建空白图像
        plt.figure(figsize=(10, 6))
        plt.title("无参数分布数据" if title is None else title)
        plt.savefig(save_path)
        plt.close()
        return
    
    # 计算每层的参数变化
    param_diffs = {}
    for name in initial_params:
        if name in current_params:
            param_diffs[name] = (
                initial_params[name].flatten(), 
                current_params[name].flatten(),
                (current_params[name] - initial_params[name]).flatten()
            )
    
    # 确定图像布局
    n_layers = len(param_diffs)
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols
    
    # 创建图像
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # 对于每层绘制分布
    for i, (name, (initial, current, diff)) in enumerate(param_diffs.items()):
        if i < len(axes):
            ax = axes[i]
            
            # 随机采样参数
            if len(initial) > n_samples:
                idx = np.random.choice(len(initial), n_samples, replace=False)
                initial_sample = initial[idx]
                current_sample = current[idx]
                diff_sample = diff[idx]
            else:
                initial_sample = initial
                current_sample = current
                diff_sample = diff
            
            # 绘制分布
            ax.hist(initial_sample, bins=30, alpha=0.5, label='初始')
            ax.hist(current_sample, bins=30, alpha=0.5, label='当前')
            
            # 设置标题和标签
            ax.set_title(f"层 {name}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 添加统计信息
            init_mean = initial.mean()
            init_std = initial.std()
            current_mean = current.mean()
            current_std = current.std()
            
            stat_text = f"初始: μ={init_mean:.4f}, σ={init_std:.4f}\n"
            stat_text += f"当前: μ={current_mean:.4f}, σ={current_std:.4f}"
            ax.text(0.05, 0.95, stat_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    # 隐藏多余的子图
    for i in range(len(param_diffs), len(axes)):
        axes[i].axis('off')
    
    # 设置总标题
    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle("参数分布变化", fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # 为总标题留出空间
    plt.savefig(save_path, dpi=300)
    plt.close() 