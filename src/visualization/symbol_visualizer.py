import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Union
import torch


def plot_symbol_usage(
    counts: Union[np.ndarray, torch.Tensor, List],
    save_path: str,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    绘制符号使用频率分布
    
    参数:
        counts: 各符号的使用计数
        save_path: 图像保存路径
        title: 图像标题
        figsize: 图像尺寸
    """
    # 转换为numpy数组
    if isinstance(counts, torch.Tensor):
        counts = counts.cpu().numpy()
    elif isinstance(counts, list):
        counts = np.array(counts)
    
    # 计算概率分布
    total = counts.sum()
    if total > 0:
        probs = counts / total
    else:
        probs = np.zeros_like(counts)
    
    # 计算熵
    non_zero_probs = probs[probs > 0]
    entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs)) if len(non_zero_probs) > 0 else 0
    
    # 创建图像
    plt.figure(figsize=figsize)
    
    # 绘制条形图
    x = np.arange(len(counts))
    plt.bar(x, probs, alpha=0.7)
    
    # 添加标题和标签
    if title:
        plt.title(title)
    else:
        plt.title(f"符号使用分布 (熵: {entropy:.2f} bits)")
    
    plt.xlabel("符号ID")
    plt.ylabel("使用频率")
    
    # 设置x轴刻度
    if len(counts) <= 50:
        plt.xticks(x)
    else:
        # 对于大量符号，只显示部分刻度
        plt.xticks(np.arange(0, len(counts), len(counts) // 10))
    
    plt.grid(axis='y', alpha=0.3)
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_symbol_sequences(
    sequences: Union[np.ndarray, torch.Tensor, List[List[int]]],
    save_path: str,
    vocab_size: Optional[int] = None,
    title: Optional[str] = None,
    max_seqs: int = 20,
    figsize: Optional[Tuple[int, int]] = None
):
    """
    可视化符号序列模式
    
    参数:
        sequences: 形状为(n_samples, seq_len)的符号序列
        save_path: 图像保存路径
        vocab_size: 词汇表大小
        title: 图像标题
        max_seqs: 最多显示的序列数
        figsize: 图像尺寸
    """
    # 转换为numpy数组
    if isinstance(sequences, torch.Tensor):
        sequences = sequences.cpu().numpy()
    elif isinstance(sequences, list) and sequences and isinstance(sequences[0], list):
        sequences = np.array(sequences)
    
    if len(sequences) == 0:
        # 创建空白图像
        plt.figure(figsize=(10, 6))
        plt.title("无序列数据" if title is None else title)
        plt.text(0.5, 0.5, "没有符号序列可供可视化", 
                 horizontalalignment='center', verticalalignment='center')
        plt.savefig(save_path)
        plt.close()
        return
    
    # 限制可视化的序列数量
    n_seqs = min(len(sequences), max_seqs)
    seq_len = sequences.shape[1]
    
    # 确定词汇表大小
    if vocab_size is None:
        vocab_size = int(np.max(sequences)) + 1
    
    # 计算图像尺寸
    if figsize is None:
        figsize = (12, max(6, n_seqs * 0.4))
    
    # 创建热力图
    plt.figure(figsize=figsize)
    
    # 使用seaborn的热力图
    ax = sns.heatmap(sequences[:n_seqs], cmap="viridis", 
               cbar_kws={'label': '符号ID'}, vmin=0, vmax=vocab_size-1)
    
    # 设置标题和标签
    if title:
        plt.title(title)
    else:
        plt.title(f"符号序列模式 (显示 {n_seqs} 个序列)")
    
    plt.xlabel("序列位置")
    plt.ylabel("样本ID")
    
    # 调整y轴刻度位置
    plt.yticks(np.arange(n_seqs) + 0.5, np.arange(n_seqs))
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_symbol_transition_matrix(
    sequences: Union[np.ndarray, torch.Tensor, List[List[int]]],
    save_path: str,
    vocab_size: Optional[int] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    绘制符号转移概率矩阵
    
    参数:
        sequences: 形状为(n_samples, seq_len)的符号序列
        save_path: 图像保存路径
        vocab_size: 词汇表大小
        title: 图像标题
        figsize: 图像尺寸
    """
    # 转换为numpy数组
    if isinstance(sequences, torch.Tensor):
        sequences = sequences.cpu().numpy()
    elif isinstance(sequences, list) and sequences and isinstance(sequences[0], list):
        sequences = np.array(sequences)
    
    if len(sequences) == 0 or sequences.shape[1] < 2:
        # 创建空白图像
        plt.figure(figsize=figsize)
        plt.title("无转移数据" if title is None else title)
        plt.text(0.5, 0.5, "没有足够的符号序列可供分析转移矩阵", 
                 horizontalalignment='center', verticalalignment='center')
        plt.savefig(save_path)
        plt.close()
        return
    
    # 确定词汇表大小
    if vocab_size is None:
        vocab_size = int(np.max(sequences)) + 1
    
    # 创建转移矩阵
    transition_matrix = np.zeros((vocab_size, vocab_size))
    
    # 计算转移次数
    for seq in sequences:
        for i in range(len(seq) - 1):
            transition_matrix[seq[i], seq[i+1]] += 1
    
    # 转换为概率
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    # 避免除以零
    row_sums[row_sums == 0] = 1
    transition_probs = transition_matrix / row_sums
    
    # 创建热力图
    plt.figure(figsize=figsize)
    sns.heatmap(transition_probs, cmap="viridis", 
               cbar_kws={'label': '转移概率'}, vmin=0, vmax=1)
    
    # 设置标题和标签
    if title:
        plt.title(title)
    else:
        plt.title("符号转移概率矩阵")
    
    plt.xlabel("目标符号")
    plt.ylabel("源符号")
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() 