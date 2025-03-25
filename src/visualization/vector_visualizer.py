import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from typing import Optional, Union, List, Tuple


def visualize_comm_vectors(
    vectors: Union[np.ndarray, torch.Tensor], 
    save_path: str, 
    method: str = "tsne", 
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    可视化通信向量，使用降维技术显示高维向量分布
    
    参数:
        vectors: 形状为(n_samples, dim)的通信向量数组
        save_path: 图像保存路径
        method: 降维方法，'pca'或'tsne'
        title: 图像标题
        figsize: 图像尺寸
    """
    # 转换为numpy数组
    if isinstance(vectors, torch.Tensor):
        vectors = vectors.detach().cpu().numpy()
        
    if vectors.shape[0] == 0:
        # 创建空白图像
        plt.figure(figsize=figsize)
        plt.title("无可视化数据" if title is None else title)
        plt.text(0.5, 0.5, "没有通信向量可供可视化", 
                 horizontalalignment='center', verticalalignment='center')
        plt.savefig(save_path)
        plt.close()
        return
        
    # 执行降维
    if method.lower() == "pca":
        n_components = min(2, vectors.shape[1])
        reducer = PCA(n_components=n_components)
    elif method.lower() == "tsne":
        reducer = TSNE(n_components=2, perplexity=min(30, max(5, vectors.shape[0] // 5)),
                      random_state=42)
    else:
        raise ValueError(f"不支持的降维方法: {method}，请使用'pca'或'tsne'")
        
    # 降维为2D
    reduced_vectors = reducer.fit_transform(vectors)
    
    # 绘制散点图
    plt.figure(figsize=figsize)
    scatter = plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], 
               c=np.arange(reduced_vectors.shape[0]), cmap='viridis', 
               alpha=0.8, s=50)
    
    # 添加颜色条表示时间顺序
    cbar = plt.colorbar(scatter)
    cbar.set_label('时间顺序')
    
    # 添加标题和标签
    if title:
        plt.title(title)
    else:
        plt.title(f"通信向量分布 ({method.upper()})")
        
    plt.xlabel("维度 1")
    plt.ylabel("维度 2")
    plt.grid(alpha=0.3)
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_vector_change_over_time(
    vector_history: List[np.ndarray],
    save_path: str,
    component_indices: Optional[List[int]] = None,
    title: Optional[str] = None
):
    """
    可视化通信向量随时间的变化
    
    参数:
        vector_history: 各个时间点的向量列表
        save_path: 图像保存路径
        component_indices: 要显示的向量分量索引
        title: 图像标题
    """
    if not vector_history or len(vector_history) == 0:
        # 创建空白图像
        plt.figure(figsize=(10, 6))
        plt.title("无时间序列数据" if title is None else title)
        plt.text(0.5, 0.5, "没有通信向量历史可供可视化", 
                 horizontalalignment='center', verticalalignment='center')
        plt.savefig(save_path)
        plt.close()
        return
    
    # 确定要显示的分量
    if component_indices is None:
        # 选择前5个分量
        component_indices = list(range(min(5, vector_history[0].shape[0])))
    
    # 提取选定分量的时间序列
    time_steps = list(range(len(vector_history)))
    component_values = []
    
    for idx in component_indices:
        values = [vec[idx] if idx < vec.shape[0] else 0 for vec in vector_history]
        component_values.append(values)
    
    # 绘制时间序列
    plt.figure(figsize=(12, 6))
    
    for i, values in enumerate(component_values):
        plt.plot(time_steps, values, marker='o', label=f"分量 {component_indices[i]}")
    
    # 添加标签和标题
    plt.xlabel("时间步")
    plt.ylabel("向量值")
    if title:
        plt.title(title)
    else:
        plt.title("通信向量随时间的变化")
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() 