import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple

class VectorCommModule(nn.Module):
    """
    向量通信模块，将隐藏状态转换为低维通信向量
    
    特点:
    - 提取语义子空间作为通信信号
    - 支持可选的量化和噪声添加（模拟通信信道）
    - 可监控通信向量的语义分布变化
    """
    
    def __init__(self, 
                 hidden_size: int, 
                 comm_size: int,
                 use_bottleneck: bool = True,
                 bottleneck_size: int = 64,
                 add_noise: bool = False,
                 noise_scale: float = 0.1,
                 quantize: bool = False,
                 num_bins: int = 16):
        """
        初始化向量通信模块
        
        参数:
            hidden_size: 模型隐藏状态大小
            comm_size: 通信向量大小
            use_bottleneck: 是否使用瓶颈层
            bottleneck_size: 瓶颈层大小
            add_noise: 是否添加噪声（模拟信道干扰）
            noise_scale: 噪声尺度
            quantize: 是否量化通信向量
            num_bins: 量化分箱数量
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.comm_size = comm_size
        self.use_bottleneck = use_bottleneck
        self.bottleneck_size = bottleneck_size
        self.add_noise = add_noise
        self.noise_scale = noise_scale
        self.quantize = quantize
        self.num_bins = num_bins
        
        # 构建网络层
        if use_bottleneck:
            # 使用瓶颈层: hidden_size -> bottleneck_size -> comm_size
            self.encoder = nn.Sequential(
                nn.Linear(hidden_size, bottleneck_size),
                nn.LayerNorm(bottleneck_size),
                nn.GELU(),
                nn.Linear(bottleneck_size, comm_size)
            )
            self.decoder = nn.Sequential(
                nn.Linear(comm_size, bottleneck_size),
                nn.LayerNorm(bottleneck_size),
                nn.GELU(),
                nn.Linear(bottleneck_size, hidden_size)
            )
        else:
            # 直接映射: hidden_size -> comm_size
            self.encoder = nn.Linear(hidden_size, comm_size)
            self.decoder = nn.Linear(comm_size, hidden_size)
        
        # 通信统计跟踪
        self.register_buffer("comm_vectors", torch.zeros(1000, comm_size))  # 存储最近的通信向量
        self.register_buffer("comm_idx", torch.tensor(0))  # 当前索引
        self.register_buffer("comm_count", torch.tensor(0))  # 通信次数计数
        
        # 量化参数（如果启用）
        if quantize:
            self.register_buffer("bin_edges", torch.linspace(-3.0, 3.0, num_bins+1))
    
    def encode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """将隐藏状态编码为通信向量"""
        # 获取序列中的CLS位置输出或使用平均池化
        if hidden_states.dim() == 3:  # [batch_size, seq_len, hidden_size]
            # 使用平均池化
            pooled = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        else:
            pooled = hidden_states  # 已经是池化后的向量
        
        # 编码为通信向量
        comm_vector = self.encoder(pooled)  # [batch_size, comm_size]
        
        # 通信信道模拟（可选）
        if self.add_noise and self.training:
            noise = torch.randn_like(comm_vector) * self.noise_scale
            comm_vector = comm_vector + noise
        
        # 向量量化（可选）
        if self.quantize:
            comm_vector = self._quantize(comm_vector)
        
        # 更新统计信息
        if self.training:
            batch_size = comm_vector.size(0)
            for i in range(batch_size):
                idx = self.comm_idx % 1000
                self.comm_vectors[idx] = comm_vector[i].detach()
                self.comm_idx += 1
            self.comm_count += batch_size
        
        return comm_vector
    
    def decode(self, comm_vector: torch.Tensor) -> torch.Tensor:
        """将通信向量解码回隐藏状态"""
        return self.decoder(comm_vector)
    
    def _quantize(self, x: torch.Tensor) -> torch.Tensor:
        """对通信向量进行量化"""
        if not self.quantize:
            return x
            
        # 对每个值进行分箱
        x_q = torch.zeros_like(x)
        
        for i in range(self.num_bins):
            # 计算每个分箱的掩码
            mask = (x > self.bin_edges[i]) & (x <= self.bin_edges[i+1])
            # 将分箱中心值分配给掩码区域
            bin_center = (self.bin_edges[i] + self.bin_edges[i+1]) / 2
            x_q[mask] = bin_center
            
        # 处理超出范围的值
        x_q[x <= self.bin_edges[0]] = self.bin_edges[0]
        x_q[x > self.bin_edges[-1]] = self.bin_edges[-1]
        
        return x_q
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            hidden_states: 输入隐藏状态 [batch_size, seq_len, hidden_size]
            
        返回:
            修改后的隐藏状态 [batch_size, seq_len, hidden_size]
        """
        # 保存原始输入用于残差连接
        residual = hidden_states
        
        # 只处理序列的第一个标记作为通信载体
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # 编码为通信向量
        comm_vector = self.encode(hidden_states)  # [batch_size, comm_size]
        
        # 解码回隐藏状态
        expanded_comm = self.decode(comm_vector)  # [batch_size, hidden_size]
        
        # 扩展到与序列长度相同的维度
        expanded_comm = expanded_comm.unsqueeze(1).expand(-1, seq_len, -1)
        
        # 添加残差连接
        output = residual + 0.1 * expanded_comm  # 缩小影响，防止过强干扰
        
        return output
    
    def get_recent_vectors(self, n: int = 100) -> torch.Tensor:
        """获取最近的n个通信向量"""
        n = min(n, int(self.comm_count.item()), 1000)
        if n == 0:
            return torch.zeros(0, self.comm_size)
            
        idx = self.comm_idx % 1000
        if idx >= n:
            return self.comm_vectors[idx-n:idx]
        else:
            # 环形缓冲区情况
            return torch.cat([
                self.comm_vectors[-(n-idx):],
                self.comm_vectors[:idx]
            ], dim=0) 