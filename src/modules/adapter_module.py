import torch
import torch.nn as nn
from typing import Optional, Dict, Any

class AdapterModule(nn.Module):
    """
    Adapter风格的通信模块，插入到Transformer层后用于通信
    
    结构:
    - 输入向量
    - 降维全连接层
    - 激活函数
    - 升维全连接层
    - 残差连接
    """
    
    def __init__(self, 
                 hidden_size: int, 
                 adapter_size: int,
                 adapter_dropout: float = 0.1,
                 adapter_act: str = "gelu",
                 init_scale: float = 1e-3):
        """
        初始化Adapter模块
        
        参数:
            hidden_size: 模型隐藏层大小
            adapter_size: Adapter内部表示大小
            adapter_dropout: Dropout率
            adapter_act: 激活函数类型 ('relu', 'gelu', 'swish')
            init_scale: 初始化权重的缩放因子
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.adapter_size = adapter_size
        
        # 选择激活函数
        if adapter_act == "relu":
            self.activation = nn.ReLU()
        elif adapter_act == "gelu":
            self.activation = nn.GELU()
        elif adapter_act == "swish":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"不支持的激活函数: {adapter_act}")
        
        # 降维层
        self.down_proj = nn.Linear(hidden_size, adapter_size)
        
        # Dropout
        self.dropout = nn.Dropout(adapter_dropout)
        
        # 升维层
        self.up_proj = nn.Linear(adapter_size, hidden_size)
        
        # 使用较小的初始化权重以减少干扰原始模型
        self._init_adapter_weights(init_scale)
        
        # 通信统计跟踪
        self.register_buffer("activation_mean", torch.zeros(adapter_size))
        self.register_buffer("activation_std", torch.ones(adapter_size))
        self.register_buffer("comm_count", torch.tensor(0))
        
    def _init_adapter_weights(self, init_scale: float):
        """初始化Adapter权重"""
        with torch.no_grad():
            nn.init.normal_(self.down_proj.weight, std=init_scale)
            nn.init.normal_(self.up_proj.weight, std=init_scale)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入隐藏状态 [batch_size, seq_len, hidden_size]
            
        返回:
            修改后的隐藏状态 [batch_size, seq_len, hidden_size]
        """
        # 保存原始输入用于残差连接
        residual = x
        
        # 降维层
        h = self.down_proj(x)  # [batch_size, seq_len, adapter_size]
        
        # 激活函数
        h = self.activation(h)
        
        # 更新通信统计信息
        if self.training:
            # 计算当前批次的激活统计数据
            batch_mean = h.detach().mean(dim=(0, 1))  # 平均值
            batch_std = h.detach().std(dim=(0, 1))    # 标准差
            
            # 更新移动平均
            momentum = 0.1
            self.activation_mean = (1 - momentum) * self.activation_mean + momentum * batch_mean
            self.activation_std = (1 - momentum) * self.activation_std + momentum * batch_std
            self.comm_count += 1
        
        # Dropout
        h = self.dropout(h)
        
        # 升维层
        h = self.up_proj(h)  # [batch_size, seq_len, hidden_size]
        
        # 残差连接
        output = residual + h
        
        return output
    
    def get_activation_stats(self) -> Dict[str, torch.Tensor]:
        """获取激活统计信息"""
        return {
            "mean": self.activation_mean.clone(),
            "std": self.activation_std.clone(),
            "count": self.comm_count.clone()
        } 