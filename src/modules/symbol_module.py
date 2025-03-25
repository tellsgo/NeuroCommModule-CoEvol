import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple

class SymbolCommModule(nn.Module):
    """
    符号通信模块，将隐藏状态映射为离散符号序列
    
    特点:
    - 生成离散符号作为更抽象的通信形式
    - 使用Gumbel-Softmax技巧实现可微分离散通信
    - 跟踪符号使用情况以分析"语言"涌现
    """
    
    def __init__(self, 
                 hidden_size: int, 
                 vocab_size: int = 32,
                 seq_len: int = 8,
                 embedding_dim: int = 64,
                 temperature: float = 1.0,
                 straight_through: bool = True):
        """
        初始化符号通信模块
        
        参数:
            hidden_size: 模型隐藏层大小
            vocab_size: 符号词汇表大小
            seq_len: 符号序列长度
            embedding_dim: 符号嵌入维度
            temperature: Gumbel-Softmax温度
            straight_through: 是否使用Straight-Through技巧
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.straight_through = straight_through
        
        # 编码器：隐藏状态 -> 符号概率
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, seq_len * vocab_size)
        )
        
        # 符号嵌入层
        self.symbol_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 解码器：符号嵌入 -> 隐藏状态
        self.decoder = nn.Sequential(
            nn.Linear(seq_len * embedding_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, hidden_size)
        )
        
        # 符号使用统计
        self.register_buffer("symbol_counts", torch.zeros(vocab_size))
        self.register_buffer("comm_count", torch.tensor(0))
        
        # 符号序列历史
        self.register_buffer("symbol_history", torch.zeros(1000, seq_len, dtype=torch.long))
        self.register_buffer("history_idx", torch.tensor(0))
    
    def gumbel_softmax(self, logits: torch.Tensor, tau: float = 1.0, hard: bool = False):
        """
        Gumbel-Softmax技巧，用于生成可微分的离散样本
        
        参数:
            logits: 形状为 [..., vocab_size] 的未归一化对数概率
            tau: 温度参数
            hard: 是否使用hard版本（straight-through）
            
        返回:
            软化或离散化的样本，形状与logits相同
        """
        # 重塑为适当的形状
        batch_size = logits.size(0)
        logits = logits.view(-1, self.vocab_size)
        
        # 使用PyTorch内置的gumbel_softmax
        y_soft = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        
        # 如果需要hard版本，但PyTorch版本不支持hard参数
        if hard and not y_soft.is_cuda:
            # 手动实现straight-through估计器
            index = y_soft.max(-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
            
        return ret.view(batch_size, -1, self.vocab_size)
    
    def encode(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将隐藏状态编码为符号序列
        
        返回:
            - 软化的符号表示 [batch_size, seq_len, vocab_size]
            - 离散的符号索引 [batch_size, seq_len]
        """
        # 池化隐藏状态
        if hidden_states.dim() == 3:  # [batch_size, seq_len, hidden_size]
            pooled = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        else:
            pooled = hidden_states
            
        batch_size = pooled.size(0)
        
        # 生成符号概率
        logits = self.encoder(pooled)  # [batch_size, seq_len*vocab_size]
        logits = logits.view(batch_size, self.seq_len, self.vocab_size)
        
        # 应用Gumbel-Softmax生成可微分样本
        soft_symbols = self.gumbel_softmax(
            logits, 
            tau=self.temperature, 
            hard=self.straight_through
        )  # [batch_size, seq_len, vocab_size]
        
        # 获取离散符号（用于监控和展示）
        symbols = soft_symbols.argmax(dim=-1)  # [batch_size, seq_len]
        
        # 更新符号使用统计
        if self.training:
            # 增加符号计数
            for i in range(batch_size):
                for j in range(self.seq_len):
                    self.symbol_counts[symbols[i, j]] += 1
                
                # 保存到历史记录
                idx = self.history_idx % 1000
                self.symbol_history[idx] = symbols[i]
                self.history_idx += 1
                
            self.comm_count += batch_size
        
        return soft_symbols, symbols
    
    def decode(self, soft_symbols: torch.Tensor) -> torch.Tensor:
        """将符号序列解码回隐藏状态"""
        batch_size = soft_symbols.size(0)
        
        # 从软符号获取嵌入表示
        # [batch_size, seq_len, vocab_size] x [vocab_size, embedding_dim]
        symbol_embeds = torch.matmul(
            soft_symbols, 
            self.symbol_embedding.weight
        )  # [batch_size, seq_len, embedding_dim]
        
        # 展平嵌入序列
        flat_embeds = symbol_embeds.view(batch_size, -1)  # [batch_size, seq_len*embedding_dim]
        
        # 解码回隐藏状态
        decoded = self.decoder(flat_embeds)  # [batch_size, hidden_size]
        
        return decoded
    
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
        
        # 编码为符号序列
        soft_symbols, _ = self.encode(hidden_states)  # [batch_size, seq_len, vocab_size]
        
        # 解码回隐藏状态
        decoded = self.decode(soft_symbols)  # [batch_size, hidden_size]
        
        # 扩展到序列维度
        batch_size, seq_len, _ = hidden_states.shape
        decoded = decoded.unsqueeze(1).expand(-1, seq_len, -1)
        
        # 添加残差连接（较小比例避免过强干扰）
        output = residual + 0.1 * decoded
        
        return output
    
    def get_symbol_stats(self) -> Dict[str, torch.Tensor]:
        """获取符号使用统计"""
        if self.comm_count == 0:
            return {
                "entropy": torch.tensor(0.0),
                "counts": self.symbol_counts.clone(),
                "total": self.comm_count.clone()
            }
            
        # 计算符号概率分布
        probs = self.symbol_counts / (self.seq_len * self.comm_count)
        
        # 计算熵
        entropy = -torch.sum(probs * torch.log2(probs + 1e-10))
        
        return {
            "entropy": entropy,
            "counts": self.symbol_counts.clone(),
            "total": self.comm_count.clone()
        }
    
    def get_recent_sequences(self, n: int = 100) -> torch.Tensor:
        """获取最近的n个符号序列"""
        n = min(n, int(self.comm_count.item()), 1000)
        if n == 0:
            return torch.zeros(0, self.seq_len, dtype=torch.long)
            
        idx = self.history_idx % 1000
        if idx >= n:
            return self.symbol_history[idx-n:idx]
        else:
            return torch.cat([
                self.symbol_history[-(n-idx):],
                self.symbol_history[:idx]
            ], dim=0) 