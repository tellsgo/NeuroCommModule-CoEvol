"""
向量通信信道，处理智能体间的连续向量通信
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import random

from .message import Message


class VectorChannel:
    """
    连续向量通信信道
    
    该通信信道允许智能体使用连续向量进行通信，
    可以模拟带宽限制、噪声干扰等现实通信条件。
    """
    
    def __init__(
        self,
        vector_dim: int = 64,
        noise_level: float = 0.0,
        drop_rate: float = 0.0,
        bandwidth_limit: Optional[int] = None,
        quantization_bits: Optional[int] = None
    ):
        """
        初始化向量通信信道
        
        参数:
            vector_dim: 向量维度
            noise_level: 噪声水平，表示添加的高斯噪声标准差
            drop_rate: 消息丢失率，表示消息被完全丢弃的概率
            bandwidth_limit: 带宽限制，每轮可发送的最大向量数
            quantization_bits: 量化位数，None表示不进行量化
        """
        self.vector_dim = vector_dim
        self.noise_level = noise_level
        self.drop_rate = drop_rate
        self.bandwidth_limit = bandwidth_limit
        self.quantization_bits = quantization_bits
        
        # 通信统计
        self.message_count = 0
        self.vector_count = 0
        self.dropped_messages = 0
        
        # 消息队列 - 用于存储带宽限制时的延迟消息
        self.message_queue = []
        
        # 向量统计
        self.vector_sum = None  # 用于计算均值
        self.vector_sum_sq = None  # 用于计算方差
    
    def send(self, message: Message) -> Optional[Message]:
        """
        发送消息，可能会添加噪声或丢弃
        
        参数:
            message: 要发送的消息
            
        返回:
            处理后的消息，如果消息被丢弃则返回None
        """
        # 检查消息类型
        if message.content_type != "vector":
            raise ValueError(f"VectorChannel只能处理vector类型的消息，收到: {message.content_type}")
        
        # 可能丢弃消息
        if random.random() < self.drop_rate:
            self.dropped_messages += 1
            return None
        
        # 获取内容
        content = message.content
        if not isinstance(content, np.ndarray):
            if torch.is_tensor(content):
                content = content.detach().cpu().numpy()
            else:
                content = np.array(content, dtype=np.float32)
        
        # 确保形状正确
        if content.shape[-1] != self.vector_dim:
            raise ValueError(f"向量维度不匹配: 预期 {self.vector_dim}, 实际 {content.shape[-1]}")
        
        # 可能添加噪声
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, content.shape)
            content = content + noise
        
        # 如果启用量化
        if self.quantization_bits is not None:
            content = self._quantize(content, self.quantization_bits)
        
        # 更新统计信息
        self.message_count += 1
        self.vector_count += 1
        
        # 更新向量统计
        if self.vector_sum is None:
            self.vector_sum = content.copy()
            self.vector_sum_sq = content ** 2
        else:
            self.vector_sum += content
            self.vector_sum_sq += content ** 2
        
        # 创建处理后的消息
        processed_message = Message(
            sender_id=message.sender_id,
            recipient_id=message.recipient_id,
            content=content,
            content_type="vector",
            metadata={**message.metadata, "processed": True}
        )
        
        return processed_message
    
    def _quantize(self, vector: np.ndarray, bits: int) -> np.ndarray:
        """
        对向量进行量化
        
        参数:
            vector: 要量化的向量
            bits: 量化位数
            
        返回:
            量化后的向量
        """
        # 计算量化区间
        levels = 2 ** bits
        min_val = vector.min()
        max_val = vector.max()
        if min_val == max_val:
            return vector
        
        # 量化
        scale = (max_val - min_val) / (levels - 1)
        quantized = np.round((vector - min_val) / scale) * scale + min_val
        
        return quantized
    
    def broadcast(self, message: Message, recipients: List[str]) -> Dict[str, Optional[Message]]:
        """
        广播消息给多个接收者
        
        参数:
            message: 要广播的消息
            recipients: 接收者ID列表
            
        返回:
            接收者ID到接收消息的映射，如果某个接收者未收到消息则对应值为None
        """
        results = {}
        for recipient in recipients:
            # 为每个接收者创建消息副本
            msg_copy = Message(
                sender_id=message.sender_id,
                recipient_id=recipient,
                content=message.content.copy() if hasattr(message.content, "copy") else message.content,
                content_type=message.content_type,
                metadata=message.metadata.copy()
            )
            # 发送消息（可能添加噪声或丢弃）
            results[recipient] = self.send(msg_copy)
        
        return results
    
    def process_bandwidth_limit(self, messages: List[Message]) -> List[Message]:
        """
        处理带宽限制，返回本轮可以发送的消息
        
        参数:
            messages: 要发送的消息列表
            
        返回:
            实际可以发送的消息列表
        """
        if self.bandwidth_limit is None:
            return messages
        
        # 将新消息添加到队列
        self.message_queue.extend(messages)
        
        # 只发送前bandwidth_limit条消息
        sent_messages = self.message_queue[:self.bandwidth_limit]
        self.message_queue = self.message_queue[self.bandwidth_limit:]
        
        return sent_messages
    
    def get_stats(self) -> Dict[str, Any]:
        """获取通信统计信息"""
        stats = {
            "message_count": self.message_count,
            "vector_count": self.vector_count,
            "dropped_messages": self.dropped_messages,
            "queue_size": len(self.message_queue)
        }
        
        # 计算向量分布统计
        if self.vector_count > 0 and self.vector_sum is not None:
            mean = self.vector_sum / self.vector_count
            variance = (self.vector_sum_sq / self.vector_count) - (mean ** 2)
            std = np.sqrt(np.maximum(variance, 0))  # 避免数值误差导致的负方差
            
            stats["vector_mean"] = mean.tolist()
            stats["vector_std"] = std.tolist()
            
            # 计算均值的范数
            mean_norm = np.linalg.norm(mean)
            std_norm = np.linalg.norm(std)
            
            stats["mean_norm"] = float(mean_norm)
            stats["std_norm"] = float(std_norm)
        
        return stats
    
    def reset(self) -> None:
        """重置信道状态"""
        self.message_count = 0
        self.vector_count = 0
        self.dropped_messages = 0
        self.message_queue = []
        self.vector_sum = None
        self.vector_sum_sq = None 