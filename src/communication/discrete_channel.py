"""
离散通信信道，处理智能体间的离散符号通信
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import random
from collections import defaultdict

from .message import Message


class DiscreteChannel:
    """
    离散符号通信信道
    
    该通信信道允许智能体使用离散符号序列进行通信，
    可以模拟有限带宽、噪声干扰等现实通信条件。
    """
    
    def __init__(
        self,
        vocab_size: int = 32,
        max_message_length: int = 8,
        noise_level: float = 0.0,
        drop_rate: float = 0.0,
        bandwidth_limit: Optional[int] = None
    ):
        """
        初始化离散通信信道
        
        参数:
            vocab_size: 词汇表大小，即可能的符号数量
            max_message_length: 最大消息长度
            noise_level: 噪声水平，表示符号被随机替换的概率
            drop_rate: 消息丢失率，表示消息被完全丢弃的概率
            bandwidth_limit: 带宽限制，每轮可发送的最大总符号数
        """
        self.vocab_size = vocab_size
        self.max_message_length = max_message_length
        self.noise_level = noise_level
        self.drop_rate = drop_rate
        self.bandwidth_limit = bandwidth_limit
        
        # 通信统计
        self.message_count = 0
        self.symbol_count = 0
        self.dropped_messages = 0
        self.corrupted_symbols = 0
        
        # 消息队列 - 用于存储带宽限制时的延迟消息
        self.message_queue = []
        
        # 符号使用统计
        self.symbol_usage = defaultdict(int)
    
    def send(self, message: Message) -> Optional[Message]:
        """
        发送消息，可能会添加噪声或丢弃
        
        参数:
            message: 要发送的消息
            
        返回:
            处理后的消息，如果消息被丢弃则返回None
        """
        # 检查消息类型
        if message.content_type != "discrete":
            raise ValueError(f"DiscreteChannel只能处理discrete类型的消息，收到: {message.content_type}")
        
        # 可能丢弃消息
        if random.random() < self.drop_rate:
            self.dropped_messages += 1
            return None
        
        # 获取内容
        content = message.content
        if isinstance(content, list):
            content = np.array(content, dtype=np.int32)
        
        # 截断过长消息
        if len(content) > self.max_message_length:
            content = content[:self.max_message_length]
        
        # 可能添加噪声
        if self.noise_level > 0:
            noisy_content = content.copy()
            for i in range(len(noisy_content)):
                if random.random() < self.noise_level:
                    # 替换为随机符号
                    original = noisy_content[i]
                    noisy_content[i] = random.randint(0, self.vocab_size - 1)
                    if noisy_content[i] != original:
                        self.corrupted_symbols += 1
            content = noisy_content
        
        # 更新统计信息
        self.message_count += 1
        self.symbol_count += len(content)
        for symbol in content:
            self.symbol_usage[int(symbol)] += 1
        
        # 创建处理后的消息
        processed_message = Message(
            sender_id=message.sender_id,
            recipient_id=message.recipient_id,
            content=content,
            content_type="discrete",
            metadata={**message.metadata, "processed": True}
        )
        
        return processed_message
    
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
        
        # 计算可用带宽
        total_symbols = self.bandwidth_limit
        sent_messages = []
        remaining_queue = []
        
        for msg in self.message_queue:
            content_length = len(msg.content)
            if content_length <= total_symbols:
                # 可以发送此消息
                sent_messages.append(msg)
                total_symbols -= content_length
            else:
                # 带宽不足，消息留在队列中
                remaining_queue.append(msg)
        
        # 更新队列
        self.message_queue = remaining_queue
        
        return sent_messages
    
    def get_stats(self) -> Dict[str, Any]:
        """获取通信统计信息"""
        stats = {
            "message_count": self.message_count,
            "symbol_count": self.symbol_count,
            "dropped_messages": self.dropped_messages,
            "corrupted_symbols": self.corrupted_symbols,
            "queue_size": len(self.message_queue)
        }
        
        # 计算符号分布
        if self.symbol_count > 0:
            distribution = {str(i): 0 for i in range(self.vocab_size)}
            for symbol, count in self.symbol_usage.items():
                if 0 <= symbol < self.vocab_size:
                    distribution[str(symbol)] = count / self.symbol_count
            
            # 计算熵
            entropy = 0
            for prob in distribution.values():
                if prob > 0:
                    entropy -= prob * np.log2(prob)
                    
            stats["symbol_distribution"] = distribution
            stats["entropy"] = entropy
        else:
            stats["symbol_distribution"] = {str(i): 0 for i in range(self.vocab_size)}
            stats["entropy"] = 0
        
        return stats
    
    def reset(self) -> None:
        """重置信道状态"""
        self.message_count = 0
        self.symbol_count = 0
        self.dropped_messages = 0
        self.corrupted_symbols = 0
        self.message_queue = []
        self.symbol_usage = defaultdict(int) 