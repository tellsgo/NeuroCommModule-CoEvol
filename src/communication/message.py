"""
消息类，定义智能体之间的通信消息格式
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Union
import time
import numpy as np


@dataclass
class Message:
    """智能体间传递的消息"""
    
    # 消息元数据
    sender_id: str
    recipient_id: Optional[str] = None  # None表示广播消息
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: f"msg_{time.time()}")
    
    # 消息内容可以是不同类型
    content: Any = None  # 可以是符号序列、向量、文本等
    content_type: str = "unknown"  # discrete, vector, text
    
    # 可选的元信息
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后的处理，自动推断内容类型"""
        if self.content_type == "unknown" and self.content is not None:
            if isinstance(self.content, str):
                self.content_type = "text"
            elif isinstance(self.content, np.ndarray) or isinstance(self.content, list):
                # 判断是离散符号序列还是连续向量
                if all(isinstance(x, (int, np.integer)) for x in np.asarray(self.content).flatten()):
                    self.content_type = "discrete"
                else:
                    self.content_type = "vector"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        content = self.content
        if isinstance(content, np.ndarray):
            content = content.tolist()
            
        return {
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "timestamp": self.timestamp,
            "message_id": self.message_id,
            "content": content,
            "content_type": self.content_type,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """从字典创建消息对象"""
        content = data["content"]
        if data["content_type"] == "discrete" and isinstance(content, list):
            content = np.array(content, dtype=np.int32)
            
        return cls(
            sender_id=data["sender_id"],
            recipient_id=data["recipient_id"],
            timestamp=data["timestamp"],
            message_id=data["message_id"],
            content=content,
            content_type=data["content_type"],
            metadata=data.get("metadata", {})
        )
    
    def __str__(self) -> str:
        """字符串表示"""
        recipient_str = self.recipient_id if self.recipient_id else "broadcast"
        if self.content_type == "text":
            content_preview = self.content[:30] + "..." if len(self.content) > 30 else self.content
            return f"Message({self.message_id[:8]}, {self.sender_id} -> {recipient_str}, text: \"{content_preview}\")"
        else:
            content_shape = "unknown"
            if hasattr(self.content, "shape"):
                content_shape = str(self.content.shape)
            elif isinstance(self.content, list):
                content_shape = str(len(self.content))
            return f"Message({self.message_id[:8]}, {self.sender_id} -> {recipient_str}, {self.content_type}, shape: {content_shape})"