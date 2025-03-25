"""
智能体基类定义，所有智能体类型都应继承此类
"""

import torch
from typing import Dict, Any, List, Optional, Union, Tuple
from abc import ABC, abstractmethod


class AgentBase(ABC):
    """
    所有智能体的抽象基类，定义了智能体的基本接口
    
    智能体是能够接收观察、采取行动并与其他智能体通信的实体。
    """
    
    def __init__(self, agent_id: str, device: str = "cpu"):
        """
        初始化智能体
        
        参数:
            agent_id: 智能体的唯一标识符
            device: 运行智能体的设备(cpu或cuda)
        """
        self.agent_id = agent_id
        self.device = device
        self.observation_space = None
        self.action_space = None
        self.communication_space = None
        self.message_history = []  # 记录通信历史
    
    @abstractmethod
    def observe(self, observation: Any) -> None:
        """
        接收环境的观察
        
        参数:
            observation: 环境提供的观察数据
        """
        pass
    
    @abstractmethod
    def act(self) -> Any:
        """
        根据当前状态选择行动
        
        返回:
            智能体选择的行动
        """
        pass
    
    @abstractmethod
    def compose_message(self, recipient_id: Optional[str] = None) -> Any:
        """
        构建要发送给其他智能体的消息
        
        参数:
            recipient_id: 消息接收方的ID，None表示广播消息
            
        返回:
            构建的消息内容
        """
        pass
    
    @abstractmethod
    def receive_message(self, message: Any, sender_id: str) -> None:
        """
        接收来自其他智能体的消息
        
        参数:
            message: 消息内容
            sender_id: 发送者的ID
        """
        pass
    
    def save(self, path: str) -> None:
        """
        保存智能体的状态
        
        参数:
            path: 保存路径
        """
        raise NotImplementedError("保存方法需要在子类中实现")
    
    def load(self, path: str) -> None:
        """
        加载智能体的状态
        
        参数:
            path: 加载路径
        """
        raise NotImplementedError("加载方法需要在子类中实现")
    
    def reset(self) -> None:
        """
        重置智能体到初始状态
        """
        self.message_history = []
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """
        获取智能体的通信统计数据
        
        返回:
            包含通信统计信息的字典
        """
        return {
            "messages_sent": len([m for m in self.message_history if m["type"] == "sent"]),
            "messages_received": len([m for m in self.message_history if m["type"] == "received"]),
            "history": self.message_history
        } 