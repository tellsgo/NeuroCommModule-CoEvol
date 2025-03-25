"""
多智能体通信协议模块
"""

from .message import Message
from .discrete_channel import DiscreteChannel 
from .vector_channel import VectorChannel

__all__ = ['Message', 'DiscreteChannel', 'VectorChannel'] 