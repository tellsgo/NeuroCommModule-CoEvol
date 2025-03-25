"""
基于大语言模型的智能体实现
"""

import os
import torch
import torch.nn as nn
import json
from typing import Dict, Any, List, Optional, Union, Tuple
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from .agent_base import AgentBase


class LLMAgent(AgentBase):
    """
    基于预训练大语言模型的智能体
    
    这个智能体使用预训练的大语言模型作为其核心决策引擎，
    能够理解自然语言观察，生成自然语言动作，
    并可以通过特定方式与其他智能体通信。
    """
    
    def __init__(
        self,
        agent_id: str,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        communication_type: str = "discrete",
        vocab_size: int = 32,
        max_message_length: int = 8,
        temperature: float = 0.7,
        learning_rate: float = 1e-5
    ):
        """
        初始化LLM智能体
        
        参数:
            agent_id: 智能体的唯一标识符
            model_name: 预训练模型的名称或路径
            device: 运行设备
            communication_type: 通信类型，'discrete'或'vector'
            vocab_size: 通信词汇表大小(仅用于离散通信)
            max_message_length: 最大消息长度
            temperature: 生成时的温度参数
            learning_rate: 学习率
        """
        super().__init__(agent_id, device)
        
        # 加载预训练模型和分词器
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)
        
        # 通信配置
        self.communication_type = communication_type
        self.vocab_size = vocab_size
        self.max_message_length = max_message_length
        self.temperature = temperature
        
        # 如果是离散通信，初始化通信词汇表
        if communication_type == "discrete":
            # 创建一个特殊的通信词汇表
            # 这里我们简单地使用数字作为原始符号
            self.comm_vocab = {i: f"<SYMBOL_{i}>" for i in range(vocab_size)}
            self.inv_comm_vocab = {v: k for k, v in self.comm_vocab.items()}
            
            # 消息编码/解码层，这是智能体学习如何通信的关键部分
            self.message_encoder = nn.Sequential(
                nn.Linear(self.model.config.hidden_size, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, vocab_size * max_message_length)
            ).to(device)
            
            self.message_decoder = nn.Sequential(
                nn.Linear(vocab_size * max_message_length, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, self.model.config.hidden_size)
            ).to(device)
        
        elif communication_type == "vector":
            # 向量通信的编码/解码层
            vector_dim = 64  # 通信向量的维度
            self.vector_dim = vector_dim
            
            self.message_encoder = nn.Sequential(
                nn.Linear(self.model.config.hidden_size, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, vector_dim)
            ).to(device)
            
            self.message_decoder = nn.Sequential(
                nn.Linear(vector_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, self.model.config.hidden_size)
            ).to(device)
        
        # 设置优化器
        self.optimizer = torch.optim.Adam(
            list(self.message_encoder.parameters()) +
            list(self.message_decoder.parameters()) +
            list(self.model.parameters()),
            lr=learning_rate
        )
        
        # 当前状态
        self.current_observation = None
        self.received_messages = {}  # sender_id -> message
        self.context_history = []  # 对话历史
        self.context_window = 5  # 保留的最近对话轮次
        
        # 通信统计
        self.symbol_usage_count = torch.zeros(vocab_size, device=device)
        self.total_messages = 0
    
    def observe(self, observation: str) -> None:
        """
        接收环境的观察
        
        参数:
            observation: 文本形式的观察
        """
        self.current_observation = observation
        
        # 更新对话历史
        self.context_history.append({"role": "system", "content": observation})
        if len(self.context_history) > self.context_window * 2:
            self.context_history = self.context_history[-self.context_window * 2:]
    
    def act(self) -> str:
        """
        根据当前观察和接收到的消息生成行动
        
        返回:
            生成的行动文本
        """
        # 构建包含观察和收到消息的提示
        prompt = self._build_prompt()
        
        # 生成回复
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=100,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码生成的回复
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # 更新对话历史
        self.context_history.append({"role": "assistant", "content": response})
        if len(self.context_history) > self.context_window * 2:
            self.context_history = self.context_history[-self.context_window * 2:]
        
        return response
    
    def _build_prompt(self) -> str:
        """构建包含观察和消息的提示"""
        prompt = f"你是智能体 {self.agent_id}。\n\n"
        
        # 添加当前观察
        if self.current_observation:
            prompt += f"当前环境: {self.current_observation}\n\n"
        
        # 添加接收到的消息
        if self.received_messages:
            prompt += "接收到的消息:\n"
            for sender_id, message in self.received_messages.items():
                if isinstance(message, str):  # 已解码的消息
                    prompt += f"- 来自 {sender_id}: {message}\n"
                else:  # 原始消息格式
                    # 这里可以选择不同的表示方式
                    if self.communication_type == "discrete":
                        symbols = [self.comm_vocab.get(idx, f"<UNK_{idx}>") for idx in message]
                        prompt += f"- 来自 {sender_id}: {' '.join(symbols)}\n"
                    elif self.communication_type == "vector":
                        # 向量太复杂，只提示有消息
                        prompt += f"- 来自 {sender_id}: [向量消息]\n"
            prompt += "\n"
        
        # 添加对话历史
        for item in self.context_history[-self.context_window:]:
            if item["role"] == "system":
                prompt += f"环境: {item['content']}\n"
            elif item["role"] == "assistant":
                prompt += f"你的回应: {item['content']}\n"
            elif item["role"] == "message":
                prompt += f"消息 ({item['sender']}): {item['content']}\n"
        
        prompt += "\n基于当前情况，你的行动是:"
        
        return prompt
    
    def compose_message(self, recipient_id: Optional[str] = None) -> Any:
        """
        构建要发送给其他智能体的消息
        
        参数:
            recipient_id: 接收方ID
            
        返回:
            如果是discrete模式，返回符号ID序列；如果是vector模式，返回向量
        """
        # 生成隐藏状态表示
        prompt = f"你是智能体 {self.agent_id}。请生成一条简短消息传递给其他智能体，帮助解决当前任务。"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        hidden_states = outputs.last_hidden_state.mean(dim=1)  # [1, hidden_size]
        
        # 编码消息
        if self.communication_type == "discrete":
            logits = self.message_encoder(hidden_states)  # [1, vocab_size * max_message_length]
            logits = logits.view(-1, self.max_message_length, self.vocab_size)  # [1, max_len, vocab_size]
            
            # 使用argmax或采样获取符号ID
            if self.training:
                # 采样模式，使用Gumbel-Softmax使其可微分
                temperature = 1.0
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
                soft_logits = (logits + gumbel_noise) / temperature
                soft_symbols = torch.softmax(soft_logits, dim=-1)
                
                # 获取硬符号ID用于记录
                symbols = torch.argmax(soft_symbols, dim=-1).squeeze(0).cpu().numpy()  # [max_len]
            else:
                # 测试模式直接使用argmax
                symbols = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()  # [max_len]
            
            # 更新符号使用统计
            for sym in symbols:
                if 0 <= sym < self.vocab_size:
                    self.symbol_usage_count[sym] += 1
            self.total_messages += 1
            
            # 记录消息历史
            self.message_history.append({
                "type": "sent",
                "recipient": recipient_id,
                "content": symbols.tolist(),
                "raw_content": [self.comm_vocab.get(idx, f"<UNK_{idx}>") for idx in symbols]
            })
            
            return symbols
        
        elif self.communication_type == "vector":
            # 生成通信向量
            vector = self.message_encoder(hidden_states).squeeze(0)  # [vector_dim]
            
            # 记录消息历史
            self.message_history.append({
                "type": "sent",
                "recipient": recipient_id,
                "content": vector.cpu().numpy().tolist()
            })
            
            return vector
    
    def receive_message(self, message: Any, sender_id: str) -> None:
        """
        接收来自其他智能体的消息
        
        参数:
            message: 消息内容，离散符号序列或向量
            sender_id: 发送者ID
        """
        # 存储原始消息
        self.received_messages[sender_id] = message
        
        # 记录消息历史
        if self.communication_type == "discrete":
            self.message_history.append({
                "type": "received",
                "sender": sender_id,
                "content": message.tolist() if isinstance(message, np.ndarray) else message,
                "raw_content": [self.comm_vocab.get(idx, f"<UNK_{idx}>") for idx in message]
            })
        else:
            self.message_history.append({
                "type": "received",
                "sender": sender_id,
                "content": message.cpu().numpy().tolist() if torch.is_tensor(message) else message
            })
        
        # 解码消息并添加到上下文历史
        decoded_message = self._decode_message(message)
        if decoded_message:
            self.context_history.append({
                "role": "message", 
                "sender": sender_id, 
                "content": decoded_message
            })
            if len(self.context_history) > self.context_window * 2:
                self.context_history = self.context_history[-self.context_window * 2:]
    
    def _decode_message(self, message: Any) -> Optional[str]:
        """
        尝试解码收到的消息为自然语言表示
        这是智能体学习如何理解其他智能体发送的符号的关键部分
        
        参数:
            message: 原始消息
            
        返回:
            解码后的消息文本，如果无法解码则返回None
        """
        # 如果智能体尚未学习如何解码，可能返回None或占位符
        # 随着训练进行，智能体应该学会更好地解码这些消息
        
        if self.communication_type == "discrete":
            # 将符号ID序列转换为模型可以处理的表示
            batch_size = 1
            message_tensor = torch.zeros(batch_size, self.max_message_length, self.vocab_size, device=self.device)
            
            # 将符号ID转换为one-hot表示
            for i, sym_id in enumerate(message):
                if i >= self.max_message_length:
                    break
                if 0 <= sym_id < self.vocab_size:
                    message_tensor[0, i, sym_id] = 1.0
            
            # 展平消息张量
            flat_message = message_tensor.view(batch_size, -1)  # [1, max_len * vocab_size]
            
            # 解码为隐藏状态
            hidden = self.message_decoder(flat_message)  # [1, hidden_size]
            
            # 使用语言模型生成可能的解释
            # 在实际训练中，这部分会通过梯度下降优化
            
            # 目前，我们可以返回原始符号的字符串表示
            symbol_strs = [self.comm_vocab.get(int(idx), f"<UNK_{idx}>") for idx in message]
            return " ".join(symbol_strs)
        
        elif self.communication_type == "vector":
            # 将向量转换为张量
            if not torch.is_tensor(message):
                message = torch.tensor(message, device=self.device).float()
            
            if message.dim() == 1:
                message = message.unsqueeze(0)  # 添加批处理维度
            
            # 解码为隐藏状态
            hidden = self.message_decoder(message)  # [1, hidden_size]
            
            # 返回向量的简单字符串表示
            return f"[向量消息，维度={message.shape[-1]}]"
    
    def update(self, reward: float) -> None:
        """
        基于接收到的奖励更新智能体参数
        
        参数:
            reward: 环境提供的奖励值
        """
        # 在实际训练循环中实现
        pass
    
    def save(self, path: str) -> None:
        """
        保存智能体的状态
        
        参数:
            path: 保存路径
        """
        os.makedirs(path, exist_ok=True)
        
        # 保存模型
        model_path = os.path.join(path, f"{self.agent_id}_model")
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        # 保存通信模块
        if self.communication_type == "discrete":
            torch.save({
                "message_encoder": self.message_encoder.state_dict(),
                "message_decoder": self.message_decoder.state_dict(),
                "symbol_usage_count": self.symbol_usage_count,
                "total_messages": self.total_messages
            }, os.path.join(path, f"{self.agent_id}_comm_modules.pt"))
        elif self.communication_type == "vector":
            torch.save({
                "message_encoder": self.message_encoder.state_dict(),
                "message_decoder": self.message_decoder.state_dict()
            }, os.path.join(path, f"{self.agent_id}_comm_modules.pt"))
        
        # 保存通信历史和其他状态
        with open(os.path.join(path, f"{self.agent_id}_history.json"), "w") as f:
            # 转换为可序列化的格式
            serializable_history = []
            for msg in self.message_history:
                new_msg = msg.copy()
                # 转换numpy数组或张量
                if "content" in new_msg and (isinstance(new_msg["content"], np.ndarray) or torch.is_tensor(new_msg["content"])):
                    if torch.is_tensor(new_msg["content"]):
                        new_msg["content"] = new_msg["content"].cpu().numpy().tolist()
                    else:
                        new_msg["content"] = new_msg["content"].tolist()
                serializable_history.append(new_msg)
            
            json.dump({
                "message_history": serializable_history,
                "context_history": self.context_history
            }, f, indent=2)
    
    def load(self, path: str) -> None:
        """
        加载智能体的状态
        
        参数:
            path: 加载路径
        """
        # 加载模型
        model_path = os.path.join(path, f"{self.agent_id}_model")
        if os.path.exists(model_path):
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model.to(self.device)
        
        # 加载通信模块
        comm_path = os.path.join(path, f"{self.agent_id}_comm_modules.pt")
        if os.path.exists(comm_path):
            state_dict = torch.load(comm_path, map_location=self.device)
            self.message_encoder.load_state_dict(state_dict["message_encoder"])
            self.message_decoder.load_state_dict(state_dict["message_decoder"])
            if "symbol_usage_count" in state_dict:
                self.symbol_usage_count = state_dict["symbol_usage_count"]
            if "total_messages" in state_dict:
                self.total_messages = state_dict["total_messages"]
        
        # 加载通信历史和其他状态
        history_path = os.path.join(path, f"{self.agent_id}_history.json")
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                data = json.load(f)
                self.message_history = data.get("message_history", [])
                self.context_history = data.get("context_history", [])
    
    def get_symbol_distribution(self) -> Dict[str, Any]:
        """
        获取符号使用分布的统计信息
        
        返回:
            包含符号使用统计的字典
        """
        if self.communication_type != "discrete":
            return {"error": "只有离散通信模式才有符号分布"}
        
        if self.total_messages == 0:
            return {
                "entropy": 0.0,
                "distribution": [0.0] * self.vocab_size,
                "total_messages": 0
            }
        
        # 计算符号分布
        distribution = self.symbol_usage_count / (self.total_messages * self.max_message_length)
        distribution = distribution.cpu().numpy()
        
        # 计算熵
        eps = 1e-10  # 防止log(0)
        entropy = -np.sum(distribution * np.log2(distribution + eps))
        
        return {
            "entropy": float(entropy),
            "distribution": distribution.tolist(),
            "total_messages": int(self.total_messages)
        }
    
    def reset(self) -> None:
        """重置智能体状态"""
        super().reset()
        self.current_observation = None
        self.received_messages = {}
        self.context_history = [] 