import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np


class DialogueDataset(Dataset):
    """对话数据集类，用于模型对话任务的训练和评估"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        """
        初始化对话数据集
        
        参数:
            data_path: 数据文件路径
            tokenizer: 用于分词的tokenizer
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                self.dialogues = json.load(f)
        else:
            # 如果文件不存在，创建一些示例对话
            print(f"警告: 找不到数据文件 {data_path}，将使用合成对话数据")
            self.dialogues = self._create_synthetic_dialogues()
    
    def _create_synthetic_dialogues(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """创建合成对话数据用于演示"""
        dialogues = []
        
        templates = [
            {"prompt": "你能告诉我关于{topic}的信息吗？", 
             "completion": "关于{topic}，主要有以下几点值得了解: {fact1}，{fact2}和{fact3}。"},
            {"prompt": "我想学习{skill}，应该从哪里开始？", 
             "completion": "学习{skill}的推荐步骤是: 首先，{step1}；其次，{step2}；最后，{step3}。"},
            {"prompt": "{greeting}！最近过得怎么样？", 
             "completion": "{greeting}！我最近{status}，正在{activity}。你呢？"},
            {"prompt": "你认为{question}？", 
             "completion": "关于{question}，我的看法是{opinion}。这是因为{reason1}和{reason2}。"}
        ]
        
        topics = ["人工智能", "量子计算", "深度学习", "神经科学", "进化心理学"]
        facts = ["它是一个快速发展的领域", "它有很多实际应用", "它融合了多学科知识", "它正在改变未来"]
        skills = ["编程", "机器学习", "数据分析", "写作", "绘画"]
        steps = ["掌握基础知识", "进行实际练习", "参与相关项目", "向专家学习", "不断复习和提升"]
        greetings = ["你好", "嗨", "早上好", "晚上好", "下午好"]
        statuses = ["很充实", "有点忙", "还不错", "过得很好", "在学习新东西"]
        activities = ["学习新技能", "做一个项目", "阅读一本好书", "探索新兴技术", "参加线上课程"]
        questions = ["AI将如何改变未来", "人类意识是什么", "如何平衡工作和生活", "教育应该如何改革"]
        opinions = ["这是个复杂问题", "这需要综合考虑", "这有多个角度", "这取决于具体情况"]
        reasons = ["历史经验表明", "研究数据显示", "从实践来看", "基于理论分析", "考虑多方利益"]
        
        # 生成示例对话
        for _ in range(num_samples):
            template = random.choice(templates)
            dialogue = {}
            
            if "topic" in template["prompt"]:
                topic = random.choice(topics)
                fact1 = random.choice(facts)
                fact2 = random.choice(facts)
                while fact2 == fact1:
                    fact2 = random.choice(facts)
                fact3 = random.choice(facts)
                while fact3 == fact1 or fact3 == fact2:
                    fact3 = random.choice(facts)
                    
                dialogue["prompt"] = template["prompt"].format(topic=topic)
                dialogue["completion"] = template["completion"].format(
                    topic=topic, fact1=fact1, fact2=fact2, fact3=fact3
                )
                
            elif "skill" in template["prompt"]:
                skill = random.choice(skills)
                step1 = random.choice(steps)
                step2 = random.choice(steps)
                while step2 == step1:
                    step2 = random.choice(steps)
                step3 = random.choice(steps)
                while step3 == step1 or step3 == step2:
                    step3 = random.choice(steps)
                    
                dialogue["prompt"] = template["prompt"].format(skill=skill)
                dialogue["completion"] = template["completion"].format(
                    skill=skill, step1=step1, step2=step2, step3=step3
                )
                
            elif "greeting" in template["prompt"]:
                greeting = random.choice(greetings)
                status = random.choice(statuses)
                activity = random.choice(activities)
                
                dialogue["prompt"] = template["prompt"].format(greeting=greeting)
                dialogue["completion"] = template["completion"].format(
                    greeting=greeting, status=status, activity=activity
                )
                
            elif "question" in template["prompt"]:
                question = random.choice(questions)
                opinion = random.choice(opinions)
                reason1 = random.choice(reasons)
                reason2 = random.choice(reasons)
                while reason2 == reason1:
                    reason2 = random.choice(reasons)
                    
                dialogue["prompt"] = template["prompt"].format(question=question)
                dialogue["completion"] = template["completion"].format(
                    question=question, opinion=opinion, reason1=reason1, reason2=reason2
                )
            
            dialogues.append(dialogue)
            
        return dialogues
    
    def __len__(self):
        return len(self.dialogues)
    
    def __getitem__(self, idx):
        dialogue = self.dialogues[idx]
        prompt = dialogue["prompt"]
        completion = dialogue["completion"]
        
        # 编码输入序列（模型A的输入：提示）
        encoded_input = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 编码目标序列（模型B的输入：提示+部分回复，用于生成下文）
        # 对于训练目的，我们使用提示词加上部分回复作为模型B的输入
        partial_completion = completion[:len(completion)//2]
        encoded_partial = self.tokenizer(
            prompt + " " + partial_completion,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 编码完整目标（用于计算损失）
        encoded_target = self.tokenizer(
            completion,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "model_a": {
                "input_ids": encoded_input["input_ids"].squeeze(0),
                "attention_mask": encoded_input["attention_mask"].squeeze(0)
            },
            "model_b": {
                "input_ids": encoded_partial["input_ids"].squeeze(0),
                "attention_mask": encoded_partial["attention_mask"].squeeze(0)
            },
            "target": {
                "input_ids": encoded_target["input_ids"].squeeze(0),
                "attention_mask": encoded_target["attention_mask"].squeeze(0)
            },
            "prompt": prompt,
            "completion": completion,
            "partial_completion": partial_completion
        }


class DialogueCompletionTask:
    """对话完成任务，测试模型协作生成连贯对话的能力"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化对话任务
        
        参数:
            config: 配置字典
        """
        self.config = config
        self.batch_size = config.get("batch_size", 8)
        
        # 对话数据文件路径
        self.data_path = config.get("data_path", "data/dialogue_samples.json")
        
        # 如果数据目录不存在，创建它
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        
        # 一些计算损失的超参数
        self.coherence_weight = config.get("coherence_weight", 0.3)  # 连贯性损失权重
        self.diversity_weight = config.get("diversity_weight", 0.1)  # 多样性损失权重
        
        # 创建分词器引用（实际分词器将在使用时由模型提供）
        self.tokenizer = None
    
    def get_train_dataloader(self) -> DataLoader:
        """获取训练数据加载器"""
        # 懒加载数据集
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            model_name = self.config.get("base_model", "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # 确保分词器有pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 创建数据集
        dataset = DialogueDataset(
            self.data_path,
            self.tokenizer,
            max_length=self.config.get("max_length", 128)
        )
        
        # 数据分割（80%用于训练）
        train_size = int(0.8 * len(dataset))
        train_dataset = torch.utils.data.Subset(dataset, range(train_size))
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0  # 避免多进程问题
        )
        
        return train_loader
    
    def get_eval_dataloader(self) -> DataLoader:
        """获取评估数据加载器"""
        # 懒加载数据集
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            model_name = self.config.get("base_model", "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # 确保分词器有pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 创建数据集
        dataset = DialogueDataset(
            self.data_path,
            self.tokenizer,
            max_length=self.config.get("max_length", 128)
        )
        
        # 数据分割（20%用于评估）
        train_size = int(0.8 * len(dataset))
        eval_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
        
        # 创建数据加载器
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0  # 避免多进程问题
        )
        
        return eval_loader
    
    def compute_loss(self, batch, outputs_a, outputs_b, comm_vector_a=None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算对话任务的损失
        
        参数:
            batch: 一个批次的数据
            outputs_a: 模型A的输出
            outputs_b: 模型B的输出
            comm_vector_a: 模型A的通信向量（可选）
            
        返回:
            总损失和损失详情字典
        """
        device = outputs_a["logits"].device
        
        # 目标序列
        target_ids = batch["target"]["input_ids"].to(device)
        target_mask = batch["target"]["attention_mask"].to(device)
        
        # 计算语言模型损失（使用模型B的输出预测完整目标）
        # 忽略pad标记的损失
        lm_loss = F.cross_entropy(
            outputs_b["logits"].view(-1, outputs_b["logits"].size(-1)),
            target_ids.view(-1),
            ignore_index=self.tokenizer.pad_token_id
        )
        
        # 计算连贯性损失（如果通信向量可用）
        coherence_loss = torch.tensor(0.0, device=device)
        if comm_vector_a is not None:
            # 简单的一致性损失实现：鼓励通信向量与模型B的隐藏状态相似
            if outputs_b["hidden_states"] is not None:
                b_hidden = outputs_b["hidden_states"][-1].mean(dim=1)  # [batch_size, hidden_size]
                if b_hidden.shape[-1] != comm_vector_a.shape[-1]:
                    # 如果维度不同，使用线性投影
                    projection = nn.Linear(b_hidden.shape[-1], comm_vector_a.shape[-1], device=device)
                    b_hidden_proj = projection(b_hidden)
                    coherence_loss = F.mse_loss(b_hidden_proj, comm_vector_a)
                else:
                    coherence_loss = F.mse_loss(b_hidden, comm_vector_a)
        
        # 简单的多样性损失（鼓励分布更加多样化）
        probs = F.softmax(outputs_b["logits"], dim=-1)
        diversity_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-10), dim=-1))
        
        # 合并损失
        total_loss = lm_loss + self.coherence_weight * coherence_loss - self.diversity_weight * diversity_loss
        
        loss_details = {
            "lm_loss": lm_loss.item(),
            "coherence_loss": coherence_loss.item(),
            "diversity_loss": diversity_loss.item(),
            "total_loss": total_loss.item()
        }
        
        return total_loss, loss_details
    
    def compute_metrics(self, batch, outputs_a, outputs_b, comm_vector_a=None) -> Dict[str, float]:
        """
        计算评估指标
        
        参数:
            batch: 一个批次的数据
            outputs_a: 模型A的输出
            outputs_b: 模型B的输出
            comm_vector_a: 模型A的通信向量（可选）
            
        返回:
            评估指标字典
        """
        device = outputs_a["logits"].device
        
        # 目标序列
        target_ids = batch["target"]["input_ids"].to(device)
        target_mask = batch["target"]["attention_mask"].to(device)
        
        # 计算语言模型困惑度
        with torch.no_grad():
            lm_loss = F.cross_entropy(
                outputs_b["logits"].view(-1, outputs_b["logits"].size(-1)),
                target_ids.view(-1),
                ignore_index=self.tokenizer.pad_token_id,
                reduction='mean'
            )
            perplexity = torch.exp(lm_loss).item()
        
        # 计算准确率（模型B预测的正确标记比例）
        pred_ids = outputs_b["logits"].argmax(dim=-1)
        correct = (pred_ids == target_ids) & target_mask.bool()
        accuracy = correct.sum().float() / target_mask.sum().float()
        
        # 计算连贯性得分（如果通信向量可用）
        coherence_score = 0.0
        if comm_vector_a is not None and outputs_b["hidden_states"] is not None:
            with torch.no_grad():
                # 连贯性是通信向量与模型B隐藏状态的相似度
                b_hidden = outputs_b["hidden_states"][-1].mean(dim=1)
                if b_hidden.shape[-1] != comm_vector_a.shape[-1]:
                    # 如果维度不同，使用线性投影
                    projection = nn.Linear(b_hidden.shape[-1], comm_vector_a.shape[-1], device=device)
                    b_hidden_proj = projection(b_hidden)
                    coherence_score = F.cosine_similarity(b_hidden_proj, comm_vector_a, dim=1).mean().item()
                else:
                    coherence_score = F.cosine_similarity(b_hidden, comm_vector_a, dim=1).mean().item()
        
        return {
            "perplexity": perplexity,
            "accuracy": accuracy.item(),
            "coherence_score": coherence_score
        } 