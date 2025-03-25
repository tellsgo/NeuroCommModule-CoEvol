import os
import torch
import json
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
from tqdm import tqdm

from ..models import ModelWrapper, TinyLlamaWrapper
from ..modules import AdapterModule, VectorCommModule, SymbolCommModule
from ..visualization import visualize_comm_vectors, plot_symbol_usage, plot_parameter_changes


class ExperimentRunner:
    """
    实验运行器，用于设置和执行通信模块协同进化实验
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化实验运行器
        
        参数:
            config: 实验配置字典，包含以下键:
                - base_model: 基础模型名称
                - module_type: 通信模块类型 ('adapter', 'vector', 'symbol')
                - module_position: 在第几层插入模块
                - task: 任务名称
                - 其他模块特定参数
        """
        self.config = config
        self.setup_logging()
        self.setup_paths()
        
        # 保存配置
        with open(os.path.join(self.exp_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
            
        # 初始化模型
        self.setup_models()
        
        # 初始化通信模块
        self.setup_comm_modules()
        
        # 初始化优化器
        self.setup_optimizers()
        
        # 初始化任务
        self.setup_task()
        
        # 训练状态跟踪
        self.current_epoch = 0
        self.global_step = 0
        self.best_metrics = {}
        self.history = {
            "train_loss": [],
            "eval_metrics": [],
            "param_changes": []
        }
        
        self.logger.info(f"实验初始化完成: {self.exp_name}")
    
    def setup_logging(self):
        """设置日志记录"""
        self.exp_name = f"{self.config['module_type']}_{self.config['task']}_{datetime.now().strftime('%m%d_%H%M')}"
        
        # 创建logger
        self.logger = logging.getLogger(self.exp_name)
        self.logger.setLevel(logging.INFO)
        
        # 添加控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
    
    def setup_paths(self):
        """设置实验目录结构"""
        # 实验根目录
        self.exp_dir = os.path.join("experiments", self.exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # 模型检查点目录
        self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 可视化结果目录
        self.vis_dir = os.path.join(self.exp_dir, "visualizations")
        os.makedirs(self.vis_dir, exist_ok=True)
        
        # 添加文件日志处理器
        file_handler = logging.FileHandler(os.path.join(self.exp_dir, "experiment.log"))
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
    
    def setup_models(self):
        """初始化模型"""
        base_model = self.config.get("base_model", "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
        
        # 确定设备
        if torch.cuda.is_available():
            self.device = "cuda"
            self.logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            self.logger.info("GPU不可用，使用CPU")
            
        # 初始化模型A和模型B
        self.logger.info(f"初始化模型A: {base_model}")
        if "TinyLlama" in base_model:
            self.model_a = TinyLlamaWrapper(base_model, self.device)
        else:
            raise ValueError(f"不支持的模型类型: {base_model}")
            
        self.logger.info(f"初始化模型B: {base_model}")
        if "TinyLlama" in base_model:
            self.model_b = TinyLlamaWrapper(base_model, self.device)
        else:
            raise ValueError(f"不支持的模型类型: {base_model}")
            
        # 获取隐藏层大小
        self.hidden_size = self.model_a.config.hidden_size
    
    def setup_comm_modules(self):
        """初始化通信模块"""
        module_type = self.config.get("module_type", "adapter")
        module_position = self.config.get("module_position", 8)
        
        self.logger.info(f"在第{module_position}层插入{module_type}通信模块")
        
        # 根据类型创建通信模块
        if module_type == "adapter":
            adapter_size = self.config.get("adapter_size", 64)
            
            module_a = AdapterModule(
                hidden_size=self.hidden_size,
                adapter_size=adapter_size,
                adapter_dropout=self.config.get("adapter_dropout", 0.1)
            )
            
            module_b = AdapterModule(
                hidden_size=self.hidden_size,
                adapter_size=adapter_size,
                adapter_dropout=self.config.get("adapter_dropout", 0.1)
            )
            
        elif module_type == "vector":
            comm_size = self.config.get("comm_size", 32)
            
            module_a = VectorCommModule(
                hidden_size=self.hidden_size,
                comm_size=comm_size,
                use_bottleneck=self.config.get("use_bottleneck", True),
                add_noise=self.config.get("add_noise", False)
            )
            
            module_b = VectorCommModule(
                hidden_size=self.hidden_size,
                comm_size=comm_size,
                use_bottleneck=self.config.get("use_bottleneck", True),
                add_noise=self.config.get("add_noise", False)
            )
            
        elif module_type == "symbol":
            vocab_size = self.config.get("vocab_size", 32)
            
            module_a = SymbolCommModule(
                hidden_size=self.hidden_size,
                vocab_size=vocab_size,
                seq_len=self.config.get("seq_len", 8)
            )
            
            module_b = SymbolCommModule(
                hidden_size=self.hidden_size,
                vocab_size=vocab_size,
                seq_len=self.config.get("seq_len", 8)
            )
            
        else:
            raise ValueError(f"不支持的通信模块类型: {module_type}")
        
        # 将模块插入模型
        self.model_a.insert_module(module_position, module_a)
        self.model_b.insert_module(module_position, module_b)
        
        # 保存模块引用便于访问
        self.module_a = module_a
        self.module_b = module_b
        self.module_position = module_position
        
        # 冻结基础模型参数
        self.model_a.freeze_base_model()
        self.model_b.freeze_base_model()
        
        # 初始模块参数快照（用于跟踪变化）
        self.initial_params_a = self._get_module_params(self.module_a)
        self.initial_params_b = self._get_module_params(self.module_b)
    
    def _get_module_params(self, module):
        """获取模块参数快照"""
        return {name: param.clone().detach().cpu() for name, param in module.named_parameters()}
    
    def setup_optimizers(self):
        """初始化优化器"""
        lr = self.config.get("learning_rate", 5e-5)
        
        # 只优化通信模块参数
        self.optimizer_a = torch.optim.AdamW(self.module_a.parameters(), lr=lr)
        self.optimizer_b = torch.optim.AdamW(self.module_b.parameters(), lr=lr)
        
        # 学习率调度器（可选）
        if self.config.get("use_lr_scheduler", False):
            self.scheduler_a = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_a, 
                T_max=self.config.get("num_epochs", 10)
            )
            self.scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_b, 
                T_max=self.config.get("num_epochs", 10)
            )
        else:
            self.scheduler_a = None
            self.scheduler_b = None
    
    def setup_task(self):
        """初始化任务"""
        task_name = self.config.get("task", "dialogue_completion")
        
        # 动态导入任务类
        if task_name == "dialogue_completion":
            from .dialogue_task import DialogueCompletionTask
            self.task = DialogueCompletionTask(self.config)
        elif task_name == "math_solving":
            from .math_solving_task import MathSolvingTask
            self.task = MathSolvingTask(self.config)
        else:
            raise ValueError(f"不支持的任务类型: {task_name}")
            
        self.logger.info(f"任务初始化完成: {task_name}")
    
    def train(self, epochs: int = 10) -> Dict[str, Any]:
        """
        训练通信模块
        
        参数:
            epochs: 训练轮数
            
        返回:
            训练历史记录
        """
        self.logger.info(f"开始训练 {epochs} 轮...")
        
        # 获取数据集
        train_dataloader = self.task.get_train_dataloader()
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            self.logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # 训练一个epoch
            train_loss = self._train_epoch(train_dataloader)
            
            # 评估
            eval_metrics = self.evaluate()
            
            # 记录参数变化
            param_changes_a = self._calculate_param_changes(
                self.initial_params_a, 
                self._get_module_params(self.module_a)
            )
            param_changes_b = self._calculate_param_changes(
                self.initial_params_b, 
                self._get_module_params(self.module_b)
            )
            
            # 更新历史记录
            self.history["train_loss"].append(train_loss)
            self.history["eval_metrics"].append(eval_metrics)
            self.history["param_changes"].append({
                "model_a": param_changes_a,
                "model_b": param_changes_b
            })
            
            # 生成可视化
            if epoch % self.config.get("vis_interval", 1) == 0:
                self._generate_visualizations(epoch)
                
            # 保存检查点
            if epoch % self.config.get("save_interval", 5) == 0 or epoch == epochs - 1:
                self._save_checkpoint(epoch)
                
            # 更新学习率
            if self.scheduler_a is not None:
                self.scheduler_a.step()
                self.scheduler_b.step()
                
            # 日志记录
            self.logger.info(f"Epoch {epoch+1} - 训练损失: {train_loss:.4f}, 评估指标: {eval_metrics}")
            
        self.logger.info("训练完成")
        return self.history
    
    def _train_epoch(self, dataloader) -> float:
        """训练一个epoch"""
        self.model_a.train()
        self.model_b.train()
        
        total_loss = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {self.current_epoch+1}"):
            loss = self._train_step(batch)
            total_loss += loss
            self.global_step += 1
            
        return total_loss / len(dataloader)
    
    def _train_step(self, batch) -> float:
        """单个训练步骤"""
        # 清空梯度
        self.optimizer_a.zero_grad()
        self.optimizer_b.zero_grad()
        
        # 进行模型A的前向传播
        inputs_a = {k: v.to(self.device) for k, v in batch["model_a"].items()}
        outputs_a = self.model_a(**inputs_a)
        
        # 获取模型A的通信向量（根据模块类型）
        if hasattr(self.module_a, "encode"):
            comm_vector_a = self.module_a.encode(
                self.model_a.get_layer_output(self.module_position, **inputs_a)
            )
        else:
            # Adapter模块没有显式的encode函数
            comm_vector_a = None
        
        # 进行模型B的前向传播，考虑模型A的通信
        inputs_b = {k: v.to(self.device) for k, v in batch["model_b"].items()}
        outputs_b = self.model_b(**inputs_b)
        
        # 计算任务损失
        loss, loss_details = self.task.compute_loss(
            batch, 
            outputs_a,
            outputs_b,
            comm_vector_a
        )
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.module_a.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.module_b.parameters(), 1.0)
        
        # 参数更新
        self.optimizer_a.step()
        self.optimizer_b.step()
        
        return loss.item()
    
    def evaluate(self) -> Dict[str, float]:
        """评估当前模型性能"""
        self.model_a.eval()
        self.model_b.eval()
        
        eval_dataloader = self.task.get_eval_dataloader()
        
        total_metrics = {}
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                # 前向传播
                inputs_a = {k: v.to(self.device) for k, v in batch["model_a"].items()}
                outputs_a = self.model_a(**inputs_a)
                
                # 获取模型A的通信向量（根据模块类型）
                if hasattr(self.module_a, "encode"):
                    comm_vector_a = self.module_a.encode(
                        self.model_a.get_layer_output(self.module_position, **inputs_a)
                    )
                else:
                    comm_vector_a = None
                
                inputs_b = {k: v.to(self.device) for k, v in batch["model_b"].items()}
                outputs_b = self.model_b(**inputs_b)
                
                # 计算指标
                batch_metrics = self.task.compute_metrics(
                    batch, 
                    outputs_a,
                    outputs_b,
                    comm_vector_a
                )
                
                # 累加指标
                for k, v in batch_metrics.items():
                    if k not in total_metrics:
                        total_metrics[k] = 0
                    total_metrics[k] += v
        
        # 计算平均值
        for k in total_metrics:
            total_metrics[k] /= len(eval_dataloader)
            
        return total_metrics
    
    def _calculate_param_changes(self, initial_params, current_params) -> Dict[str, float]:
        """计算参数变化"""
        changes = {}
        
        for name in initial_params:
            # 计算欧几里得距离
            changes[name] = torch.norm(
                current_params[name] - initial_params[name]
            ).item()
            
        # 计算总体变化
        changes["total"] = sum(changes.values())
        
        return changes
    
    def _generate_visualizations(self, epoch: int):
        """生成可视化结果"""
        module_type = self.config.get("module_type", "adapter")
        
        # 创建当前epoch的可视化目录
        vis_epoch_dir = os.path.join(self.vis_dir, f"epoch_{epoch}")
        os.makedirs(vis_epoch_dir, exist_ok=True)
        
        # 生成参数变化图
        plot_parameter_changes(
            self.history["param_changes"],
            os.path.join(vis_epoch_dir, "param_changes.png")
        )
        
        # 根据模块类型生成不同的可视化
        if module_type == "vector":
            # 获取最近的通信向量
            vectors_a = self.module_a.get_recent_vectors(100)
            vectors_b = self.module_b.get_recent_vectors(100)
            
            # 生成向量可视化
            visualize_comm_vectors(
                vectors_a, 
                os.path.join(vis_epoch_dir, "vectors_a.png"),
                title=f"模型A通信向量 (Epoch {epoch})"
            )
            visualize_comm_vectors(
                vectors_b, 
                os.path.join(vis_epoch_dir, "vectors_b.png"),
                title=f"模型B通信向量 (Epoch {epoch})"
            )
            
        elif module_type == "symbol":
            # 获取符号使用统计
            stats_a = self.module_a.get_symbol_stats()
            stats_b = self.module_b.get_symbol_stats()
            
            # 生成符号使用可视化
            plot_symbol_usage(
                stats_a["counts"].cpu().numpy(), 
                os.path.join(vis_epoch_dir, "symbols_a.png"),
                title=f"模型A符号使用 (Epoch {epoch}), 熵={stats_a['entropy']:.2f}"
            )
            plot_symbol_usage(
                stats_b["counts"].cpu().numpy(), 
                os.path.join(vis_epoch_dir, "symbols_b.png"),
                title=f"模型B符号使用 (Epoch {epoch}), 熵={stats_b['entropy']:.2f}"
            )
    
    def _save_checkpoint(self, epoch: int):
        """保存模型检查点"""
        checkpoint = {
            "epoch": epoch,
            "config": self.config,
            "module_a_state": self.module_a.state_dict(),
            "module_b_state": self.module_b.state_dict(),
            "optimizer_a_state": self.optimizer_a.state_dict(),
            "optimizer_b_state": self.optimizer_b.state_dict(),
            "history": self.history
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"保存检查点: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        self.logger.info(f"加载检查点: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载模块参数
        self.module_a.load_state_dict(checkpoint["module_a_state"])
        self.module_b.load_state_dict(checkpoint["module_b_state"])
        
        # 加载优化器状态
        self.optimizer_a.load_state_dict(checkpoint["optimizer_a_state"])
        self.optimizer_b.load_state_dict(checkpoint["optimizer_b_state"])
        
        # 恢复训练状态
        self.current_epoch = checkpoint["epoch"] + 1
        self.history = checkpoint["history"]
        
        self.logger.info(f"加载完成，从epoch {self.current_epoch} 继续训练")
    
    def visualize_evolution(self, history: Optional[Dict[str, Any]] = None):
        """可视化进化过程"""
        if history is None:
            history = self.history
            
        # 创建可视化目录
        summary_dir = os.path.join(self.vis_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        
        # 绘制训练损失曲线
        epochs = range(1, len(history["train_loss"]) + 1)
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history["train_loss"], marker='o')
        plt.title('训练损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(summary_dir, "training_loss.png"))
        plt.close()
        
        # 绘制评估指标
        for metric_name in history["eval_metrics"][0].keys():
            metric_values = [metrics[metric_name] for metrics in history["eval_metrics"]]
            
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, metric_values, marker='o')
            plt.title(f'评估指标: {metric_name}')
            plt.xlabel('Epoch')
            plt.ylabel(metric_name)
            plt.grid(True)
            plt.savefig(os.path.join(summary_dir, f"metric_{metric_name}.png"))
            plt.close()
            
        # 绘制参数变化
        total_changes_a = [changes["model_a"]["total"] for changes in history["param_changes"]]
        total_changes_b = [changes["model_b"]["total"] for changes in history["param_changes"]]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, total_changes_a, marker='o', label='模型A')
        plt.plot(epochs, total_changes_b, marker='x', label='模型B')
        plt.title('通信模块参数变化')
        plt.xlabel('Epoch')
        plt.ylabel('参数距离')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(summary_dir, "parameter_changes.png"))
        plt.close()
        
        self.logger.info(f"进化过程可视化已保存至 {summary_dir}") 