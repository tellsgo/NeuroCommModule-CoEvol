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

# 添加分布式训练支持
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


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
                - distributed: 是否启用分布式训练
                - devices: 使用的GPU设备列表
                - 其他模块特定参数
        """
        self.config = config
        self.setup_logging()
        self.setup_paths()
        
        # 保存配置
        with open(os.path.join(self.exp_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        # 分布式训练设置
        self.distributed = config.get("distributed", False)
        self.devices = config.get("devices", [0])
        self.world_size = len(self.devices) if self.distributed else 1
        self.rank = 0  # 默认rank，分布式训练时会被覆盖
        
        # 如果启用分布式训练，初始化进程组
        if self.distributed:
            self.logger.info(f"启用分布式训练，使用GPU: {self.devices}")
            # 如果已经初始化，避免重复初始化
            if not dist.is_initialized():
                # 使用环境变量初始化
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                # 如果是主进程，直接初始化模型
                if 'RANK' not in os.environ and 'LOCAL_RANK' not in os.environ:
                    self.setup_distributed()
                else:
                    # 如果环境变量已存在（可能通过torch.distributed.launch启动），使用这些
                    self.rank = int(os.environ.get('RANK', 0))
                    self.logger.info(f"使用现有分布式环境，当前进程rank: {self.rank}")
                    self._initialize_components()
            else:
                self.rank = dist.get_rank()
                self.logger.info(f"分布式环境已初始化，当前进程rank: {self.rank}")
                self._initialize_components()
        else:
            # 非分布式训练，直接初始化组件
            self._initialize_components()
    
    def setup_distributed(self):
        """设置分布式训练环境"""
        if self.distributed:
            mp.spawn(self._distributed_worker, args=(self.world_size,), nprocs=self.world_size, join=True)
        else:
            self._initialize_components()
    
    def _distributed_worker(self, rank: int, world_size: int):
        """分布式训练工作进程"""
        self.rank = rank
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(rank)
        
        self.logger.info(f"初始化分布式进程组：rank {rank}/{world_size}")
        # 初始化进程组
        dist.init_process_group(
            backend='nccl',  # 使用NCCL后端，适合NVIDIA GPU
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        # 设置当前设备
        torch.cuda.set_device(self.devices[rank])
        self.logger.info(f"进程 {rank} 使用 GPU: {self.devices[rank]}")
        
        # 初始化组件
        self._initialize_components()
        
        # 清理进程组
        dist.destroy_process_group()
    
    def _initialize_components(self):
        """初始化所有组件"""
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
        if self.distributed:
            self.device = f"cuda:{self.devices[self.rank]}"
            self.logger.info(f"进程 {self.rank} 使用GPU: {self.device}")
        elif torch.cuda.is_available():
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
            
        # 如果是分布式训练，封装模型为DDP模型
        if self.distributed:
            # 确保模型在正确的设备上
            self.model_a.model.to(self.device)
            self.model_b.model.to(self.device)
            
            # 使用混合精度训练
            self.scaler = torch.cuda.amp.GradScaler() if self.config.get("use_amp", True) else None
            
            # 注意：我们不直接包装ModelWrapper实例，而是包装内部的实际模型
            self.model_a.model = DDP(
                self.model_a.model,
                device_ids=[self.devices[self.rank]],
                output_device=self.devices[self.rank],
                find_unused_parameters=True  # 由于我们只更新部分参数，需要启用这个选项
            )
            self.model_b.model = DDP(
                self.model_b.model,
                device_ids=[self.devices[self.rank]],
                output_device=self.devices[self.rank],
                find_unused_parameters=True
            )
            
            self.logger.info(f"模型已封装为DDP模型，设备ID: {self.devices[self.rank]}")
            
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
        
        # 确保模块在正确的设备上
        module_a.to(self.device)
        module_b.to(self.device)
        
        # 将模块插入模型
        if hasattr(self.model_a, 'module'):
            # 对于DDP模型，需要访问.module属性
            self.model_a.insert_module(module_position, module_a)
            self.model_b.insert_module(module_position, module_b)
        else:
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
        
        # 如果是分布式训练，使用DistributedSampler
        if self.distributed:
            from torch.utils.data.distributed import DistributedSampler
            # 重新创建分布式版本的数据加载器
            train_sampler = DistributedSampler(
                train_dataloader.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
            train_dataloader = torch.utils.data.DataLoader(
                train_dataloader.dataset,
                batch_size=train_dataloader.batch_size,
                sampler=train_sampler,
                num_workers=4,
                pin_memory=True
            )
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            self.logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # 如果是分布式训练，设置epoch用于shuffle
            if self.distributed:
                train_dataloader.sampler.set_epoch(epoch)
            
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
            
            # 仅在主进程生成可视化和保存检查点
            if not self.distributed or self.rank == 0:
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
            
            # 如果是分布式训练，同步进程
            if self.distributed:
                dist.barrier()
                
        self.logger.info("训练完成")
        return self.history
    
    def _train_epoch(self, dataloader) -> float:
        """训练一个epoch"""
        self.model_a.train()
        self.model_b.train()
        
        total_loss = 0
        num_batches = 0  # 用于分布式环境下的正确平均
        
        for batch in tqdm(dataloader, desc=f"Epoch {self.current_epoch+1}", disable=self.distributed and self.rank != 0):
            loss = self._train_step(batch)
            total_loss += loss
            self.global_step += 1
            num_batches += 1
            
        # 如果是分布式训练，收集所有进程的损失并计算平均值
        if self.distributed:
            # 收集所有进程的总损失
            total_loss_tensor = torch.tensor([total_loss], device=self.device)
            batches_tensor = torch.tensor([num_batches], device=self.device)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(batches_tensor, op=dist.ReduceOp.SUM)
            total_loss = total_loss_tensor.item() / batches_tensor.item()
        else:
            total_loss = total_loss / num_batches
            
        return total_loss
    
    def _train_step(self, batch) -> float:
        """单个训练步骤"""
        # 清空梯度
        self.optimizer_a.zero_grad()
        self.optimizer_b.zero_grad()
        
        # 是否使用混合精度训练
        use_amp = hasattr(self, 'scaler') and self.scaler is not None
        
        with torch.cuda.amp.autocast() if use_amp else torch.no_grad():
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
        if use_amp:
            # 使用混合精度训练
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            self.scaler.unscale_(self.optimizer_a)
            self.scaler.unscale_(self.optimizer_b)
            torch.nn.utils.clip_grad_norm_(self.module_a.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.module_b.parameters(), 1.0)
            
            # 参数更新
            self.scaler.step(self.optimizer_a)
            self.scaler.step(self.optimizer_b)
            self.scaler.update()
        else:
            # 普通训练
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
        
        # 如果是分布式训练，使用DistributedSampler
        if self.distributed:
            from torch.utils.data.distributed import DistributedSampler
            # 重新创建分布式版本的数据加载器
            eval_sampler = DistributedSampler(
                eval_dataloader.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False
            )
            eval_dataloader = torch.utils.data.DataLoader(
                eval_dataloader.dataset,
                batch_size=eval_dataloader.batch_size,
                sampler=eval_sampler,
                num_workers=4,
                pin_memory=True
            )
        
        total_metrics = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating", disable=self.distributed and self.rank != 0):
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
                
                num_batches += 1
        
        # 如果是分布式训练，收集所有进程的指标并计算平均值
        if self.distributed:
            # 对每个指标进行规约
            for k in total_metrics.keys():
                metric_tensor = torch.tensor([total_metrics[k]], device=self.device)
                dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
                total_metrics[k] = metric_tensor.item()
            
            # 收集批次数量
            batches_tensor = torch.tensor([num_batches], device=self.device)
            dist.all_reduce(batches_tensor, op=dist.ReduceOp.SUM)
            num_batches = batches_tensor.item()
        
        # 计算平均值
        for k in total_metrics:
            total_metrics[k] /= num_batches
            
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
        
        # 如果使用AMP，保存scaler状态
        if hasattr(self, 'scaler') and self.scaler is not None:
            checkpoint["scaler"] = self.scaler.state_dict()
        
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
        
        # 如果使用AMP，加载scaler状态
        if "scaler" in checkpoint and hasattr(self, 'scaler') and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])
        
        # 恢复训练状态
        self.current_epoch = checkpoint["epoch"] + 1
        self.history = checkpoint["history"]
        
        self.logger.info(f"加载完成，从epoch {self.current_epoch} 继续训练")
    
    def visualize_evolution(self, history: Optional[Dict[str, Any]] = None):
        """可视化进化过程"""
        # 在分布式环境中，只有主进程执行可视化
        if self.distributed and self.rank != 0:
            return
            
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
        plt.title('通信模块参数变化 - 语言如何雕刻神经连接')
        plt.xlabel('训练轮次 (语言游戏的迭代过程)')
        plt.ylabel('参数距离 (认知结构的重组幅度)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(summary_dir, "parameter_changes.png"))
        plt.close()
        
        self.logger.info(f"进化过程可视化已保存至 {summary_dir}")
        # 添加维特根斯坦风格的哲学解读
        with open(os.path.join(summary_dir, "philosophical_interpretation.txt"), "w") as f:
            f.write("# 维特根斯坦视角下的神经通信进化解读\n\n")
            f.write("> \"语言的边界，就是神经网络的边界。\" — 借鉴维特根斯坦《逻辑哲学论》5.6\n\n")
            f.write("## 通信模块的演化轨迹与语言游戏\n\n")
            f.write("1. **初始阶段**：模块参数随机分布，对应维特根斯坦所谓的'语言前状态'—符号尚未获得意义\n")
            f.write("2. **中间阶段**：参数变化曲线陡峭，表明语言规则正在形成，对应《哲学研究》中语言游戏规则的建立\n")
            f.write("3. **后期阶段**：曲线趋于平缓，呈现'家族相似性'，暗示通信系统已经稳定\n\n")
            f.write("## 符号使用的统计学模式\n\n")
            f.write("熵的变化反映了'私人语言不可能性'的计算证明—只有共享的符号系统才能稳定演化\n\n")
            f.write("维特根斯坦曾说:'意义不在符号内部，而在其使用中'—神经网络通过任务驱动的符号使用，重塑了自身的认知结构\n")
            
        # 创建分形式视觉设计，象征语言递归性塑造神经连接
        try:
            from matplotlib import cm
            from matplotlib.colors import ListedColormap
            import numpy as np
            
            def mandelbrot_set(h, w, max_iter):
                y, x = np.ogrid[-1.4:1.4:h*1j, -2:0.8:w*1j]
                c = x + y*1j
                z = c
                divtime = max_iter + np.zeros(z.shape, dtype=int)
                
                for i in range(max_iter):
                    z = z**2 + c
                    diverge = z*np.conj(z) > 2**2
                    div_now = diverge & (divtime == max_iter)
                    divtime[div_now] = i
                    z[diverge] = 2
                    
                return divtime
                
            plt.figure(figsize=(12, 10))
            mandel = mandelbrot_set(1000, 1000, 100)
            viridis_cmap = cm.get_cmap('viridis', 100)
            plt.imshow(mandel, cmap=viridis_cmap, extent=[-2, 0.8, -1.4, 1.4])
            plt.title("语言的递归性塑造神经连接 - 维特根斯坦隐喻")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(summary_dir, "language_fractal.png"), dpi=300)
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"创建分形图像失败: {e}") 