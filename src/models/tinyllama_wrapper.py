import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model_wrapper import ModelWrapper

class TinyLlamaWrapper(ModelWrapper):
    """
    TinyLlama模型包装器，提供对模型内部结构的访问和修改功能
    """
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", 
                 device: Optional[str] = None):
        super().__init__(model_name, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _initialize_model(self):
        """加载TinyLlama预训练模型"""
        print(f"加载模型: {self.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.config = self.model.config
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                **kwargs) -> Dict[str, torch.Tensor]:
        """模型前向传播"""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        outputs = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            **kwargs
        )
        
        return {
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states if hasattr(outputs, "hidden_states") else None
        }
    
    def insert_module(self, layer_idx: int, module: nn.Module) -> bool:
        """在指定Transformer层插入通信模块"""
        if not hasattr(self.model, "transformer"):
            print("错误: 无法访问transformer模块")
            return False
            
        if not hasattr(self.model.transformer, "h"):
            print("错误: 无法访问transformer层")
            return False
            
        num_layers = len(self.model.transformer.h)
        if layer_idx < 0 or layer_idx >= num_layers:
            print(f"错误: layer_idx {layer_idx} 超出范围 [0, {num_layers-1}]")
            return False
            
        # 保存原始模块的引用
        original_module = self.model.transformer.h[layer_idx]
        
        # 替换为包含通信模块的新层
        class WrappedLayer(nn.Module):
            def __init__(self, original_layer, comm_module):
                super().__init__()
                self.original_layer = original_layer
                self.comm_module = comm_module
                
            def forward(self, *args, **kwargs):
                outputs = self.original_layer(*args, **kwargs)
                # TinyLlama输出一般是元组(hidden_states, ...)
                if isinstance(outputs, tuple):
                    # 将通信模块应用于隐藏状态
                    modified_hidden = self.comm_module(outputs[0])
                    return (modified_hidden,) + outputs[1:]
                return self.comm_module(outputs)
        
        # 创建并应用包装层
        wrapped_layer = WrappedLayer(original_module, module)
        self.model.transformer.h[layer_idx] = wrapped_layer
        
        # 保存对已修改层的引用
        self.modified_layers[layer_idx] = module
        
        return True
    
    def get_layer_output(self, layer_idx: int, input_ids: torch.Tensor, 
                         attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """获取指定层的输出"""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        # 设置输出隐藏状态
        self.model.config.output_hidden_states = True
        
        # 前向传播
        outputs = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            output_hidden_states=True
        )
        
        # TinyLlama的隐藏状态顺序: (输入嵌入, 层1, 层2, ..., 最终输出)
        # 要获取指定层的输出，需要访问hidden_states[layer_idx+1]
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            if layer_idx + 1 < len(outputs.hidden_states):
                return outputs.hidden_states[layer_idx + 1]
            else:
                print(f"错误: layer_idx {layer_idx} 超出范围")
                return None
        else:
            print("错误: 无法访问隐藏状态")
            return None
    
    def freeze_base_model(self):
        """冻结基础模型参数"""
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 重新启用通信模块的参数
        for module in self.modified_layers.values():
            for param in module.parameters():
                param.requires_grad = True
    
    def unfreeze_base_model(self):
        """解冻基础模型参数"""
        for param in self.model.parameters():
            param.requires_grad = True 