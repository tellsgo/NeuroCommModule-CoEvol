import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple

class ModelWrapper(ABC, nn.Module):
    """
    大型语言模型的抽象基类，为模块化通信实验提供统一接口
    """
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        super().__init__()
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_model()
        self.modified_layers = {}  # 跟踪被修改的层
        
    @abstractmethod
    def _initialize_model(self):
        """加载预训练模型"""
        pass
    
    @abstractmethod
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                **kwargs) -> Dict[str, torch.Tensor]:
        """模型前向传播"""
        pass
    
    @abstractmethod
    def insert_module(self, layer_idx: int, module: nn.Module) -> bool:
        """在指定层插入通信模块"""
        pass
    
    @abstractmethod
    def get_layer_output(self, layer_idx: int, input_ids: torch.Tensor, 
                         attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """获取指定层的输出"""
        pass
    
    @abstractmethod
    def freeze_base_model(self):
        """冻结基础模型参数"""
        pass
    
    @abstractmethod
    def unfreeze_base_model(self):
        """解冻基础模型参数"""
        pass
    
    def get_modified_layers(self) -> Dict[int, nn.Module]:
        """返回所有被修改的层"""
        return self.modified_layers
    
    def save_modules(self, path: str):
        """保存通信模块参数"""
        modules_state = {
            f"layer_{layer_idx}": module.state_dict() 
            for layer_idx, module in self.modified_layers.items()
        }
        torch.save(modules_state, path)
        
    def load_modules(self, path: str):
        """加载通信模块参数"""
        modules_state = torch.load(path, map_location=self.device)
        for layer_name, state in modules_state.items():
            layer_idx = int(layer_name.split("_")[1])
            if layer_idx in self.modified_layers:
                self.modified_layers[layer_idx].load_state_dict(state)
            else:
                print(f"警告: 模块 {layer_name} 在模型中不存在") 