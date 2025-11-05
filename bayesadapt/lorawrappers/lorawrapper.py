import torch
from typing import Any
from peft.tuners.lora import LoraLayer

class LoraWrapper(torch.nn.Module):
    def __init__(self, lora_layer: LoraLayer, *args: Any, **kwargs: Any):
        super().__init__()
        self.base_layer = lora_layer.base_layer
        self.lora_A = lora_layer.lora_A
        self.lora_B = lora_layer.lora_B
        self.lora_dropout = lora_layer.lora_dropout
        self.scaling = lora_layer.scaling
        self.active_adapters = lora_layer.active_adapters
    
    #standard lora forward pass
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        previous_dtype = x.dtype
        result = self.base_layer(x, *args, **kwargs)
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            x = x.to(self.lora_B[active_adapter].weight.dtype)
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            x = dropout(x)
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            result = result + lora_B(lora_A(x)) * scaling
        result = result.to(previous_dtype)
        return result
