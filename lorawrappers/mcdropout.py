import torch
from typing import Any
from .lorawrapper import LoraWrapper

class MCDropoutLoraWrapper(LoraWrapper):
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        previous_dtype = x.dtype
        result = self.base_layer(x, *args, **kwargs)
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            x = x.to(self.lora_B[active_adapter].weight.dtype)
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            x = dropout.train()(x)  # always apply dropout even in eval mode
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            result += lora_B(lora_A(x)) * scaling
        result = result.to(previous_dtype)
        return result
