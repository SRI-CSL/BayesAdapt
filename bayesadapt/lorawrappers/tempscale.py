import copy
import torch
import torch.nn as nn
from typing import Any
from peft.tuners.lora import LoraLayer
from .lorawrapper import LoraWrapper

class TempScaleLoraWrapper(LoraWrapper):
    def __init__(self, lora_layer: LoraLayer, per_class: bool = True, *args: Any, **kwargs: Any):
        super().__init__(lora_layer, *args, **kwargs)
        #set grad off for everything except the temp
        for name, param in self.named_parameters():
            param.requires_grad = False

        if per_class:
            self.lora_temp_scale = nn.Parameter(torch.ones(lora_layer.out_features))
        else:
            self.lora_temp_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        output = super().forward(x, *args, **kwargs)
        output = output * self.lora_temp_scale
        return output
