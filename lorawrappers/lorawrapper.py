import torch
from typing import Any
from peft.tuners.lora import LoraLayer, Linear

class LoraWrapper(torch.nn.Module):
    def __init__(self, lora_layer: LoraLayer, *args: Any, **kwargs: Any):
        super().__init__()
        self.base_layer = lora_layer.base_layer
        self.lora_A = lora_layer.lora_A
        self.lora_B = lora_layer.lora_B
        self.lora_dropout = lora_layer.lora_dropout
        self.scaling = lora_layer.scaling
        self.active_adapters = lora_layer.active_adapters
        # self.wrap(*args, **kwargs)

    # def wrap(self, *args: Any, **kwargs: Any) -> None:
        # raise NotImplementedError("This method should be implemented in subclasses.")

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
            result += lora_A(x, lora_B, scaling)
        result = result.to(previous_dtype)
        return result

class VILoraWrapper(LoraWrapper):
    def sample(self, status=True):
        if self.training is True and status is False:
            raise ValueError("blobsample should be set to True only during training.")
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            self.lora_A[active_adapter].blobsample = status

    @property
    def kl_div(self) -> torch.Tensor:
        kl_vals = []
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            kl_vals.append(self.lora_A[active_adapter].kl_div)
        kl = torch.stack(kl_vals).sum()
        return kl
