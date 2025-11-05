import torch
from typing import Any
from .lorawrapper import LoraWrapper

class VILoraWrapper(LoraWrapper):
    def sample(self, status=True):
        if self.training is True and status is False:
            raise ValueError("blobsample should be set to True only during training.")
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            self.lora_A[active_adapter].blobsample = status

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
            #result += lora_A(x, lora_B, scaling)
            result = result + lora_A(x, lora_B, scaling)
        result = result.to(previous_dtype)
        return result

    @property
    def kl_div(self) -> torch.Tensor:
        kl_vals = []
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            kl_vals.append(self.lora_A[active_adapter].kl_div)
        kl = torch.stack(kl_vals).sum()
        return kl
