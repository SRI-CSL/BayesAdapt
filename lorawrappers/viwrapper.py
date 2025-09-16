import torch
from .lorawrapper import LoraWrapper

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
