import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from peft.tuners.lora import LoraLayer
from .blob import BlobLoraWrapper, BlobLinear
from .viwrapper import VILoraWrapper

class TFBLoraWrapper(BlobLoraWrapper):
    def __init__(self, lora_layer: LoraLayer, eps: float = 0.05, beta: float = 0.2, *args: Any, **kwargs: Any):
        VILoraWrapper.__init__(self, lora_layer, *args, **kwargs)
        self.s_vals = {}
        for adapter_name in self.active_adapters:
            self.lora_A[adapter_name].requires_grad_(False)
            self.lora_B[adapter_name].requires_grad_(False)

            A_weight = self.lora_A[adapter_name].weight.clone()
            B_weight = self.lora_B[adapter_name].weight.clone()

            U, s, V = torch.svd(B_weight)
            # U, s, Vh = torch.linalg.svd(B_weight, full_matrices=False)
            self.s_vals[adapter_name] = s
            
            # self.lora_B[adapter_name].weight.data = U @ torch.diag(s)
            # old_weight = self.lora_A[adapter_name].weight.data.clone()
            
            self.lora_A[adapter_name] = BlobLinear(
                in_features=self.lora_A[adapter_name].in_features,
                out_features=self.lora_A[adapter_name].out_features,
                eps=eps,
                beta=beta,
                blobsample=True,
            ).to(V.device)

            self.lora_A[adapter_name].weight.data = V.T @ A_weight
            self.lora_B[adapter_name].weight.data = U @ torch.diag(s)

        self.set_cov(beta=beta)


    def set_cov(self, beta=0.0):
        for adapter_name in self.active_adapters:
            s = self.s_vals[adapter_name]
            lora_std = beta / (torch.tile(s.reshape(-1, 1), dims=(1, self.lora_A[adapter_name].in_features)) + 1e-6)
            self.lora_A[adapter_name].lora_A_rho.data = torch.sqrt(lora_std)
