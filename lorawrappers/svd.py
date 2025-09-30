import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from peft.tuners.lora import LoraLayer

class SVDLoraWrapper(torch.nn.Module):
    def __init__(self, lora_layer: LoraLayer, s_sigma_init_eps: float = 0.2, *args: Any, **kwargs: Any):
        super().__init__()
        W = lora_layer.base_layer.weight.detach()
        previous_dtype = W.dtype
        self.U, self.s, self.Vh = torch.linalg.svd(W.float(), full_matrices=False)
        self.log_s = self.s.log().to(previous_dtype)
        self.U, self.s, self.Vh = self.U.to(previous_dtype), self.s.to(previous_dtype), self.Vh.to(previous_dtype)
        
        z = torch.zeros_like(self.s)
        self.log_s_sigma = nn.Parameter(z)
        nn.init.uniform_(
            self.log_s_sigma,
            s_sigma_init_eps / math.sqrt(2),
            s_sigma_init_eps,
        )
        self.log_s_sigma.data = self.log_s_sigma.data.log()

        self.s_scale = nn.Parameter(torch.ones_like(self.s))
        self.blobsample = True

        # self.base_layer = lora_layer.base_layer
        # self.lora_A = lora_layer.lora_A
        # self.lora_B = lora_layer.lora_B
        # self.lora_dropout = lora_layer.lora_dropout
        # self.scaling = lora_layer.scaling
        # self.active_adapters = lora_layer.active_adapters

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        # previous_dtype = x.dtype
        # x = x.to(self.U.dtype)
        s = (self.log_s * self.s_scale).exp()
        W = self.U @ torch.diag(s) @ self.Vh
        result = F.linear(x, W)
        if self.blobsample:
            s_sigma = self.log_s_sigma.exp()
            s_noise = s_sigma * torch.randn_like(self.s)
            S_noise = torch.diag(s_noise)
            W_noise = self.U @ S_noise @ self.Vh
            result += F.linear(x, W_noise)
        # result = result.to(previous_dtype)
        return result
