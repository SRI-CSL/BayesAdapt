#adapted from: https://github.com/Wang-ML-Lab/bayesian-peft/blob/main/modelwrappers/blob.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from peft.tuners.lora import LoraLayer, Linear
from .viwrapper import VILoraWrapper

def kl_div_stable(mu_q, sigma_q, mu_p, sigma_p):
    eps = 1e-6
    kl = (
        math.log(sigma_p + eps)
        - torch.log(sigma_q.to(torch.float64) + eps)
        + (sigma_q.to(torch.float64) ** 2 + (mu_q.to(torch.float64) - mu_p) ** 2)
        / (2 * (sigma_p**2) + eps)
        - 0.5
    )
    return kl.sum()

class BlobLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 eps: float = 0.05, beta: float = 0.2, blobsample: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(in_features, out_features) #problem if num_classes < rank?
        self.weight = nn.Linear(in_features, out_features, bias=False).weight
        self.lora_A_rho = nn.Parameter(torch.zeros(self.rank, self.in_features))
        if eps < 0:
            nn.init.uniform_(
                self.lora_A_rho,
                eps - 1,
                eps,
            )
        else:
            nn.init.uniform_(
                self.lora_A_rho,
                eps / math.sqrt(2),
                eps,
            )
        self.beta = beta
        self.blobsample = blobsample
    
    def forward(self, x, lora_B, scaling):
        result = lora_B(F.linear(x, self.weight)) * scaling

        if self.blobsample: 
            A_sigma = self.lora_A_rho ** 2
            if x.dim() == 2:
                r_A = (
                    torch.ones(
                        (x.size(0), self.in_features), device=x.device, dtype=x.dtype
                    )
                    .uniform_(-1, 1)
                    .sign()
                )
                s_A = (
                    torch.ones(
                        (x.size(0), self.lora_A_rho.size(0)),
                        device=x.device,
                        dtype=x.dtype,
                    )
                    .uniform_(-1, 1)
                    .sign()
                )
            else:
                r_A = (
                    torch.ones(
                        (x.size(0), x.size(1), self.in_features),
                        device=x.device,
                        dtype=x.dtype,
                    )
                    .uniform_(-1, 1)
                    .sign()
                )
                s_A = (
                    torch.ones(
                        (x.size(0), x.size(1), self.lora_A_rho.size(0)),
                        device=x.device,
                        dtype=x.dtype,
                    )
                    .uniform_(-1, 1)
                    .sign()
                )

            # x = dropout(x)
            lora_noise_a = A_sigma * torch.randn_like(self.weight)

            noise = (((x * r_A) @ lora_noise_a.transpose(0, 1)) * s_A) @ lora_B.weight.transpose(0, 1)

            result += noise * scaling

        return result
    
    @property
    def kl_div(self) -> torch.Tensor:
        sigma_weight = self.lora_A_rho ** 2
        kl = kl_div_stable(
            self.weight, sigma_weight, 0, self.beta
        )
        return kl

class BlobLoraWrapper(VILoraWrapper):
    def __init__(self, lora_layer: LoraLayer, eps: float = 0.05, beta: float = 0.2, *args: Any, **kwargs: Any):
        super().__init__(lora_layer, *args, **kwargs)
        for adapter_name in self.active_adapters:
            self.lora_A[adapter_name] = BlobLinear(
                in_features=self.lora_A[adapter_name].in_features,
                out_features=self.lora_A[adapter_name].out_features,
                eps=eps,
                beta=beta,
                blobsample=True,
            )
