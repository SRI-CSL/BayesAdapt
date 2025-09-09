import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Any, List, Optional, Union
from peft.utils.integrations import dequantize_bnb_weight
import bitsandbytes as bnb
from full_cov import FullRankCov
import numpy as np 
from peft.tuners.lora import LoraLayer, Linear

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
                 bayes_eps: float = 0.2, bayes_beta: float = 0.2, blobsample: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(in_features, out_features)
        self.weight = nn.Linear(in_features, out_features, bias=False).weight
        self.lora_A_rho = nn.Parameter(torch.zeros(self.rank, self.in_features))
        if bayes_eps < 0:
            nn.init.uniform_(
                self.lora_A_rho,
                bayes_eps - 1,
                bayes_eps,
            )
        else:
            nn.init.uniform_(
                self.lora_A_rho,
                bayes_eps / math.sqrt(2),
                bayes_eps,
            )
        self.bayes_beta = bayes_beta
        self.blobsample = blobsample
    
    def forward(self, x, lora_B, scaling):
        #W = self.U @ torch.diag(self.s) @ self.Vh
        # W = torch.diag(self.s) @ self.Vh
        # W = self.weight
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
                        (x.size(0), self.rank),
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
                        (x.size(0), x.size(1), self.rank),
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
            self.weight, sigma_weight, 0, self.bayes_beta
        )
        return kl

class BlobLoraWrapper(nn.Module):
    def __init__(self, lora_layer: LoraLayer, bayes_eps: float = 0.05, bayes_beta: float = 0.2):
        super().__init__()
        self.base_layer = lora_layer.base_layer
        self.lora_A = lora_layer.lora_A
        self.lora_B = lora_layer.lora_B
        self.lora_dropout = lora_layer.lora_dropout
        self.scaling = lora_layer.scaling
        self.active_adapters = lora_layer.active_adapters
        for adapter_name in self.active_adapters:
            self.lora_A[adapter_name] = BlobLinear(
                in_features=lora_layer.lora_A[adapter_name].in_features,
                out_features=lora_layer.lora_A[adapter_name].out_features,
                bayes_eps=bayes_eps,
                bayes_beta=bayes_beta,
                blobsample=True,
            )

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
    
    @property
    def kl_div(self) -> torch.Tensor:
        kl_vals = []
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            kl_vals.append(self.lora_A[active_adapter].kl_div)
        # if len(kl_vals) == 1:
            # return kl_vals[0]
        kl = torch.stack(kl_vals).sum()
        return kl

    def sample(self, status=True):
        if self.training is True and status is False:
            raise ValueError("blobsample should be set to True only during training.")
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            self.lora_A[active_adapter].blobsample = status
        # self.lora_A['default'].blobsample = status

