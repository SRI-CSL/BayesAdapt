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

class ScalaBLLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 s_sigma_init_eps: float = 0.2, blobsample: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        rand_weights = nn.Linear(in_features, out_features, bias=False).weight.detach()
        _, s, Vh = torch.linalg.svd(rand_weights, full_matrices=False)
        self.rank = min(in_features, out_features)
        # self.U = nn.Parameter(U.contiguous())
        self.log_s = nn.Parameter(torch.log(s))
        self.Vh = nn.Parameter(Vh.contiguous())
        # self.register_buffer("Vh", Vh.contiguous())
        z = torch.zeros(self.rank)
        self.log_s_sigma = nn.Parameter(z)
        nn.init.uniform_(
            self.log_s_sigma,
            s_sigma_init_eps / math.sqrt(2),
            s_sigma_init_eps,
        )
        self.log_s_sigma.data = self.log_s_sigma.data.log()

        self.register_buffer("log_p_mu", torch.log(z + 1e-8))
        self.register_buffer("log_p_sigma", torch.log(z + 1))
        self.blobsample = blobsample
    
    @property
    def s(self) -> torch.Tensor:
        return self.log_s.exp()

    @property
    def s_sigma(self) -> torch.Tensor:
        return self.log_s_sigma.exp()

    def forward(self, x, lora_B, scaling):
        #W = self.U @ torch.diag(self.s) @ self.Vh
        W = torch.diag(self.s) @ self.Vh
        result = lora_B(F.linear(x, W)) * scaling

        if self.blobsample: 
            s_noise = self.s_sigma * torch.randn_like(self.log_s)
            S_noise = torch.diag(s_noise)
            #W_noise = self.U @ S_noise @ self.Vh
            W_noise = S_noise @ self.Vh
            result += lora_B(F.linear(x, W_noise)) * scaling
        return result
    
    @property
    def kl_div(self) -> torch.Tensor:
        max_log_mu = torch.max(self.log_p_mu, self.log_s)
        mu_p_minus_mu_q = torch.exp(max_log_mu) * (
            # torch.exp(log_mu_p - max_log_mu) - torch.exp(log_mu_q - max_log_mu)
            torch.exp(self.log_p_mu - max_log_mu) - torch.exp(self.log_s - max_log_mu)
        )
        mu_p_minus_mu_q_squared = mu_p_minus_mu_q ** 2

        # Compute sigma^2 terms in log-space
        # log_sigma_p_squared = 2 * log_sigma_p
        # log_sigma_q_squared = 2 * log_sigma_q

        # Exponentiate for sigma^2 terms
        sigma_p_squared = torch.exp(2 * self.log_p_sigma)
        sigma_s_squared = torch.exp(2 * self.log_s_sigma)

        # KL divergence formula with log-space components
        kl_div = (
            # (log_sigma_q - log_sigma_p)  # log(sigma_q / sigma_p)
            (self.log_s_sigma - self.log_p_sigma)
            + (sigma_p_squared + mu_p_minus_mu_q_squared) / (2 * sigma_s_squared)
            - 0.5
        )
        return kl_div.sum()


class SimpleLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 s_sigma_init_eps: float = 0.2, blobsample: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        rand_weights = nn.Linear(in_features, out_features, bias=False).weight.detach()
        U, s, Vh = torch.linalg.svd(rand_weights, full_matrices=False)
        self.rank = min(in_features, out_features)
        # self.U = nn.Parameter(U.contiguous())
        self.log_s = nn.Parameter(torch.log(s))
        # self.Vh = nn.Parameter(Vh.contiguous())
        out_dim = Vh.shape[0] * Vh.shape[1]
        self.P = nn.Linear(self.rank, out_dim, bias=False)
        z = torch.zeros(self.rank)
        self.log_s_sigma = nn.Parameter(z)
        nn.init.uniform_(
            self.log_s_sigma,
            s_sigma_init_eps / math.sqrt(2),
            s_sigma_init_eps,
        )
        self.log_s_sigma.data = self.log_s_sigma.data.log()

        self.register_buffer("log_p_mu", torch.log(z + 1e-8))
        self.register_buffer("log_p_sigma", torch.log(z + 1))
        self.blobsample = blobsample
    
    @property
    def s(self) -> torch.Tensor:
        return self.log_s.exp()

    @property
    def s_sigma(self) -> torch.Tensor:
        return self.log_s_sigma.exp()

    def forward(self, x, lora_B, scaling):
        # W = self.U @ torch.diag(self.s) @ self.Vh
        W = self.P(self.s).reshape(self.rank, -1)
        result = lora_B(F.linear(x, W)) * scaling

        if self.blobsample: 
            s_noise = self.s_sigma * torch.randn_like(self.log_s)
            W_noise = self.P(s_noise).reshape(self.rank, -1)
            # S_noise = torch.diag(s_noise)
            # W_noise = self.U @ S_noise @ self.Vh
            result += lora_B(F.linear(x, W_noise)) * scaling
        return result
    
    @property
    def kl_div(self) -> torch.Tensor:
        max_log_mu = torch.max(self.log_p_mu, self.log_s)
        mu_p_minus_mu_q = torch.exp(max_log_mu) * (
            # torch.exp(log_mu_p - max_log_mu) - torch.exp(log_mu_q - max_log_mu)
            torch.exp(self.log_p_mu - max_log_mu) - torch.exp(self.log_s - max_log_mu)
        )
        mu_p_minus_mu_q_squared = mu_p_minus_mu_q ** 2

        # Compute sigma^2 terms in log-space
        # log_sigma_p_squared = 2 * log_sigma_p
        # log_sigma_q_squared = 2 * log_sigma_q

        # Exponentiate for sigma^2 terms
        sigma_p_squared = torch.exp(2 * self.log_p_sigma)
        sigma_s_squared = torch.exp(2 * self.log_s_sigma)

        # KL divergence formula with log-space components
        kl_div = (
            # (log_sigma_q - log_sigma_p)  # log(sigma_q / sigma_p)
            (self.log_s_sigma - self.log_p_sigma)
            + (sigma_p_squared + mu_p_minus_mu_q_squared) / (2 * sigma_s_squared)
            - 0.5
        )
        return kl_div.sum()


class LinearFullRank(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 s_sigma_init_eps: float = 0.2, blobsample: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        rand_weights = nn.Linear(in_features, out_features, bias=False).weight.detach()
        U, s, Vh = torch.linalg.svd(rand_weights, full_matrices=False)
        self.rank = min(in_features, out_features)
        # self.U = nn.Parameter(U.contiguous())
        self.log_s = nn.Parameter(torch.log(s))
        self.Vh = nn.Parameter(Vh.contiguous())
        self.full_rank_cov = FullRankCov(self.rank)
        # self.register_buffer("p_mu", z)
        # self.register_buffer("log_p_sigma", z + 1)
        self.blobsample = blobsample

    @property
    def s(self) -> torch.Tensor:
        return self.log_s.exp()
    
    def forward(self, x, lora_B, scaling):
        W = torch.diag(self.s) @ self.Vh
        result = lora_B(F.linear(x, W)) * scaling

        if self.blobsample: 
            cov = self.full_rank_cov()
            L = torch.linalg.cholesky(cov)
            noise = torch.randn_like(self.log_s)
            s_noise = L @ noise
            S_noise = torch.diag(s_noise)
            W_noise = S_noise @ self.Vh
            result += lora_B(F.linear(x, W_noise)) * scaling
        return result
    
    @property
    def kl_div(self) -> torch.Tensor:
        full_cov = self.full_rank_cov()
        trace = torch.trace(full_cov)
        s = self.s
        quad = s @ s
        log_det_cov = torch.logdet(full_cov)
        kl_div = 0.5 * (trace + quad - self.rank - log_det_cov)
        return kl_div.sum()

