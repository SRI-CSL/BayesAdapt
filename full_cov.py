import torch
import torch.nn as nn
import torch.nn.functional as F

class FullRankCov(nn.Module):
    def __init__(self, dim=8, cov_add_eps=1e-8): 
        super().__init__()
        self.eigenvalues = nn.Parameter(torch.randn(dim))
        self.eigenvectors = nn.Parameter(torch.randn(dim, dim))
        self.register_buffer('cov_add_eps', torch.eye(dim) * cov_add_eps)
    
    def forward(self):
        eigenvalues = F.softplus(self.eigenvalues)
        eigenvalues = torch.diag_embed(eigenvalues)
        eigenvectors, _ = torch.linalg.qr(self.eigenvectors) #orthogonalize
        cov = eigenvectors @ eigenvalues @ eigenvectors.T
        cov = (cov + cov.T) / 2.0 #symmetrize
        cov = cov + self.cov_add_eps
        return cov
