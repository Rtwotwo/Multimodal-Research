"""
TODO: transformer layers layer_norm
Author: Redal
Date: 2025/03/15
Homepage: https://github.com/Redal/Multimodal-Research.git
"""
import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # -1 means last dimension
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        
        # normlization with gamma and beta
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma*out + self.beta
        return out
    