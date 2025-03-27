"""
TODO: Bert architecture layers
Author: Redal
Date: 2025/03/25
Homepage: https://github.com/Redal/Multimodal-Research.git
"""
import torch 
import torch.nn as nn
from torch.nn import functional as F
import math


def gelu(x):
    """ GELU activation function
    In the GPT architecture, an approximate version of the gelu function is used, with the following formula:
    0.5 * x * (1 + torch.tanh (math.sqrt (2/math.pi) * (x + 0.044715 * torch.pow (x, 3)))
    Here is the direct analytical solution, which is the formula given in the original paper
    Thesis https://arxiv.org/pdf/1606.08415"""
    x = x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return x

def swish(x):
    return x * torch.sigmoid(x)
activations = {'gelu': gelu, 'relu':F.relu, 'swish':swish}


class LayerNorm(nn.Module):
    """Layernorm layer, implemented here, the purpose is to be compatible with conditianal layernorm, 
        making it possible to do tasks such as conditional text generation and conditional classification
        Conditional layernorm comes from Su Jianlin's idea, details: https://spaces.ac.cn/archives/7124"""
    def __init__(self, hidden_size, eps=1e-12, conditional=False):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps
        self.conditional = conditional
        if self.conditional:
            #Conditional layernorm, for conditional text generation,
            #All-zero initialization is used here to avoid interfering
            #with the original pre-trained weights in the initial state
            self.dense1 = nn.Linear(2*hidden_size, hidden_size, bias=False)
            self.dense1.weight.data.uniform_(0, 0)
            self.dense2 = nn.Linear(2*hidden_size, hidden_size, bias=False)
            self.dense2.weight.data.uniform_(0, 0)
    
    def forward(self, x):
        if self.conditional:
            inputs = x[0]
            cond = x[1]
            for _ in range(len(inputs.shape)) - len(cond.shape):
                cond = cond.unsqueeze(dim=1)
            u = inputs.mean(-1, keepdim=True)
            s = (inputs - u).pow(2).mean(-1, keepdim=True)
            x = (inputs - u) / torch.sqrt(s + self.eps)
            return (self.weight + self.dense1(cond))*x + (self.bias + self.dense2(cond))
        else:
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            return self.weight * x + self.bias