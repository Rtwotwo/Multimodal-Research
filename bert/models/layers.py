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

