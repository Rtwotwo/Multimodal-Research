"""
TODO: transformer layers scale_dot_product_attention
Author: Redal
Date: 2025/03/15
Homepage: https://github.com/Redal/Multimodal-Research.git
"""
import math 
from torch import nn


class ScaleDotProductAttention(nn.Module):
    """ compute scale dot product attention
    :param query: given sentence that we focused on (decoder)
    :param key: every sentence to check relationship with query(encoder)
    :param value: every sentence same with key(encoder)"""
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input size [batch_size, n_heads, length, d_tensor]
        batch_size, n_heads, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)
        score = (q @ k_t) / math.sqrt(d_tensor)
        # 2. apply masking(optional)
        if mask is not None:
            score = score.masked_fill(mask==0. -10000)
        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)
        # 4. multiply with value
        v = score @ v
        return v, score