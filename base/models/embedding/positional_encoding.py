"""
TODO: transformer embedding positional_encoding
Author: Redal
Date: 2025/03/15
Homepage: https://github.com/Redal/Multimodal-Research.git
"""
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """compute positional encoding for input sentense:
    :param d_model: dimension of model
    :param max_len: max length of input sentense
    :param device: hardware device setting """
    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()
        # same size with input matrix for adding with input matrix
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        # 1D => 2D unsqueeze to represent word's postion
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)
        _2i = torch.arrange(0, d_model, step=2, device=device)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        # self.encoding [max_len=512, d_model=512]
        # [batch_size=128, seq_len=30]
        batch_size, seq_len = x.size()

        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]
        return self.encoding[:seq_len, :]
    