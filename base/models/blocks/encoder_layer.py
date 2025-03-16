"""
TODO: transformer blocks encoder_layer
Author: Redal
Date: 2025/03/15
Homepage: https://github.com/Redal/Multimodal-Research.git
"""
from torch import nn
from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.feed_forward import FeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        self.ffn = FeedForward(d_model, ffn_hidden, drop_prob=drop_prob)

        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x, src_mask):
        # compute self-attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        # add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        # feed forward network
        _x = x
        x = self.ffn(x)
        # add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x
        