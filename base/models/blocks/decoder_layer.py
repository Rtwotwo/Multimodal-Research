"""
TODO: transformer blocks decoder_layer
Author: Redal
Date: 2025/03/15
Homepage: https://github.com/Redal/Multimodal-Research.git
"""
from torch import nn
from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.feed_forward import FeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_heads, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

        
        self.ffn = FeedForward(d_model, ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(drop_prob)

    def forward(self, dec, enc, tgt_mask, src_mask):
        # 1.comute self attention
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=tgt_mask)
        x = self.dropout1(x)
        x = self.norm(x + _x)

        # 2.compute cross attention
        if enc is not None:
            _x = x 
            x = self.cross_attention(q=x, k=enc, v=enc, mask=src_mask)
            x = self.dropout2(x)
            x = self.norm(x + _x)

        # 3.feed forward
        _x = x
        x  = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm(x + _x)
        return x
