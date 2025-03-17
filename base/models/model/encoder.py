"""
TODO: transformer model encoder
Author: Redal
Date: 2025/03/15
Homepage: https://github.com/Redal/Multimodal-Research.git
"""
from torch import nn
from models.blocks.encoder_layer import EncoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, ffn_hidden, n_heads,
                  n_layers, max_len, drop_prob, device):
        super(Encoder, self).__init__()
        self.emb = TransformerEmbedding(vocab_size, d_model, 
                                max_len, drop_prob, device)
        self.layers = nn.ModuleList([EncoderLayer(d_model,
                                                  ffn_hidden,
                                                  n_heads,
                                                  drop_prob)
                                    for _ in range(n_layers)])
        
    def forward(self, x, src_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)
        
        return x
