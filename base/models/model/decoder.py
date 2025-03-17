"""
TODO: transformer model decoder
Author: Redal
Date: 2025/03/15
Homepage: https://github.com/Redal/Multimodal-Research.git
"""

import torch
from torch import nn
from models.blocks.decoder_layer import DecoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, 
                 n_heads, n_layers, drop_prob, device):
        """dec_voc_size: The size of the decoder's vocabulary.
        max_len: The maximum sequence length.
        d_model: The dimension of the model (embedding dimension).
        ffn_hidden: The dimension of the hidden layer in the feed-forward network.
        n_heads: The number of heads in the multi-head attention mechanism.
        n_layers: The number of stacked layers in the decoder.
        drop_prob: The probability of dropout.
        device: The device on which the model runs (CPU or GPU)."""
        super(Decoder, self).__init__()
        self.emb = TransformerEmbedding(dec_voc_size,
                     d_model, max_len, drop_prob, device)
        self.layers = nn.ModuleList([DecoderLayer(d_model, 
                                                  ffn_hidden, 
                                                  n_heads, 
                                                  drop_prob) 
                                    for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, dec_voc_size)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        """trg: The input of the target sequence.
        enc_src: The output of the encoder (context information from the encoder).
        trg_mask: The mask for the target sequence, used to mask future information.
        src_mask: The mask for the source sequence, used to handle padding in the 
                input sequence and prevent the model from attending to padded positions."""
        trg = self.emb(trg)
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        # pass to LM head
        output = self.linear(trg)
        return output
        