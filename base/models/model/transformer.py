"""
TODO: transformer model transformer
Author: Redal
Date: 2025/03/15
Homepage: https://github.com/Redal/Multimodal-Research.git
"""
import torch
from torch import nn
from models.model.encoder import Encoder
from models.model.decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx,
                  enc_voc_size, dec_voc_size, d_model, n_heads,
                max_len, ffn_hidden, n_layers, drop_prob, device):
        """src_pad_idx: The index of the padding token in the source vocabulary.
        trg_pad_idx: The index of the padding token in the target vocabulary.
        trg_sos_idx: The index of the "Start Of Sentence" token in the target vocabulary.
        enc_voc_size: The size of the encoder's vocabulary.
        dec_voc_size: The size of the decoder's vocabulary."""
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        self.encoder = Encoder(enc_voc_size, d_model, n_heads, max_len, 
                               ffn_hidden, n_layers, drop_prob, device)
        self.decoder = Decoder(dec_voc_size, d_model, n_heads, max_len,
                                ffn_hidden, n_layers, drop_prob, device)
    
    def forward(self, src, trg):
        """src: The source sentence represented as a tensor of shape (seq_len, batch_size).
        trg: The target sentence represented as a tensor of shape (seq_len, batch_size)."""
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, trg_mask, src_mask)
        return out
    
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask
    