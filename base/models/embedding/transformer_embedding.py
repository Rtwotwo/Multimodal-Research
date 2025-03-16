"""
TODO: transformer embedding transformer_embedding
Author: Redal
Date: 2025/03/15
Homepage: https://github.com/Redal/Multimodal-Research.git
"""
from torch import nn
from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.token_embedding import TokenEmbedding


class TransformerEmbedding(nn.Mudule):
    """compute transformer embedding with token embedding 
    and positional encoding(sinusoid)"""
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.tok_emb(x) + self.pos_emb(x)
        return self.dropout(x)
