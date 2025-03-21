"""
TODO: transformer embedding token_embedding
Author: Redal
Date: 2025/03/15
Homepage: https://github.com/Redal/Multimodal-Research.git
"""
from torch import nn
from models.embedding.positional_encoding import PositionalEncoding


class TokenEmbedding(nn.Module):
    """compute token embedding with positional encoding
    :param vocab_size: size of vocabulary
    :param d_model: dimension of model"""
    def __init__(self, vocab_size, d_model, padding_idx=1):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

    def forward(self, x):
        return self.embedding(x)