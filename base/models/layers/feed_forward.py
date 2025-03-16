"""
TODO: transformer layers feed_forward
Author: Redal
Date: 2025/03/15
Homepage: https://github.com/Redal/Multimodal-Research.git
"""
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        # two layers of MLP
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        # nonlinear transformation
        x = self.relu( self.linear1(x) )
        # linear transformation and for residual join
        x = self.linear2( self.dropout(x) )
        return x