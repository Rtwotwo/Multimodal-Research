"""
TODO: Bert architecture layers
Author: Redal
Date: 2025/03/25
Homepage: https://github.com/Redal/Multimodal-Research.git
"""
import torch 
import torch.nn as nn
from torch.nn import functional as F
import math


def gelu(x):
    """ GELU activation function
    In the GPT architecture, an approximate version of the gelu function is used, with the following formula:
    0.5 * x * (1 + torch.tanh (math.sqrt (2/math.pi) * (x + 0.044715 * torch.pow (x, 3)))
    Here is the direct analytical solution, which is the formula given in the original paper
    Thesis https://arxiv.org/pdf/1606.08415"""
    x = x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return x


def swish(x):
    return x * torch.sigmoid(x)
activations = {'gelu': gelu, 'relu':F.relu, 'swish':swish}


class LayerNorm(nn.Module):
    """Layernorm layer, implemented here, the purpose is to be compatible with conditianal layernorm, 
        making it possible to do tasks such as conditional text generation and conditional classification
        Conditional layernorm comes from Su Jianlin's idea, details: https://spaces.ac.cn/archives/7124"""
    def __init__(self, hidden_size, eps=1e-12, conditional=False):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps
        self.conditional = conditional
        if self.conditional:
            #Conditional layernorm, for conditional text generation,
            #All-zero initialization is used here to avoid interfering
            #with the original pre-trained weights in the initial state
            self.dense1 = nn.Linear(2*hidden_size, hidden_size, bias=False)
            self.dense1.weight.data.uniform_(0, 0)
            self.dense2 = nn.Linear(2*hidden_size, hidden_size, bias=False)
            self.dense2.weight.data.uniform_(0, 0)
    def forward(self, x):
        if self.conditional:
            inputs = x[0]
            cond = x[1]
            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(dim=1)
            u = inputs.mean(-1, keepdim=True)
            s = (inputs - u).pow(2).mean(-1, keepdim=True)
            x = (inputs - u) / torch.sqrt(s + self.eps)
            return (self.weight + self.dense1(cond))*x + (self.bias + self.dense2(cond))
        else:
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            return self.weight * x + self.bias
        

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, 
                 dropout_rate, attention_scale=True,
                 return_attention_scores=False):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % n_heads == 0
        # define the basic parameters
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.attention_scale = attention_scale
        self.return_attention_scores = return_attention_scores
        # define the linear layers consisting of q, k, v, and o
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.o = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
    def transpose_for_scores(self,x):
        # return x's shape: (batch_size, seq_len, n_heads, attention_head_size)
        new_x_shape = x.size()[:-1] + (self.n_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    def forward(self, q, k, v, attention_mask=None):
        # query shape: [batch_size, query_len, hidden_size]
        # key shape: [batch_size, key_len, hidden_size]
        # value shape: [batch_size, value_len, hidden_size]
        # In general, three shapes are equal
        mixed_query_layer = self.q(q)
        mixed_key_layer = self.k(k)
        mixed_value_layer = self.v(v)
        # The shapes of the three are the same as above

        # query_layer shape: [batch_size, num_attention_heads, query_len, attention_head_size]
        # key_layer shape: [batch_size, num_attention_heads, key_len, attention_head_size]
        # value_layer shape: [batch_size, num_attention_heads, value_len, attention_head_size]
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # exchange the last two dims of k, in oroder to calculate attention scores
        # attention_scores shape: [batch_size, num_attention_heads, query_len, key_len]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.attention_scale:
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # process the attention_mask
        #The value is -1e10, after softmax, the attention_probs is almost 0, 
        # so the part with mask 0 will not be noticed
        if attention_mask is not None:
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask
        
        # Normalize the attention scores to probabilities about 0~1
        # context_layer shape: [batch_size, num_attention_heads, query_len, attention_head_size]
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)

        # After the dimension transformation operations such as transpose and permute, the tensor 
        # is no longer stored contiguously in memory, while the view operation requires 
        # the tensor's memory to be stored contiguously.
        # So before calling view, you need contiguous to return a contiguous copy
        context_layer = context_layer.permute(0,2,1,3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size, )
        context_layer = context_layer.view(*new_context_layer_shape)

        # whether return attention scores
        if self.return_attention_scores:
            # The attention_scores returned here has not gone through softmax 
            # and can be normalized externally
            return self.o(context_layer), attention_scores
        else: return self.o(context_layer)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, 
                 dropout_rate=0.5, is_dropout=True):
        super(PositionWiseFeedForward, self).__init__()
        