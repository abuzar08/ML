import torch
from torch import nn
from utils import *

class AttentionHead(nn.Module):
    '''
    1. create the three linear layers to pass the query, keym and value
    through.
    2. return the scaled dot-product attention of the three.
    '''
    def __init__(self,dim_in, dim_q, dim_k):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)
    
    def forward(self, Q, K, V):
        # print(f"AttentionHead:\n{Q.shape=}, {K.shape}, {V.shape}")
        return scaled_dot_product_attention(self.q(Q),self.k(K),self.v(V))

class MultiHeadAttention(nn.Module):
    '''
    Create a multiple heads that pay attention to different parts of
    the input sequence. Each head passes through the scaled dot-product
    and finally gets concatenated along dimension 1 to create output 
    of size ?? - Confirm.
    '''
    def __init__(self, num_heads: int, dim_in: int, dim_q, dim_k):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads*dim_k, dim_in)
        # print(f"MHA: {dim_in=}, {dim_k=}, {dim_q=}, {num_heads=}")
    
    def forward(self, Q, K, V):
        # print(f"MHAttention:\n{Q.shape=}, {K.shape}, {V.shape}")
        ans =  self.linear(
            torch.cat([h(Q,K,V) for h in self.heads], dim=2)
        )
        # print(f"{ans.shape=}")
        return ans

class Residual(nn.Module):
    '''
    Whatever's left. End of each sub-layer.
    LayerNorm(x + sublayer(x))
    '''
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        '''
        Assume that Q is given first, compute residual as such.
        This matches MultiHeadAttention signature
        '''
        # print(f"Residual:\n{tensors=}")s
        return self.norm(tensors[0] + self.dropout(self.sublayer(*tensors)))
