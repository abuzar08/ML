
import torch
from torch import nn
from utils import feed_forward, positional_encoding, scaled_dot_product_attention
from blocks import AttentionHead, MultiHeadAttention, Residual
from tqdm import tqdm

'''
TRANSFORMER:

1. Encoder: Processes inputs and returns a feauture vector.
2. Decoder: Processes target sequence, incorporates info from encoder memory.
            Outputs the Prediction!

'''

class TransformerEncoderLayer(nn.Module):
    '''
    Contain 2 residuals:
        1. Residual for MultiheadAttention (src, src, src)
        2. Residual for FeedForward
    '''
    def __init__ (
        self, 
        dim_model = 512, 
        num_heads = 6,
        dim_feedforward = 2048,
        dropout = 0.1
        ):

        super().__init__()
        dim_q = dim_k = max(dim_model // num_heads, 1)
        self.attention = Residual(
            MultiHeadAttention(
                num_heads = num_heads, 
                dim_in = dim_model,
                dim_q = dim_q,
                dim_k = dim_k
                ),
                dimension=dim_model,
                dropout = dropout
        )

        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension = dim_model,
            dropout = dropout
        )
    
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        '''
        Q,K,V = src
        '''
        attention = self.attention(src, src, src)
        return self.feed_forward(attention)

class TransformerEncoder(nn.Module): 
    '''
    1. Applies positional encoding
    2. Uses N x TransformerEncoderLayers
    '''
    def __init__(
        self, 
        num_layers = 6, 
        dim_model = 512, 
        num_heads = 8, 
        dim_feedforward = 2048, 
        dropout = 0.1
        ):

        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
    
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        seq_len, dimension = src.size(1), src.size(2)
        out = src + positional_encoding(seq_len, dimension)
        for layer in tqdm(self.layers, desc="Training Encoder"):
            out = layer(out)
        
        return out

class TransformerDecoderLayer(nn.Module):
    '''
    Decoder layer contains 3 residuals:
        1. Residual for MultiHeadAttention from (tgt, tgt, tgt) = l1
        2. Resudual for MultiHeadAttention from (memory, memory, l1) = l2
        3. Feed Forward (l2)
    '''
    def __init__(
        self,
        dim_model = 512,
        num_heads = 6,
        dim_feedforward = 2048,
        dropout = 0.1 ):
        super().__init__()
        dim_q = dim_k = max(dim_model // num_heads, 1)
        self.attention_1 = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension = dim_model,
            dropout = dropout
        )

        self.attention_2 = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension = dim_model,
            dropout = dropout
        )

        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout
        )
    
    def forward(self, tgt, memory):
        attention_1 = self.attention_1(tgt, tgt, tgt)
        attention_2 = self.attention_2(attention_1, memory, memory)
        return self.feed_forward(attention_2)

class TransformerDecoder(nn.Module):
    '''
    1. Gives positional encoding to target
    2. Contains N x Decoder Layers
    3. Linear layer at the end
    '''
    def __init__(
        self, 
        num_layers = 6, 
        dim_model = 512, 
        num_heads = 6,
        dim_feedforward = 2048,
        dropout = 0.1
        ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(dim_model, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

        self.linear = nn.Linear(dim_model, dim_model)
    
    def forward(self, tgt, memory):
        
        seq_len, dimension = tgt.size(1), tgt.size(2)
        out = tgt + positional_encoding(seq_len, dimension)

        for layer in tqdm(self.layers, desc="Training Decoder"):
            out = layer(out, memory)
        
        return torch.softmax(self.linear(tgt), dim=1)
        
