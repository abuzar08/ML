# from Transformer_1.utils import feed_forward
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, dataloader
import torchvision.transforms as transforms
import torch.nn.functional as f
from tqdm import tqdm
import math

# Assuming input is Batch_size x Sequence_length x Number_features
# batch_first = True

def scaled_dot_product_attention(Q, K, V, mask = False):
    '''
    Calculates scaled dot-product atention
    '''
    scaled_dot_product = Q.bmm(K.transpose(1,2))/math.sqrt(K.shape[2])
    # Optional Masking is here
    if mask: 
        indices = torch.triu_indices(scaled_dot_product.shape[1], scaled_dot_product.shape[1], offset=1)
        scaled_dot_product[:, indices[0], indices[1]] = float('-inf')

    softmax = f.softmax(scaled_dot_product, dim = 2)
    attention = softmax.bmm(V)
    return attention, softmax

def positional_encoding(seq_length, d_model):
    pos = torch.arange(seq_length, dtype = torch.float).reshape(1,-1,1)
    i = torch.arange(d_model, dtype=torch.float).reshape(1,1,-1)

    theta = pos / (1e4 ** ((i - i%2)/d_model))

    return torch.where(i.long() % 2==0, torch.sin(theta), torch.cos(theta))


# Blocks

class AttentionHead(nn.Module):
    def __init__(self, dim_in, dim_q, dim_v, mask = False):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_q)
        self.v = nn.Linear(dim_in, dim_v)
        self.mask = mask
    
    def forward(self, Q, K, V):
        attention, weights = scaled_dot_product_attention(self.q(Q), self.k(K), self.v(V), self.mask)
        return attention

class FeedForward(nn.Module):
    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, in_size)
        )
    
    def forward(self, x):
        return self.linear(x)

class Residual(nn.Module):
    def __init__(self, sublayer, dim_size, dropout):
        super().__init__()
        self.dim = dim_size
        self.sublayer = sublayer
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.dim)
    
    def forward(self, *args):
        return self.norm(args[0] + self.dropout(self.sublayer(*args))) #LayerNorm(x + sublayer(x))


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_in, dim_q, dim_v, num_heads, mask = False):
        super().__init__()
        self.mask = mask
        self.layers = nn.ModuleList(
            [
                AttentionHead(dim_in, dim_q, dim_v, self.mask)
                for _ in range(num_heads)
            ]
        )

        self.linear = nn.Linear(num_heads * dim_q, dim_in)
    
    def forward(self, Q, K, V):
        outs = [h(Q,K,V) for h in self.layers]
        out = torch.cat(outs, dim =  2)
        # return self.linear(out), out, outs
        return self.linear(out)


class encoderBlock(nn.Module):
    def __init__(self, d_in, num_heads, hidden_size, dropout):
        super().__init__()
        self.d_in = d_in
        self.d_q = self.d_v = self.d_in // num_heads
        self.dropout = nn.Dropout(dropout)
        self.attention = Residual(
                MultiHeadAttention(self.d_in, self.d_q, self.d_v, num_heads),
                self.d_in,
                dropout
            )
        
        self.ff = Residual(
                FeedForward(self.d_in, hidden_size),
                self.d_in,
                dropout
            )
    
    def forward(self, src):
        attention =  self.attention(src, src, src)
        return self.ff(attention)

class decoderBlock(nn.Module):
    def __init__(self, d_in, num_heads, hidden_size, dropout):
        super().__init__()
        self.d_in = d_in
        self.d_q = self.d_v = self.d_in // num_heads
        self.dropout = nn.Dropout(dropout)
        # self.mask = mask
        self.attention_1 = Residual(
                MultiHeadAttention(self.d_in, self.d_q, self.d_v, num_heads, mask = True),
                self.d_in,
                dropout
            )
        
        self.attention_2 = Residual(
                MultiHeadAttention(self.d_in, self.d_q, self.d_v, num_heads),
                self.d_in,
                dropout
            )
        
        self.ff = Residual(
                FeedForward(self.d_in, hidden_size),
                self.d_in,
                dropout
            )
    
    def forward(self, trg, memory = None):
        attention_1 = self.attention_1(trg, trg, trg)
        attention_2 = self.attention_2(attention_1, memory, memory)
        return self.ff(attention_2)

class encoder(nn.Module):
    def __init__(self, d_model, num_heads, dropout, hidden_size, num_layers):
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.hidden_size = hidden_size

        self.encoders = nn.ModuleList(
            [
                encoderBlock(self.d_model, self.num_heads, self.hidden_size, self.dropout)
                for _ in range(num_layers)
            ]
        )
    
    def forward(self, src):
        out = src
        seq_length, embedding_size = src.shape[1], src.shape[2]
        out += positional_encoding(seq_length, embedding_size)

        for layer in self.encoders:
            out = layer(out)

        return out

class decoder(nn.Module):
    def __init__(self, d_model, num_heads, dropout, hidden_size, num_layers):
        self.decoders = nn.ModuleList(
            [
                decoderBlock(d_model, num_heads, hidden_size, dropout)
                for _ in range(num_layers)
            ]
        )

        self.linear = nn.Linear(d_model, d_model)
    
    def forward(self, trg, memory):
        seq_length, embedding_size = trg.shape[1], trg.shape[2]
        out = trg + positional_encoding(seq_length, embedding_size)

        for layer in self.decoders:
            out = layer(out, memory)
        
        return torch.softmax(self.linear(out))

src = torch.rand(3, 4, 10)
y, weights = scaled_dot_product_attention(src, src, src)

# print("src\n", src)
# print("weights\n",weights)
# print("y\n",y)


# with torch.no_grad():
#     MHA = MultiHeadAttention(10, 5, 5, 8)
#     y, concat, heads = MHA(src, src, src)
#     print(len(heads), "heads of shape: ", heads[0].shape)
#     print(concat.shape)
#     print(y.shape)
#     print(src.shape)

with torch.no_grad():
    enc = encoderBlock(10, 5, 20, 0.3)
    encoder_out = enc(src)
    print(encoder_out.shape)

    



