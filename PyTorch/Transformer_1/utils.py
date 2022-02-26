import torch
import torch.nn.functional as f
from torch import nn

def scaled_dot_product_attention(Q, K, V, mask = None):
    '''
    1. Without mask: Attention = softmax(QK.T/sqrt(d_k)).V
    2. With mask, we mask out the future in the decoder block.
    '''

    if mask is None:
        # print(f"{Q.shape=}")
        # print(f"{K.shape=}")

        M1 = Q.bmm(K.transpose(1,2))
        root_dk = Q.size(-1)**(0.5)
        softmax = f.softmax(M1/root_dk, dim=1)
        # print(f"{softmax.shape=}")
        attention = softmax.bmm(V)
        # print(f"{attention.shape=}")
        return attention
    
    else:
        return NotImplemented

def positional_encoding(seq_len, dim_model, device = torch.device("cpu")):
    '''
    All calculations in the attention heads happen along sfeature dimension,
    and are independent of sequence dimension. Need to give positional information.
    Do this by trigonometric functions: sin and cos.

    Why? Allow easy extention to longer sequences.
    '''
    pos = torch.arange(seq_len, dtype = torch.float, device = device).reshape(1,-1,1)
    dim = torch.arange(dim_model, dtype = torch.float, device = device).reshape(1,1,-1)
    # phase = pos/(1e4**(dim//dim_model))
    phase = pos / (1e4 ** ((dim - dim%2)/dim_model))

    return torch.where(dim.long() %2 ==0, torch.sin(phase), torch.cos(phase))

def feed_forward(dim_input=512, dim_feedforward=2048):
    '''
    Returns a feed-forward block
    '''
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_input)
    )

