import torch
from torch import nn
from transformer import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer
from utils import scaled_dot_product_attention
from blocks import *

class Transformer(nn.Module):
    '''
    Putting the Encoder and Decoder blocks together.
    '''
    def __init__(
        self,
        num_encoder_layers = 6,
        num_decoder_layers = 6,
        dim_model = 512,
        num_heads = 6,
        dim_feedforward = 2048,
        dropout = 0.1,
        activation = nn.ReLU()
        ):
        super().__init__()
        self.encoder = TransformerEncoder(
            num_encoder_layers, 
            dim_model, 
            num_heads, 
            dim_feedforward, 
            dropout
        )

        self.decoder = TransformerDecoder(
            num_decoder_layers,
            dim_model,
            num_heads,
            dim_feedforward,
            dropout
        )
    
    def forward(self, src, tgt):
        return self.decoder(tgt, self.encoder(src))


if __name__ == "__main__":
    src = torch.rand(64, 32, 512)
    tgt = torch.rand(64, 16, 512)
    transformer = Transformer()
    out = transformer(src, tgt)
    print(out.shape)
