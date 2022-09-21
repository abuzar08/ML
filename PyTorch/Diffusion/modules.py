import torch
from torch import nn 

class SelfAtttention(nn.Module):
    def __init__(self):
        super().__init__()

class Unet(nn.Module):
    '''
    Unet architecture
    '''
    def __init__(self, c_in = 3, c_out = 3, time_dim=256, device='cpu'):
        super().__init__()
        