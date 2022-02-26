import torch
import numpy as np

def basics():
    x = torch.rand(5)
    y = torch.ones(4,5)
    z = torch.zeros(5,3)

    print(y.size())

def ops():
    x = torch.rand(2)
    y = torch.rand(2)

    z = x + y # Element wise addition
    z = torch.add(x,y)

    # in place (trailing _ is in place)
    y.add_(x)

    # Slicing is regular
    # x[:,2] or something

def get_item(tensor, i, j):
    return tensor[i,j].item()

def reshape():
    x = torch.rand(4,4)
    print(x.view(-1,8))

def convert_to_np():
    x = torch.ones(4)
    y = x.numpy()

    # Both share the same location if Tensor on CPU, not GPU.
    # Changing one changes the other as well

    # REVERSE:
    z = torch.from_numpy(y)

# reshape()
