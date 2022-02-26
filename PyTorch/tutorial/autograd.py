# Gradients using autograd

import torch

x = torch.randn(3, requires_grad=True)
print(f"{x=}")
y = x*x + 2

y.retain_grad() # required for non-leaf

print(f"{y=}")
z = y*y*3
print(f"{z=}")
l = z.mean()
l.backward() # dl/dx
print(x.grad)
print(y.grad) # Non-leaf needs .retain_grad() to be called
