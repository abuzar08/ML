# import numpy as np
import torch
from torch import optim
import torch.nn as nn

x = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
y = torch.tensor([[2],[4],[6],[8]], dtype = torch.float32)

x_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = x.shape
input_size = n_features
output_size = n_features
# model = nn.Linear(input_size, output_size)

class LinearRegression(nn.Module):

    def __init__(self, in_size, out_size):
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(in_size, out_size)
    
    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)

print(f"Prediction before training: {model(x_test).item():.3f}")

# Training Loop
n_iters = 100
lr = 0.01

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for i in range(n_iters):
    y_hat = model(x)
    l = loss(y,y_hat)
    
    # Backward
    l.backward()
    # grad = w.grad
    
    optimizer.step()
    
    # Zero Gradients
    optimizer.zero_grad()
    
    

print(f"Prediction after training: {model(x_test).item():.3f}")

