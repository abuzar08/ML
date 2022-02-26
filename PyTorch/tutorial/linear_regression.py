import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Prepare data

X_np, Y_np = datasets.make_regression(n_samples = 100, n_features = 1, noise = 20, random_state = 1)

X = torch.from_numpy(X_np.astype(np.float32))
y = torch.from_numpy(Y_np.astype(np.float32))

y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape

# 1 Model
input_size, output_size = 1,1
model = nn.Linear(input_size, output_size)

# 2 Define loss and optimizer
J = nn.MSELoss()
lr = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr = lr)

# 3 Training loop
num_iters = 100

for epoch in range(num_iters):
    y_hat = model(X)
    loss = J(y_hat, y)

    # Backward
    loss.backward()

    # update
    optimizer.step()

    # zero grad
    optimizer.zero_grad()

    if (epoch+1)%10 ==0:
        print(f"epoch: {epoch+1}, loss: {loss.item():.3f}")

# Plot
predicted = model(X).detach().numpy()
plt.plot(X_np, Y_np, 'ro')
plt.plot(X_np, predicted, 'b')
plt.show()
