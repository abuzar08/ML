import numpy as np

x = np.array([1,2,3,4], dtype=np.float32)
y = np.array([2,4,6,8], dtype = np.float32)

w = 0.0

def forward(x):
    return w*x

def loss(y, y_hat):
    return ((y_hat-y)**2).mean()
    # MSE
# dJ/dw = (1/N) * 2 * (y_hat-y) * x

def gradient(x,y,y_hat):
    return np.dot(2*x, y_hat - y).mean()

print(f"Prediction before training: {forward(5):.3f}")

# Training Loop
n_iters = 20
lr = 0.01

for i in range(n_iters):
    y_hat = forward(x)
    l = loss(y,y_hat)
    grad = gradient(x,y,y_hat)

    w = w - lr*grad

print(f"Prediction after training: {forward(5):.3f}")

