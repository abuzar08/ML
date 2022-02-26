from scipy.sparse.construct import rand
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler as ss

# 0 Data

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape

x_train, x_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=1234)

# Scale
sc = ss()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


# 1 Model
# f(x) = wx + b, sigmoid(f(x))

class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
    
    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(n_features)

# 2 Loss and Opt
lr = 0.01
J = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = lr)

# 3 Training
num_iters = 100

for epoch in range(num_iters):
    # forward
    y_hat = model(x_train)

    # loss
    loss = J(y_hat, y_train)

    # backward
    loss.backward()

    # update
    optimizer.step()

    # zero_grad
    optimizer.zero_grad()

    if (epoch+1)%10==0:
        print(f"epoch: {epoch+1}, loss: {loss.item():.3f}")

with torch.no_grad():
    y_predicted = model(x_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / y_test.shape[0]
    print(f"{acc=:.3f}")