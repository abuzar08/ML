import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader, dataloader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Hyper parameters
input_size =  28*28
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
lr = 0.001

# MNIST
train_dataset = torchvision.datasets.MNIST(root = './data', train=True, 
    transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root = './data', train=False, 
    transform=transforms.ToTensor(), download=False)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=batch_size,
    shuffle= True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size=batch_size,
    shuffle= False)

examples = iter(train_loader)
samples,lables = examples.next()

# print(samples.shape, lables.shape)

class NeuralNet(nn.Module):
    def __init__(self, n, h, k):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(n,h)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(h,k)
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)

        # NO SOFTMAX - WILL USE CROSSENTROPY FROM TORCH
        return out

model = NeuralNet(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss() # already aplies the softmax for us.

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training loop
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # reshape: flatten 100,1,28,28 to 100,784
        images = images.reshape(-1, 28*28)

        # forward
        out = model(images)
        loss = criterion(out, labels)

        # backward
        loss.backward()

        # update
        optimizer.step()

        optimizer.zero_grad()

        if (i+1) %100 == 0:
            print(f"epoch: {epoch+1} / {num_epochs}, step: {i+1} / {n_total_steps}, loss: {loss.item():.3f}")

# Testing
with torch.no_grad():
    n_correct = 0
    n_sample = 0

    for images, labels in test_loader:
        images = images.reshape(-1, 28*28)
        outputs = model(images)

        # val, index
        _, prediction = torch.max(outputs, 1)

        n_sample += labels.shape[0]
        n_correct += (prediction == labels).sum().item()
    
    acc = 100.0 * (n_correct/n_sample)
    print(f"{acc=}")