import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, dataloader
import numpy as np
import math

class WineDatatset(Dataset):

    def __init__(self):
        xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])# n_samples x 1
        self.n_samples = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

dataset = WineDatatset()
loader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
# data_iter = iter(loader)
# first_data = data_iter.next()

# features, labels = first_data

# print(features, labels)

# Training Loop
num_epochs = 100
totol_samples = len(dataset)
num_iter = math.ceil(totol_samples/4)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(loader):

        if (i+1)%5==0:
            print(f"epoch {epoch+1} / {num_epochs}, step {i+1}/{num_iter}, input {inputs.shape}")