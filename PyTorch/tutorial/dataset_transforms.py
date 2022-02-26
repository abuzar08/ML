import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, dataloader
import numpy as np
import math

class WineDatatset(Dataset):

    def __init__(self, transform = None):
        xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])# n_samples x 1
        self.n_samples = self.x.shape[0]

        self.transform = transform

    def __getitem__(self, index):
        sample =  self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)
        
        return sample

    def __len__(self):
        return self.n_samples

class ToTensor:
    def __call__(call, sample):
        inputs, lables = sample
        return torch.from_numpy(inputs), torch.from_numpy(lables)

class MulTransform:
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, sample):
        inputs, lables = sample
        inputs *= self.factor
        return inputs, lables

dataset = WineDatatset()
transformed = WineDatatset(transform=ToTensor()) # Transformation applied

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
compose_transformed = WineDatatset(transform=composed) 