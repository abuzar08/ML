import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader, dataloader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as f
from tqdm import tqdm



# Device Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 4
batch_size = 4
lr = 1e-3

# CIFAR10 has PILImage size of range [0,1]
# We transform them to Tensors of normalized range [-1,1]
transform =  transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
    )

# Load datasets
train_dataset = torchvision.datasets.CIFAR10(root = './data', train=True,
             download=True, transform = transform)

test_dataset = torchvision.datasets.CIFAR10(root = './data', train = False,
            download=False, transform = transform)

# Dataloaders

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size,
             shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
             shuffle=False)

classes = "plane car bird cat deer dog frog horse ship truck".split()
classes = tuple(classes)

class relu(nn.Module):
    def __init__(self):
        super().__init__()
        # self.relu = f.relu

    def forward(self, x):
        return f.relu(x)

class conv(nn.Module):
    def __init__(self, prev, channels, kernel_width):
        super().__init__()
        self.prev = prev
        self.channels = channels
        self.kernel_width = kernel_width
        self.conv = nn.Conv2d(self.prev, self.channels, self.kernel_width)
    
    def forward(self, x):
        return self.conv(x)

class maxPool(nn.Module):
    def __init__(self, k, s):
        super().__init__()
        self.k = k
        self.s = s
        self.maxpool = nn.MaxPool2d(k,s)
    
    def forward(self, x):
        return self.maxpool(x)

# Implement the COnvNet
class CNN(nn.Module):
    def __init__(self):

        super().__init__()

        self.convolution = nn.ModuleList(
            [
                conv(3, 6, 5),
                relu(),
                maxPool(2,2),
                conv(6, 16, 5),
                relu(),
                maxPool(2,2)
            ]
        )
        self.linear = nn.ModuleList(
            [
                nn.Linear(16*5*5,120),
                relu(),
                nn.Linear(120, 84),
                relu(),
                nn.Linear(84, 10)
            ]
        )

    def forward(self, x):

        out_conv = x
        for layer in self.convolution:
            out_conv = layer(out_conv)

        x = out_conv.view(-1,16*5*5)

        out_lin = x
        for layer in self.linear:
            out_lin = layer(out_lin)

        out = out_lin

        return out

model = CNN().to(device)

# criterion = nn.KLDivLoss()
criterion = nn.CrossEntropyLoss()
optimzer = torch.optim.AdamW(model.parameters(), lr = lr)

items = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Original shape: [4, 3, 32, 32] -> [4, 3, 1024]

        images = images.to(device)
        labels = labels.to(device)

        # Forward
        out = model(images)
        loss = criterion(out, labels)

        # Backward
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()

        if (i+1) % 2000 == 0:
            print(f"Epoch: {epoch+1}/{num_epochs}, Step: {i+1}/{items}, Loss: {loss.item():.3f}")

print("Finished Training")

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            
            n_class_samples[label] += 1
    
    acc = 100.0 * n_correct / n_samples
    print(f"Accuracy of the network: {acc:.2f}%")

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i] 
        print(f"Accuracy of {classes[i]}: {acc:.2f}%")

