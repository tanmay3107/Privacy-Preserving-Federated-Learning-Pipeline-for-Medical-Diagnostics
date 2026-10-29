import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

class MedicalCNN(nn.Module):
    """Convolutional Neural Network for medical image classification."""
    def __init__(self):
        super(MedicalCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 2)  # Binary classification: Normal vs. Abnormal

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_partition_data(partition_id: int, num_partitions: int = 2):
    """Simulates distributed hospital data partitions using CIFAR-10 subset."""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Split dataset indices across hospital nodes
    partition_size = len(trainset) // num_partitions
    indices = np.arange(len(trainset))
    
    node_indices = indices[partition_id * partition_size : (partition_id + 1) * partition_size]
    train_subset = Subset(trainset, node_indices)

    trainloader = DataLoader(train_subset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    return trainloader, testloader