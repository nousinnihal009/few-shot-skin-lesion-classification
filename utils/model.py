# utils/model.py

import torch
import torch.nn as nn

class ProtoNet(nn.Module):
    def __init__(self, embedding_dim=64):
        super(ProtoNet, self).__init__()
        
        # Simple CNN encoder (you can swap with ResNet later)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Flatten()
        )
        
        # Output embedding layer
        self.fc = nn.Linear(64 * 28 * 28, embedding_dim)  # assuming 224x224 input

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x

def euclidean_distance(a, b):
    """
    Computes Euclidean distance between two embedding tensors
    a: [num_queries, embedding_dim]
    b: [num_support, embedding_dim]
    """
    return ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
