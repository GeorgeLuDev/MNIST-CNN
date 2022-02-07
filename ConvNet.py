import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(1,40,5)
        self.conv2 = nn.Conv2d(40,40,5)
        self.fc1 = nn.Linear(40*4*4,100)
        self.fc2 = nn.Linear(100,100)
        self.output_layer = nn.Linear(100,10)
        
    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X,(2,2))
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X,(2,2))
        X = self.flatten(X)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.output_layer(X)
        return X
