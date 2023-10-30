import torch
import torch.nn as nn
from torch.nn import functional as F
"""
https://www.sciencedirect.com/science/article/pii/S016786551930056X?via%3Dihub
"""

class UlasNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(12, 64, kernel_size=5, stride=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1)
        self.maxpool1 = nn.MaxPool1d(2, stride=2)
        self.dropout = nn.Dropout(0.2)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=13, stride=1)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=7, stride=1)
        self.maxpool2 = nn.MaxPool1d(2, stride=2)
        self.flatten = nn.Flatten(1)
        self.fc1 = nn.Linear(61184, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, features, labels=None):
        x = self.conv1(features)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        logits = torch.sigmoid(x)

        if labels is not None:
            loss = F.binary_cross_entropy(logits, labels)
            return logits, loss
        else:
            return logits
