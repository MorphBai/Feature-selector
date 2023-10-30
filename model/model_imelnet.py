import torch
import torch.nn as nn
from torch.nn import functional as F
import torchsummary as summary

"""
https://ieeexplore.ieee.org/document/9658706
"""

class Attention(nn.Module):
    def __init__(self, w_size, dim=64):
        super().__init__()
        self.W = nn.Linear(w_size, dim)
        self.V = nn.Linear(dim, 1)

    def forward(self, x):
        e = torch.tanh(self.W(x))
        e = self.V(e)
        a = torch.softmax(e, dim=1)
        output = x * a
        return torch.sum(output, dim=1), a


class ResidualBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 downsample,
                 stride=1):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv1d(in_channels, out_channels,
                               kernel_size=7, stride=stride, padding=3)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels,
                               kernel_size=7, stride=1, padding=3)
        if downsample:
            self.conv3 = nn.Conv1d(in_channels, out_channels,
                                   kernel_size=1, stride=2)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.bn1(y)
        y = self.conv2(y)
        if self.downsample:
            x = self.conv3(x)
        out = x + y
        out = self.relu2(out)
        out = self.bn2(out)
        return out


class IMELNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 7, padding=3)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(32, 32, 2)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        self.attention1 = Attention(128, 13)
        self.lstm = nn.LSTM(input_size=128, hidden_size=64,
                            batch_first=True, bidirectional=True)
        self.attention2 = Attention(128, 20)
        self.attention3 = Attention(128, 12)
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        if stride != 1 or in_channels != out_channels:
            downsample = True
        else:
            downsample = False
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels,
                                    downsample, stride=stride))
        in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(
                in_channels, out_channels, downsample=False))
        return nn.Sequential(*layers)

    def forward(self, features, labels=None):
        x = features.view(-1, 50, 1)
        x = x.transpose(2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.transpose(2, 1)
        x, _ = self.attention1(x)
        x = x.view(-1, 20, 128)
        x, (hn, cn) = self.lstm(x)
        x, _ = self.attention2(x)
        x = x.view(-1, 12, 128)
        x, _ = self.attention3(x)
        x = self.fc(x)
        logits = torch.sigmoid(x)

        if labels is not None:
            loss = F.binary_cross_entropy(logits, labels)
            return logits, loss
        else:
            return logits


if __name__ == '__main__':
    model = IMELNet(num_classes=5)
    x = torch.randn((1, 1000, 12))
    model(x)
    # summary.summary(model, input_size=(12, 1000), batch_size=1, device='cpu')
