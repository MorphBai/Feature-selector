import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from skimage import measure
from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from .resnet import conv1x1, conv3x3, BasicBlock, Bottleneck
from torch import Tensor


class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, layers=[3, 4, 23, 3], num_classes=5) -> None:
        super().__init__()

        self.inplanes = 64

        # PTB-XL为12导联数据
        self.conv1 = nn.Conv1d(
            12, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2)
        self.globalavgpool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, Bottleneck):
                # type: ignore[arg-type]
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                # type: ignore[arg-type]
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, features, labels=None) -> Tensor:
        x = self.conv1(features)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.globalavgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_out(torch.relu(x))
        x = torch.sigmoid(x)

        if labels is not None:
            loss = F.binary_cross_entropy(x, labels)    # 计算loss
            return x, loss
        else:
            return x
