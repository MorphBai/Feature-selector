import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor, dropout
import numpy as np
import transformers.models.vit_mae.modeling_vit_mae


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        norm_layer=None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(.2)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MSNet(nn.Module):
    def __init__(self, block=SEBasicBlock, layers=[3, 4, 6, 3], num_classes=5) -> None:
        super().__init__()

        self.inplanes = 64

        # PTB-XL为12导联数据
        mlayers = []
        for size in [11, 15, 19, 23]:
            mlayers.append(nn.Sequential(
                nn.Conv1d(12, self.inplanes, kernel_size=size,
                          stride=2, padding=5, bias=False),
                nn.BatchNorm1d(self.inplanes),
                nn.ReLU(inplace=True),
                # nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
                self._make_layer(block, 64, layers[0]),
                self._make_layer(
                    block, 128, layers[1], stride=2),
                self._make_layer(
                    block, 256, layers[2], stride=2),
                self._make_layer(
                    block, 512, layers[3], stride=2),
                # nn.AdaptiveAvgPool1d(1),
                nn.Linear(1024, 1024),
                nn.TransformerEncoderLayer(
                    d_model=1024, nhead=8, batch_first=True, dim_feedforward=4096)
            ))
        self.layers = nn.ModuleList(mlayers)
        self.fc = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(0.2)
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
            if isinstance(m, SEBasicBlock):
                # type: ignore[arg-type]
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm1d(planes)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, features, labels=None) -> Tensor:
        x = features.transpose(1, 2)
        tmp = []
        s = 0
        for _ in range(11):
            part = x[:, s:s+100, :]
            tmp.append(part)
            s += 90
        x = torch.stack(tmp, dim=1)

        x = x.reshape(-1, x.size(2), x.size(3))
        x = x.transpose(2, 1)

        fusion = []
        for layer in self.layers:
            f = layer[:-2](x)
            f1 = F.adaptive_avg_pool1d(f, 1)
            f2 = F.adaptive_max_pool1d(f, 1)
            f = torch.cat([f1, f2], dim=-1)
            f = torch.flatten(f, 1)
            f = f.view(-1, 11, f.size(1))
            t = layer[-2](f)
            t = torch.layer_norm(t, t.shape[1:])
            t = torch.relu(t)
            f = layer[-1](t)
            f = f + t
            fusion.append(f)

        x = torch.mean(torch.stack(fusion), dim=0)

        x = torch.mean(x, dim=1)

        x = self.fc(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        x = torch.sigmoid(x)

        if labels is not None:
            loss = F.binary_cross_entropy(x, labels)
            return x, loss
        else:
            return x


if __name__ == '__main__':
    x = torch.randn((2, 12, 1000))
    model = MSNet(num_classes=5)
    model(x)
