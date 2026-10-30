"""ResNet18/34，输入按 64x64 调整过。"""
from __future__ import annotations

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """残差块。"""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


def _make_layer(in_channels: int, block: type[BasicBlock], out_channels: int, blocks: int, stride: int) -> tuple[nn.Sequential, int]:
    layers = [block(in_channels, out_channels, stride)]
    in_channels = out_channels
    for _ in range(1, blocks):
        layers.append(block(in_channels, out_channels, 1))
    return nn.Sequential(*layers), in_channels


class ResNet(nn.Module):
    """ResNet 主体，stem 可选 small 或 imagenet。"""

    def __init__(self, block: type[BasicBlock], layers: list[int], num_classes: int = 10,
                 in_channels: int = 3, base_width: int = 64, stem: str = "small",
                 strides: list[int] | None = None):
        super().__init__()
        if stem == "small":
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, base_width, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(base_width),
                nn.ReLU(inplace=True),
            )
            in_channels = base_width
        elif stem == "imagenet":
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, base_width, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(base_width),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            in_channels = base_width
        else:
            raise ValueError(f"unknown stem: {stem}")

        if strides is None:
            strides = [2, 2, 2, 2] if stem == "small" else [1, 2, 2, 2]

        self.layer1, in_channels = _make_layer(in_channels, block, base_width, layers[0], strides[0])
        self.layer2, in_channels = _make_layer(in_channels, block, base_width * 2, layers[1], strides[1])
        self.layer3, in_channels = _make_layer(in_channels, block, base_width * 4, layers[2], strides[2])
        self.layer4, in_channels = _make_layer(in_channels, block, base_width * 8, layers[3], strides[3])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_width * 8, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def resnet18(num_classes: int = 10, **kwargs) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)


def resnet34(num_classes: int = 10, **kwargs) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)
