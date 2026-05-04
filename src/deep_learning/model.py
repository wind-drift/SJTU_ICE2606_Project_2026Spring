"""CNN classifier with squeeze-excitation for log-Mel digit spectrograms."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1) -> None:
        padding = dilation * (kernel_size // 2)
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class TinyKeywordCNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.30) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 16),
            SEBlock(16),
            nn.MaxPool2d(kernel_size=2),
            ConvBlock(16, 32),
            SEBlock(32),
            nn.MaxPool2d(kernel_size=2),
            ConvBlock(32, 64),
            SEBlock(64),
            nn.MaxPool2d(kernel_size=2),
            ConvBlock(64, 128),
            SEBlock(128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
