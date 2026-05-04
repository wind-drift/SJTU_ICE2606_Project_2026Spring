"""CNN classifier for log-Mel digit spectrograms."""

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


class TinyKeywordCNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.30) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            ConvBlock(32, 64),
            nn.MaxPool2d(kernel_size=2),
            ConvBlock(64, 128),
            nn.MaxPool2d(kernel_size=2),
            ConvBlock(128, 192, dilation=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(192, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
