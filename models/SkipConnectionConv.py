import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class SkipConnectionConv(nn.Module):
    def __init__(
        self,
        img_x_size,
        img_y_size,
    ):
        super().__init__()
        self.block1 = ConvBlock(1, 32)
        self.block2 = ConvBlock(32, 1)
        self.block3 = ConvBlock(1, 32)
        self.block4 = ConvBlock(32, 1)
        self.end_perceptron = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_y_size * img_x_size, 12),
            nn.ReLU(),
            nn.Linear(12, 1),
        )

    def forward(self, x):
        org = torch.unsqueeze(x, dim=1)  # (batch, 1, x, y)
        x = self.block1(org)  # (batch, 32, x, y)
        org2 = self.block2(x) + org  # (batch, 1, x, y)
        x = self.block3(org2)  # (batch, 32, x, y)
        x = self.block4(x) + org2  # (batch, 1, x, y)
        x = self.end_perceptron(x)
        return x
