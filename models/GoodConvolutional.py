import torch
from torch import nn


class GoodConvolutional(nn.Module):
    def __init__(
        self,
        img_x_size,
        img_y_size,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(img_y_size * img_x_size, 12),
            nn.ReLU(),
            nn.Linear(12, 1),
        )

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = self.layers(x)
        return x
