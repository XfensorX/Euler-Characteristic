import torch
from torch import nn


class ConvolutionalNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        img_x_size,
        img_y_size,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )

        self.fc1 = nn.Linear(32 * img_y_size * img_x_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.flatten = nn.Flatten()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x
