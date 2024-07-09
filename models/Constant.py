from torch import nn
import torch


class Constant(nn.Module):
    def __init__(self, constant_output_value):
        super().__init__()
        self.constant_output_value = nn.Parameter(torch.tensor([constant_output_value]))

    def forward(self, x):
        return self.constant_output_value.expand(x.size(0), 1)
