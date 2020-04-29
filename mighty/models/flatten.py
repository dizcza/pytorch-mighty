import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor):
        return x.flatten(start_dim=1)


class Reshape(nn.Module):
    def __init__(self, height: int, width: int = None):
        super().__init__()
        if width is None:
            width = height
        self.height = height
        self.width = width

    def forward(self, x: torch.Tensor):
        return x.view(x.shape[0], -1, self.height, self.width)
