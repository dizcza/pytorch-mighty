import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor):
        return x.flatten(start_dim=1)
