import torch
import torch.nn as nn


class Flatten(nn.Module):
    """
    Flattens the input tensor, starting from the dimension :code:`1`.
    """

    def forward(self, x: torch.Tensor):
        return x.flatten(start_dim=1)


class Reshape(nn.Module):
    """
    Reshapes the input tensor to a given shape.

    Parameters
    ----------
    height, width : int
        The height and the width of the resulting tensor images.
    """
    def __init__(self, height: int, width: int = None):
        super().__init__()
        if width is None:
            width = height
        self.height = height
        self.width = width

    def forward(self, x: torch.Tensor):
        return x.view(x.shape[0], -1, self.height, self.width)
