"""
Stubs
-----

.. autosummary::
    :toctree: toctree/utils/

    OptimizerStub

"""

from typing import Optional, Callable

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer


class OptimizerStub(Optimizer):
    """
    An Optimizer stub for trainers that update model weights in a gradient-free
    fashion.
    """

    def __init__(self):
        self.param_groups = []

    def step(self, closure: Optional[Callable[[], float]] = ...):
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        pass


class CriterionStub(nn.Module):
    def forward(self, *args, **kwargs):
        return torch.Tensor([0.0])
