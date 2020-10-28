"""
An Optimizer stub for trainers that update model weights in a gradient-free
fashion.

.. autosummary::
    :toctree: toctree/utils/

    OptimizerStub

"""

from typing import Optional, Callable

from torch.optim.optimizer import Optimizer


class OptimizerStub(Optimizer):

    def __init__(self):
        self.param_groups = []

    def step(self, closure: Optional[Callable[[], float]] = ...):
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        pass
