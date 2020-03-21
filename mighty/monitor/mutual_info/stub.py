import torch.nn as nn


class MutualInfoStub:
    # Stub to work with the Monitor

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def register(self, layer: nn.Module, name: str):
        pass

    def force_update(self, model: nn.Module):
        pass

    def plot(self, viz):
        pass
