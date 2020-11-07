import torch.nn as nn

from mighty.monitor.mutual_info.mutual_info import MutualInfo


class MutualInfoStub(MutualInfo):
    """
    Mutual Info stub that does nothing.
    """

    def __init__(self):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def register_layer(self, layer: nn.Module, name: str):
        pass

    def start_listening(self):
        pass

    def finish_listening(self):
        pass

    def force_update(self, model: nn.Module):
        pass

    def plot(self, viz):
        pass

    def _prepare_input(self):
        pass

    def _process_activations(self, layer_name: str, activations):
        pass

    def _save_mutual_info(self):
        pass
