"""
Layer hooks.

.. autosummary::
    :toctree: toctree/utils/

    get_layers_ordered

"""

import pickle

import shutil
import torch
import torch.nn as nn
from pathlib import Path

from mighty.utils.constants import DUMPS_DIR


def get_layers_ordered(model: nn.Module, input_sample: torch.Tensor,
                       ignore_layers=(nn.Sequential,), ignore_children=()):
    """
    Returns a list of ordered layers of the input model.

    Parameters
    ----------
    model : nn.Module
        An input model.
    input_sample : torch.Tensor
        A sample tensor to be used with the model.
    ignore_layers : tuple of type
        A tuple of model classes not to include in the final result.
        Default: (nn.Sequential,)
    ignore_children : tuple of type
        A tuple of model classes to skip entering their children.
        Default: ()

    Returns
    -------
    layers_ordered : list of torch.Tensor
        An ordered list of layers of the input model. Note that some modules
        might be added in the list more than once.

    """
    ignore_layers = ignore_layers + (type(model),)
    hooks = []
    layers_ordered = []

    def register_hooks(a_model: nn.Module):
        children = tuple(a_model.children())
        if any(children) and not isinstance(a_model, ignore_children):
            for layer in children:
                register_hooks(layer)
        if not isinstance(a_model, ignore_layers):
            handle = a_model.register_forward_pre_hook(append_layer)
            hooks.append(handle)

    def append_layer(layer, tensor_input):
        layers_ordered.append(layer)

    register_hooks(model)

    if input_sample.ndim == 3:
        input_sample = input_sample.unsqueeze(dim=0)  # batch of 1 sample
    if torch.cuda.is_available():
        input_sample = input_sample.cuda()
    with torch.no_grad():
        model(input_sample)

    for handle in hooks:
        handle.remove()

    return layers_ordered


class DumpActivationsHook:
    """
    A use-case for :func:`get_layers_ordered`.
    """
    def __init__(self, model: nn.Module,
                 inspect_layers=(nn.Linear, nn.Conv2d),
                 dumps_dir=DUMPS_DIR):
        self.hooks = []
        self.layer_to_name = {}
        self.inspect_layers = inspect_layers
        self.dumps_dir = Path(dumps_dir) / model._get_name()
        shutil.rmtree(self.dumps_dir, ignore_errors=True)
        self.dumps_dir.mkdir(parents=True)
        self.register_hooks(model)
        print(f"Dumping activations from {self.layer_to_name.values()} layers "
              f"to {self.dumps_dir}.")

    def register_hooks(self, model: nn.Module, prefix=''):
        children = tuple(model.named_children())
        if any(children):
            for name, layer in children:
                self.register_hooks(layer, prefix=f"{prefix}.{name}")
        elif isinstance(model, self.inspect_layers):
            self.layer_to_name[model] = prefix.lstrip('.')
            handle = model.register_forward_hook(self.dump_activations)
            self.hooks.append(handle)

    def dump_activations(self, layer, tensor_input, tensor_output):
        layer_name = self.layer_to_name[layer]
        layer_path = self.dumps_dir / layer_name
        activations_input_path = f"{layer_path}_inp.pkl"
        activations_output_path = f"{layer_path}_out.pkl"
        if isinstance(tensor_input, tuple):
            assert len(tensor_input) == 1, "Expected only 1 input tensor"
            tensor_input = tensor_input[0]
        with open(activations_input_path, 'ab') as f:
            pickle.dump(tensor_input.detach().cpu(), f)
        with open(activations_output_path, 'ab') as f:
            pickle.dump(tensor_output.detach().cpu(), f)

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
