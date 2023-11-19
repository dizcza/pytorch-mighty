"""
Finding a layer
---------------

.. autosummary::
    :toctree: toctree/utils/

    find_layers
    find_named_layers


Working with batches
--------------------

.. autosummary::
    :toctree: toctree/utils/

    input_from_batch
    batch_to_cuda
    clone_cpu


Miscellaneous
-------------

.. autosummary::
    :toctree: toctree/utils/

    set_seed

"""

import torch
import torch.nn as nn
import torch.utils.data


__all__ = [
    "batch_to_cuda",
    "input_from_batch",
    "find_layers",
    "find_named_layers",
    "find_param_by_name",
    "set_seed",
    "clone_cpu"
]


def batch_to_cuda(batch):
    """
    Transfers the batch to CUDA.

    This function is used when you don't know what is the structure of a batch.

    Parameters
    ----------
    batch : torch.Tensor or iterable
        A tensor or an iterable of tensors.

    Returns
    -------
    batch : torch.Tensor or tuple
        Transferred to CUDA batch.

    """
    if not torch.cuda.is_available():
        return batch
    if isinstance(batch, torch.Tensor):
        batch = batch.cuda()
    else:
        # iterable
        batch = tuple(map(batch_to_cuda, batch))
    return batch


def input_from_batch(batch):
    """
    If the input is a tensor, return it. Otherwise, return the first element.

    Parameters
    ----------
    batch : torch.Tensor or tuple of torch.Tensor
        Input batch.

    Returns
    -------
    torch.Tensor
        A tensor, used as the input to a model.

    """
    if isinstance(batch, torch.Tensor):
        # unsupervised learning, no labels
        return batch
    # iterable
    return batch[0]


def find_layers(model, layer_class):
    """
    Find all layers of type `layer_class` in the input model and yield them
    as a generator.

    Parameters
    ----------
    model : nn.Module
        A model.
    layer_class : type
        A layer class to look for.

    Yields
    ------
    nn.Module
        A children layer of instance `layer_class` found in the input model.

    """
    for name, layer in find_named_layers(model, layer_class=layer_class):
        yield layer


def find_named_layers(model: nn.Module, layer_class, name_prefix=''):
    """
    Find all layers of type `layer_class` in the input model and yield them
    as ``(name, layer)`` tuples.

    Parameters
    ----------
    model : nn.Module
        A model.
    layer_class : type or tuple of type
        A layer class to look for.
    name_prefix : str, optional
        A name prefix to add to the final result.

    Yields
    ------
    str
        The name of a children layer.
    nn.Module
        A children layer of instance `layer_class` found in the input model.

    """
    for name, layer in model.named_children():
        yield from find_named_layers(layer, layer_class,
                                     name_prefix=f"{name_prefix}.{name}")
    if isinstance(model, layer_class):
        yield name_prefix.lstrip('.'), model


def find_param_by_name(module: nn.Module, name: str):
    for _name, param in module.named_parameters():
        if _name == name:
            return param
    return None


def set_seed(seed: int):
    """
    Sets the global seed for PyTorch, Numpy, and built-in `random` function.

    Parameters
    ----------
    seed : int
        The global seed.

    """
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clone_cpu(tensor):
    """
    Clones a tensor, followed by to CPU transfer.

    If the input tensor is bounded to GPU, no extra copy is made.

    Parameters
    ----------
    tensor : torch.Tensor
        An input tensor.

    Returns
    -------
    torch.Tensor
        A CPU-bounded tensor copy.

    """
    tensor_clone = tensor.cpu()
    if tensor_clone is tensor:
        tensor_clone = tensor_clone.clone()
    return tensor_clone
