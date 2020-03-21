import time
from collections import defaultdict
from functools import wraps

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data


def find_layers(model: nn.Module, layer_class):
    for name, layer in find_named_layers(model, layer_class=layer_class):
        yield layer


def find_named_layers(model: nn.Module, layer_class, name_prefix=''):
    for name, layer in model.named_children():
        yield from find_named_layers(layer, layer_class, name_prefix=f"{name_prefix}.{name}")
    if isinstance(model, layer_class):
        yield name_prefix.lstrip('.'), model


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clone_cpu(tensor: torch.Tensor) -> torch.Tensor:
    tensor_clone = tensor.cpu()
    if tensor_clone is tensor:
        tensor_clone = tensor_clone.clone()
    return tensor_clone


def timer_profile(func):
    """
    For debug purposes only.
    """
    func_duration = defaultdict(list)

    @wraps(func)
    def wrapped(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        elapsed = time.time() - start
        elapsed *= 1e3
        func_duration[func.__name__].append(elapsed)
        print(f"{func.__name__} {elapsed: .3f} "
              f"(mean: {np.mean(func_duration[func.__name__]): .3f}) ms")
        return res

    return wrapped
