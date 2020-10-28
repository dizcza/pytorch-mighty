"""
Convert a model to the train or test mode.

.. autosummary::
    :toctree: toctree/utils/

    ModelMode
    prepare_eval

"""

import torch.nn as nn


class ModelMode:
    """
    Stores the model state with its parameters to be restored later on.

    Parameters
    ----------
    mode : bool
        Original model mode extracted as ``model.training``.
    requires_grad : dict
        A dict with keys that match ``model.named_parameters()`` dict which
        store a boolean state of each model parameter.

    """
    def __init__(self, mode, requires_grad):
        self.mode = mode
        self.requires_grad = requires_grad

    def restore(self, model):
        """
        Restore the original state of the model and its parameters.

        Parameters
        ----------
        model : nn.Module
            A model that was used as the input to :func:`prepare_eval`
            function.

        """
        model.train(self.mode)
        for name, param in model.named_parameters():
            param.requires_grad_(self.requires_grad[name])


def prepare_eval(model):
    """
    Sets the model and its parameters to the eval state.

    Parameters
    ----------
    model : nn.Module
        An input model.

    Returns
    -------
    ModelMode
        A model mode state that can recover the original input state.

    """
    mode_saved = model.training
    requires_grad_saved = {}
    model.eval()
    for name, param in model.named_parameters():
        requires_grad_saved[name] = param.requires_grad
        param.requires_grad_(False)
    return ModelMode(mode=mode_saved, requires_grad=requires_grad_saved)
