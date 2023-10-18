"""
Model starters.

.. autosummary::
    :toctree: toctree/models/

    MLP
    MLPDropout
    AutoencoderLinear
    SerializableModule

"""


from .autoencoder import AutoencoderLinear, AutoencoderOutput
from .flatten import Flatten, Reshape
from .mlp import MLP, MLPDropout
from .serialize import SerializableModule
