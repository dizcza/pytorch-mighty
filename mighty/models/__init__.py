"""
Model starters.

.. autosummary::
    :toctree: toctree/models/

    MLP
    AutoencoderLinear
    SerializableModule

"""


from .autoencoder import AutoencoderLinear, AutoencoderOutput
from .flatten import Flatten, Reshape
from .mlp import MLP
from .serialize import SerializableModule
