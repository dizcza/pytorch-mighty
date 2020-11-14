"""
Main trainers
-------------

.. autosummary::
    :toctree: toctree/trainer

    Trainer
    TrainerGrad
    TrainerEmbedding
    TrainerAutoencoder


Idle test "trainer"
-------------------

.. autosummary::
    :toctree: toctree/trainer

    Test


Interpretable mask trainers
---------------------------

.. autosummary::
    :toctree: toctree/trainer

    MaskTrainer
"""

from .autoencoder import TrainerAutoencoder
from .embedding import TrainerEmbedding
from .gradient import TrainerGrad
from .mask import MaskTrainer
from .test import Test
from .trainer import Trainer
