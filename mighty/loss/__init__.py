"""
Sample pairs and triplets out of conventional `(outputs, labels)` batches.

.. autosummary::
    :toctree: toctree/loss/

    ContrastiveLossSampler
    TripletLossSampler


Loss functions not included in PyTorch.

.. autosummary::
    :toctree: toctree/loss/

    TripletCosineLoss

"""

from .contrastive import PairLossSampler, ContrastiveLossSampler
from .triplet import TripletLossSampler, TripletCosineLoss
from .penalty import LossPenalty
