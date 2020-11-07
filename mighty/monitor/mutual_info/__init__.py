"""
Mutual Information Estimates
----------------------------

.. autosummary::
    :toctree: toctree/monitor

    MutualInfoNeuralEstimation
    MutualInfoGCMI
    MutualInfoIDTxl
    MutualInfoNPEET
    MutualInfoKMeans
    MutualInfoStub
"""


from .gcmi import MutualInfoGCMI
from .kmeans import MutualInfoKMeans
from .neural_estimation import MutualInfoNeuralEstimation
from .npeet import MutualInfoNPEET
from .stub import MutualInfoStub

try:
    from .idtxl import MutualInfoIDTxl
except ImportError:
    # idtxl is not installed
    pass
