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
