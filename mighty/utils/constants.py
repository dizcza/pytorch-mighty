"""
Constants.

.. data:: BATCH_SIZE
    :type: int
    :value: 256

    Default batch size.

.. data:: MIGHTY_DIR
    :type: str

    Pytorch-mighty root directory. Defaults to ``${HOME}/.mighty``.

.. data:: DATA_DIR
    :type: str

    A directory with downloaded datasets. Defaults to ``${HOME}/.mighty/data``.

.. data:: CHECKPOINTS_DIR
    :type: str

    A directory with checkpoints. Defaults to ``{HOME}/.mighty/checkpoints``.

.. data:: DUMPS_DIR
    :type: str

    A directory with dumped layer activations. Defaults to
    ``{HOME}/.mighty/dumps``.

.. data:: PCA_DIR
    :type: str

    A directory with PCA pretrained instances that transforms layer activations
    to lower dimensions to be able to estimate Mutual Information. Defaults
    to ``{HOME}/.mighty/pca``.

.. data:: VISDOM_LOGS_DIR
    :type: str

    A directory with visdom logs, when a visdom server works in offline mode.
    Defaults to ``{HOME}/.mighty/visdom_logs``.
"""

from pathlib import Path

MIGHTY_DIR = Path.home() / ".mighty"
DATA_DIR = MIGHTY_DIR / "data"
CHECKPOINTS_DIR = MIGHTY_DIR / "checkpoints"
DUMPS_DIR = DATA_DIR / "dumps"
PCA_DIR = DATA_DIR / "pca"
VISDOM_LOGS_DIR = MIGHTY_DIR / "visdom_logs"

BATCH_SIZE = 256
