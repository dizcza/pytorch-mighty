from pathlib import Path
import os

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
DUMPS_DIR = DATA_DIR / "dumps"
PCA_DIR = DATA_DIR / "pca"
VISDOM_LOGS_DIR = ROOT_DIR / "visdom_logs"

BATCH_SIZE = 256
