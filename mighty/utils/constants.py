from pathlib import Path

MIGHTY_DIR = Path.home() / ".mighty"
DATA_DIR = MIGHTY_DIR / "data"
CHECKPOINTS_DIR = MIGHTY_DIR / "checkpoints"
DUMPS_DIR = DATA_DIR / "dumps"
PCA_DIR = DATA_DIR / "pca"
VISDOM_LOGS_DIR = MIGHTY_DIR / "visdom_logs"

BATCH_SIZE = 256
