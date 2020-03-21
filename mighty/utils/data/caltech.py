import os
import random
import shutil
import tarfile
from urllib.request import urlretrieve

import torchvision
from torchvision.datasets.utils import check_integrity
from tqdm import tqdm

from mighty.utils.constants import DATA_DIR

CALTECH_RAW = DATA_DIR / "Caltech_raw"
CALTECH_256 = DATA_DIR / "Caltech256"
CALTECH_10 = DATA_DIR / "Caltech10"

CALTECH_URL = "http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar"
TAR_FILEPATH = CALTECH_RAW / CALTECH_URL.split('/')[-1]


class TqdmWget(tqdm):
    """Alternative Class-based version of the above.
    Provides `update_to(n)` which uses `tqdm.update(delta_n)`.
    Inspired by [twine#242](https://github.com/pypa/twine/pull/242),
    [here](https://github.com/pypa/twine/commit/42e55e06).
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def download():
    os.makedirs(CALTECH_RAW, exist_ok=True)
    if check_integrity(TAR_FILEPATH, md5="67b4f42ca05d46448c6bb8ecd2220f6d"):
        print(f"Using downloaded and verified {TAR_FILEPATH}")
    else:
        with TqdmWget(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                      desc=f"Downloading {CALTECH_URL}") as t:
            urlretrieve(CALTECH_URL, filename=TAR_FILEPATH,
                        reporthook=t.update_to, data=None)
    print(f"Extracting {TAR_FILEPATH}")
    with tarfile.open(TAR_FILEPATH) as tar:
        tar.extractall(path=CALTECH_RAW)


def move_files(filepaths, folder_to):
    os.makedirs(folder_to, exist_ok=True)
    for filepath in filepaths:
        filepath.rename(folder_to / filepath.name)


def split_train_test(train_part=0.8):
    # we don't need noise/background class
    caltech_root = CALTECH_RAW / TAR_FILEPATH.stem
    shutil.rmtree(caltech_root / "257.clutter", ignore_errors=True)
    for category in caltech_root.iterdir():
        images = list(filter(lambda filepath: filepath.suffix == '.jpg',
                             category.iterdir()))
        random.shuffle(images)
        n_train = int(train_part * len(images))
        images_train = images[:n_train]
        images_test = images[n_train:]
        move_files(images_train, CALTECH_256 / "train" / category.name)
        move_files(images_test, CALTECH_256 / "test" / category.name)
    print("Split Caltech dataset.")


class Caltech256(torchvision.datasets.ImageFolder):
    def __init__(self, train=True, root=CALTECH_256):
        self.prepare()
        transforms = []
        if train:
            fold = "train"
            transforms.append(torchvision.transforms.RandomHorizontalFlip())
        else:
            fold = "test"
        transforms.extend([
            torchvision.transforms.Resize(size=(224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        super().__init__(root=root / fold,
                         transform=torchvision.transforms.Compose(transforms))

    def prepare(self):
        if not CALTECH_256.exists():
            download()
            split_train_test()


class Caltech10(Caltech256):
    classes = (
        "169.radio-telescope", "009.bear", "010.beer-mug", "024.butterfly",
        "025.cactus", "028.camel", "030.canoe", "055.dice", "056.dog",
        "060.duck"
    )

    def __init__(self, train=True):
        super().__init__(train=train, root=CALTECH_10)

    def _prepare_category(self, category):
        for fold in ("train", "test"):
            shutil.copytree(CALTECH_256 / fold / category,
                            CALTECH_10 / fold / category)

    def prepare(self):
        super().prepare()
        if not CALTECH_10.exists():
            for category in self.classes:
                self._prepare_category(category)
