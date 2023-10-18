"""
Inverse normalizations of data transforms
-----------------------------------------

.. autosummary::
    :toctree: toctree/utils/

    NormalizeInverse
    get_normalize_inverse
    get_normalize
    dataset_mean_std

"""


import torch
import torch.utils.data
import torchvision.transforms.functional as F
from torchvision.transforms import Normalize, Compose, ToTensor
from tqdm import tqdm

from mighty.utils.common import input_from_batch
from mighty.utils.constants import DATA_DIR, BATCH_SIZE
from mighty.utils.var_online import VarianceOnlineBatch


class NormalizeInverse(Normalize):
    """
    Undoes the normalization and returns reconstructed images in the input
    domain.

    Parameters
    ----------
    mean, std : array-like
        Sequence of means and standard deviations for each channel.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv, inplace=False)

    def __call__(self, tensor):
        # (B, C, H, W) tensor
        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        return F.normalize(tensor, mean=mean, std=std, inplace=False)


def get_normalize_inverse(transform):
    """
    Traverses the input `transform`, finds a class of ``Normalize``, and
    returns ``NormalizeInverse``.

    Parameters
    ----------
    transform:
        Torchvision transform.

    Returns
    -------
    NormalizeInverse
        NormalizeInverse object to undo the normalization in the input
        `transform` or None, if ``Normalize`` instance is not found.
    """
    normalize = get_normalize(transform)
    if normalize:
        return NormalizeInverse(mean=normalize.mean,
                                std=normalize.std)
    return None


def get_normalize(transform, normalize_cls=Normalize):
    """
    Traverses the input `transform` and finds an instance of `normalize_cls`.

    Parameters
    ----------
    transform:
        Torchvision transform
    normalize_cls : type, optional
        A class to look for in the input transform.
        Default: Normalize

    Returns
    -------
    Normalize
        Found normalize instance or None.
    """
    if isinstance(transform, Compose):
        for child in transform.transforms:
            norm_inv = get_normalize(child, normalize_cls)
            if norm_inv is not None:
                return norm_inv
    elif isinstance(transform, Normalize):
        return transform
    return None


def dataset_mean_std(dataset_cls: type, cache=True, verbose=False):
    """
    Estimates dataset mean and std.

    Parameters
    ----------
    dataset_cls : type
        A dataset class.
    cache : bool, optional
        Compute once (True) or every time (False).
        Default: True
    verbose : bool, optional
        Verbosity flag.
        Default: False

    Returns
    -------
    mean, std : (C, H, W) torch.Tensor
        Channel- and pixel-wise dataset mean and std, estimated over all
        samples.
    """
    mean_std_file = (DATA_DIR / "mean_std" / dataset_cls.__name__
                     ).with_suffix('.pt')
    if not cache or not mean_std_file.exists():
        dataset = dataset_cls(DATA_DIR, train=True, download=True,
                              transform=ToTensor())
        loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                             shuffle=False)
        var_online = VarianceOnlineBatch()
        for batch in tqdm(
                loader,
                desc=f"{dataset_cls.__name__}: running online mean, std",
                disable=not verbose):
            input = input_from_batch(batch)
            var_online.update(input)
        mean, std = var_online.get_mean_std()
        mean_std_file.parent.mkdir(exist_ok=True, parents=True)
        with open(mean_std_file, 'wb') as f:
            torch.save((mean, std), f)
    with open(mean_std_file, 'rb') as f:
        mean, std = torch.load(f)
    return mean, std
