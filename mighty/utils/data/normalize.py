"""
Inverse normalizations of data transforms.

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
from torchvision.datasets import MNIST
from torchvision.transforms import Normalize, Compose, ToTensor
from tqdm import tqdm

from mighty.monitor.var_online import VarianceOnlineBatch
from mighty.monitor.viz import VisdomMighty
from mighty.utils.common import input_from_batch
from mighty.utils.constants import DATA_DIR, BATCH_SIZE


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


def dataset_mean_std(dataset_cls: type):
    """
    Estimates dataset mean and std.

    Parameters
    ----------
    dataset_cls : type
        A dataset class.

    Returns
    -------
    mean, std : (C, H, W) torch.Tensor
        Channel- and pixel-wise dataset mean and std, estimated over all
        samples.
    """
    mean_std_file = (DATA_DIR / "mean_std" / dataset_cls.__name__
                     ).with_suffix('.pt')
    if not mean_std_file.exists():
        dataset = dataset_cls(DATA_DIR, train=True, download=True,
                              transform=ToTensor())
        loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                             shuffle=False)
        var_online = VarianceOnlineBatch()
        for batch in tqdm(
                loader,
                desc=f"{dataset_cls.__name__}: running online mean, std"):
            input = input_from_batch(batch)
            var_online.update(new_tensor=input)
        mean, std = var_online.get_mean_std()
        mean_std_file.parent.mkdir(exist_ok=True, parents=True)
        with open(mean_std_file, 'wb') as f:
            torch.save((mean, std), f)
    with open(mean_std_file, 'rb') as f:
        mean, std = torch.load(f)
    return mean, std


def plot_dataset_mean_std(viz=None, dataset_cls=MNIST):
    """
    Plots dataset mean and std, averaged across channels.

    Parameters
    ----------
    viz : Visdom
        A Visdom instance.
    dataset_cls : type, optional
        A dataset class to plot its mean and std.
    """
    if viz is None:
        viz = VisdomMighty(env="main")
    mean, std = dataset_mean_std(dataset_cls=dataset_cls)
    viz.heatmap(mean.mean(dim=0), win=f'{dataset_cls.__name__} mean',
                opts=dict(
                    title=f'{dataset_cls.__name__} Mean',
                ))
    viz.heatmap(std.mean(dim=0), win=f'{dataset_cls.__name__} std', opts=dict(
        title=f'{dataset_cls.__name__} STD',
    ))


if __name__ == '__main__':
    plot_dataset_mean_std(dataset_cls=MNIST)
