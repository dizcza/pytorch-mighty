import torch
import torch.utils.data
from torchvision.datasets import MNIST
from torchvision.transforms import Normalize, Compose, ToTensor
from tqdm import tqdm

from mighty.monitor.var_online import VarianceOnlineBatch
from mighty.monitor.viz import VisdomMighty
from mighty.utils.common import input_from_batch
from mighty.utils.constants import DATA_DIR, BATCH_SIZE


class NormalizeInverse(Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv, inplace=False)


def get_normalize_inverse(transform):
    if isinstance(transform, Compose):
        for child in transform.transforms:
            norm_inv = get_normalize_inverse(child)
            if norm_inv is not None:
                return norm_inv
    elif isinstance(transform, Normalize):
        return NormalizeInverse(mean=transform.mean,
                                std=transform.std)
    return None


def dataset_mean_std(dataset_cls: type):
    """
    :param dataset_cls: class type of torch.utils.data.Dataset
    :return: samples' mean and std per channel, estimated from a training set
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


def visualize_mean_std(dataset_cls=MNIST):
    """
    Plots dataset mean and std, averaged across channels.
    Run as module: 'python -m monitor.var_online'.
    :param dataset_cls: class type of torch.utils.data.Dataset
    """
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
    visualize_mean_std(dataset_cls=MNIST)
