import torch
import torch.utils.data
from torchvision import transforms, datasets
from tqdm import tqdm

from mighty.monitor.var_online import VarianceOnline
from mighty.monitor.viz import VisdomMighty
from mighty.utils.constants import DATA_DIR


class _NormalizeTensor:
    def __init__(self, mean, std):
        # TODO replace by torchvision.Normalize
        mean = self.as3d(mean)
        std = self.as3d(std)
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)

    @staticmethod
    def as3d(tensor):
        """
        :param tensor: (C,) or (C, H, W) Tensor
        :return: (C, 1, 1) [unsqueezed] or (C, H, W) [original] Tensor
        """
        for dim_extra in range(3 - tensor.ndimension()):
            tensor = tensor.unsqueeze(dim=-1)
        return tensor

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        mean = torch.as_tensor(self.mean, dtype=torch.float32, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=torch.float32, device=tensor.device)
        tensor = tensor.clone()
        tensor.sub_(mean).div_(std)
        return tensor


class NormalizeFromDataset(_NormalizeTensor):
    """
    Normalize dataset by subtracting channel-wise and pixel-wise mean and dividing by STD.
    Mean and STD are estimated from a training set only.
    """

    def __init__(self, dataset_cls: type):
        mean, std = dataset_mean_std(dataset_cls=dataset_cls)
        std += 1e-6
        super().__init__(mean=mean, std=std)


class NormalizeInverse(_NormalizeTensor):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


def get_normalize_inverse(normalize_transform):
    if isinstance(normalize_transform, (transforms.Normalize, _NormalizeTensor)):
        return NormalizeInverse(mean=normalize_transform.mean,
                                std=normalize_transform.std)
    return None


def dataset_mean_std(dataset_cls: type):
    """
    :param dataset_cls: class type of torch.utils.data.Dataset
    :return: samples' mean and std per channel, estimated from a training set
    """
    mean_std_file = (DATA_DIR / "mean_std" / dataset_cls.__name__).with_suffix('.pt')
    if not mean_std_file.exists():
        dataset = dataset_cls(DATA_DIR, train=True, download=True,
                              transform=transforms.ToTensor())
        loader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                             shuffle=False, num_workers=4)
        var_online = VarianceOnline()
        for images, labels in tqdm(loader,
                desc=f"{dataset_cls.__name__}: running online mean, std"):
            for image in images:
                var_online.update(new_tensor=image)
        mean, std = var_online.get_mean_std()
        mean_std_file.parent.mkdir(exist_ok=True, parents=True)
        with open(mean_std_file, 'wb') as f:
            torch.save((mean, std), f)
    with open(mean_std_file, 'rb') as f:
        mean, std = torch.load(f)
    return mean, std


def visualize_mean_std(dataset_cls=datasets.MNIST):
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
    visualize_mean_std(dataset_cls=datasets.MNIST)
