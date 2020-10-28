"""
Data loader with simple API.

.. autosummary::
    :toctree: toctree/utils/

    DataLoader

"""

import math

import torch
import torch.utils.data
from torchvision.transforms import ToTensor
from tqdm import tqdm

from mighty.utils.constants import DATA_DIR, BATCH_SIZE
from mighty.utils.data.normalize import get_normalize_inverse


class DataLoader:
    """
    Data loader with simple API.

    Parameters
    ----------
    dataset_cls : type
        Dataset class.
    transform : object, optional
        Torchvision transform object that implements ``__call__`` method.
        Default: ToTensor()
    loader_cls : type, optional
        A batches loader class.
        Default: torch.utils.data.DataLoader
    batch_size : int, optional
        Batch size. Default: 256
    eval_size : int or None, optional
        Evaluation size in the minimum number of samples. If None, the length
        of the dataset is used.
        Default: None
    num_workers : int, optional
        The number of workers passed to `loader_cls`.
        Default: 0
    """

    def __init__(self, dataset_cls, transform=ToTensor(),
                 loader_cls=torch.utils.data.DataLoader,
                 batch_size=BATCH_SIZE, eval_size=None, num_workers=0):
        self.dataset_cls = dataset_cls
        self.loader_cls = loader_cls
        self.transform = transform
        self.batch_size = batch_size
        if eval_size is None:
            eval_size = float('inf')
        dataset = self.dataset_cls(DATA_DIR, train=True, download=True)
        eval_size = min(eval_size, len(dataset))
        self.eval_size = eval_size
        self.num_workers = num_workers
        self.normalize_inverse = get_normalize_inverse(self.transform)

        # hack to check if the dataset is with labels
        self.has_labels = False
        sample = self.sample()
        if isinstance(sample, (tuple, list)) and len(sample) > 1:
            labels = sample[1]
            self.has_labels = isinstance(labels, torch.Tensor) \
                              and labels.dtype is torch.long

    def get(self, train=True):
        """
        Returns a train or test loader.

        Parameters
        ----------
        train : bool, optional
            Train (True) or test (False) fold.
            Default: true

        Returns
        -------
        loader : torch.utils.data.DataLoader
            A data loader with batches.
        """
        dataset = self.dataset_cls(DATA_DIR, train=train, download=True,
                                   transform=self.transform)
        loader = self.loader_cls(dataset,
                                 batch_size=self.batch_size,
                                 shuffle=train,
                                 num_workers=self.num_workers)
        return loader

    def eval(self, description=None):
        """
        Returns a generator over train samples with no shuffling.

        The generator exits after producing at least :attr:`eval_size` samples.

        Parameters
        ----------
        description : str or None, optional
            Message description.
            Default: None

        Yields
        ------
        batch : torch.Tensor
            Eval batch, same as in train.
        """
        dataset = self.dataset_cls(DATA_DIR, train=True, download=True,
                                   transform=self.transform)
        eval_loader = self.loader_cls(dataset,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      num_workers=self.num_workers)

        n_batches = math.ceil(self.eval_size / self.batch_size)
        for batch_id, batch in tqdm(
                enumerate(iter(eval_loader)),
                desc=description,
                total=n_batches,
                disable=not description,
                leave=False):
            if batch_id >= n_batches:
                break
            yield batch

    def sample(self):
        """
        Returns the first batch from :meth:`DataLoader.eval`.

        No shuffling/sampling is performed.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            A tensor or a batch of tensors.

        """
        return next(iter(self.eval()))

    def __repr__(self):
        return f"{self.__class__.__name__}({self.dataset_cls.__name__}, " \
               f"has_labels={self.has_labels}, " \
               f"transform={self.transform}, batch_size={self.batch_size}, " \
               f"eval_size={self.eval_size}, " \
               f"num_workers={self.num_workers}), normalize_inverse=" \
               f"{repr(self.normalize_inverse)})"
