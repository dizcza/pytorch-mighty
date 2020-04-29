import math

import torch
import torch.utils.data
from torchvision.transforms import ToTensor
from tqdm import tqdm

from mighty.utils.constants import DATA_DIR, BATCH_SIZE
from mighty.utils.data.normalize import get_normalize_inverse


class DataLoader:
    def __init__(self, dataset_cls,
                 transform=ToTensor(),
                 batch_size=BATCH_SIZE,
                 eval_size=None, num_workers=0):
        self.dataset_cls = dataset_cls
        self.transform = transform
        self.batch_size = batch_size
        if eval_size is None:
            eval_size = float('inf')
        self.eval_size = eval_size
        self.num_workers = num_workers
        self.normalize_inverse = get_normalize_inverse(self.transform)

        # hack to check if the dataset is with labels
        self.has_labels = False
        sample = self.sample()
        if isinstance(sample, (tuple, list)) and len(sample) > 1:
            labels = sample[1]
            self.has_labels = labels.dtype is torch.long

    def get(self, train=True) -> torch.utils.data.DataLoader:
        dataset = self.dataset_cls(DATA_DIR, train=train, download=True,
                                   transform=self.transform)
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=self.batch_size,
                                             shuffle=train,
                                             num_workers=self.num_workers)
        return loader

    def eval(self, verbose=False) -> torch.utils.data.DataLoader:
        dataset = self.dataset_cls(DATA_DIR, train=True, download=True,
                                   transform=self.transform)
        eval_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=False,
                                                  num_workers=self.num_workers)

        n_samples_take = min(self.eval_size, len(dataset))
        n_batches = math.ceil(n_samples_take / self.batch_size)
        for batch_id, batch in tqdm(
                enumerate(iter(eval_loader)),
                desc="Full forward pass (eval)",
                total=n_batches,
                disable=not verbose,
                leave=False):
            if batch_id >= n_batches:
                break
            yield batch

    def sample(self):
        # always returns the first sample, no shuffling!
        return next(iter(self.eval()))

    def __repr__(self):
        return f"{self.__class__.__name__}({self.dataset_cls.__name__}, " \
               f"has_labels={self.has_labels}, "\
               f"transform={self.transform}, batch_size={self.batch_size}, " \
               f"eval_size={self.eval_size}, " \
               f"num_workers={self.num_workers}), normalize_inverse=" \
               f"{repr(self.normalize_inverse)})"
