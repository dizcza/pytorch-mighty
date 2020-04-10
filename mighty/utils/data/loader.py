import math

import torch
import torch.utils.data
import torchvision
from tqdm import tqdm

from mighty.utils.constants import DATA_DIR, BATCH_SIZE
from mighty.utils.data import get_normalize_inverse


class DataLoader:
    def __init__(self, dataset_cls, normalize=None, batch_size=BATCH_SIZE,
                 eval_size=None, num_workers=0):
        self.dataset_cls = dataset_cls
        self.normalize = normalize
        self.batch_size = batch_size
        if eval_size is None:
            eval_size = float('inf')
        self.eval_size = eval_size
        self.num_workers = num_workers
        self.normalize_inverse = get_normalize_inverse(self.normalize)

        transform = [torchvision.transforms.ToTensor()]
        if self.normalize is not None:
            transform.append(self.normalize)
        self.transform = torchvision.transforms.Compose(transform)

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
        return next(iter(self.eval()))

    def __repr__(self):
        return f"{self.__class__.__name__}({self.dataset_cls.__name__}, " \
               f"normalize={self.normalize}, batch_size={self.batch_size}, " \
               f"eval_size={self.eval_size}, num_workers={self.num_workers})"
