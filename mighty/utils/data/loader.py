import os

import torch
import torch.utils.data
import torchvision

from mighty.utils.constants import DATA_DIR, BATCH_SIZE


class DataLoader:
    def __init__(self, dataset_cls, normalize=None, batch_size=BATCH_SIZE):
        self.dataset_cls = dataset_cls
        self.normalize = normalize
        self.batch_size = batch_size

        transform = [torchvision.transforms.ToTensor()]
        if self.normalize is not None:
            transform.append(self.normalize)
        self.transform = torchvision.transforms.Compose(transform)

    @property
    def num_workers(self):
        return int(os.environ.get('LOADER_WORKERS', 4))

    def get(self, train=True) -> torch.utils.data.DataLoader:
        dataset = self.dataset_cls(DATA_DIR, train=train, download=True,
                                   transform=self.transform)
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=self.batch_size,
                                             shuffle=train,
                                             num_workers=self.num_workers)
        return loader

    @property
    def eval(self):
        dataset = self.dataset_cls(DATA_DIR, train=True, download=True,
                                   transform=self.transform)
        eval_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=False,
                                                  num_workers=self.num_workers)
        return eval_loader

    def __repr__(self):
        return f"{self.__class__.__name__}({self.dataset_cls.__name__}, " \
               f"normalize={self.normalize})"
