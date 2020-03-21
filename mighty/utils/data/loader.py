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

    @property
    def num_workers(self):
        return int(os.environ.get('LOADER_WORKERS', 4))

    def get(self, train=True) -> torch.utils.data.DataLoader:
        transform = [torchvision.transforms.ToTensor()]
        if self.normalize is not None:
            transform.append(self.normalize)
        transform = torchvision.transforms.Compose(transform)
        dataset = self.dataset_cls(DATA_DIR, train=train, download=True,
                                   transform=transform)
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=self.batch_size,
                                             shuffle=train,
                                             num_workers=self.num_workers)
        return loader

    @property
    def eval(self):
        train_loader = self.get(train=True)
        eval_loader = torch.utils.data.DataLoader(dataset=train_loader.dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=False,
                                                  num_workers=self.num_workers)
        return eval_loader
