import unittest

import psutil
import torch
import torch.nn as nn
from numpy.testing import assert_array_almost_equal
from torchvision.datasets import MNIST

from mighty.loss import *
from mighty.models import MLP
from mighty.trainer import *
from mighty.utils.common import set_seed
from mighty.utils.data import TransformDefault, DataLoader


class TrainerTestCase(unittest.TestCase):

    def setUp(self):
        set_seed(1)
        self.model = MLP(784, 64, 10)
        cpu_count = psutil.cpu_count(logical=False)
        self.data_loader = DataLoader(MNIST, eval_size=10000,
                                      transform=TransformDefault.mnist(),
                                      num_workers=cpu_count)

        self.optimizer = torch.optim.Adam(
            filter(lambda param: param.requires_grad, self.model.parameters()),
            lr=1e-3,
            weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer)

    def test_TrainerGrad(self):
        set_seed(2)
        trainer = TrainerGrad(self.model,
                              criterion=nn.CrossEntropyLoss(),
                              data_loader=self.data_loader,
                              optimizer=self.optimizer,
                              scheduler=self.scheduler)
        trainer.open_monitor(offline=True)
        loss_epochs = trainer.train(n_epochs=1, mutual_info_layers=0)
        assert_array_almost_equal(loss_epochs, [0.2287357747])

    def test_TrainerEmbedding(self):
        set_seed(3)
        trainer = TrainerEmbedding(self.model,
                                   criterion=TripletLoss(),
                                   data_loader=self.data_loader,
                                   optimizer=self.optimizer,
                                   scheduler=self.scheduler)
        trainer.open_monitor(offline=True)
        loss_epochs = trainer.train(n_epochs=1, mutual_info_layers=0)
        assert_array_almost_equal(loss_epochs, [0.04264699295])
