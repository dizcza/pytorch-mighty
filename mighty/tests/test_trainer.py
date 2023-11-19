import os
import unittest

import numpy as np
import torch
import torch.nn as nn
from numpy.testing import assert_array_almost_equal, assert_array_less
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Resize, Normalize, Compose

from mighty.loss import *
from mighty.models import MLP, AutoencoderLinear
from mighty.monitor.accuracy import AccuracyEmbedding
from mighty.monitor.monitor import MonitorLevel
from mighty.monitor.mutual_info import MutualInfoNPEET
from mighty.trainer import *
from mighty.utils.common import set_seed
from mighty.utils.data import DataLoader


class MNIST_Short(MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = self.data[:1000]
        self.targets = self.targets[:1000]


class TrainerTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.environ['VISDOM_SERVER'] = "http://85.217.171.57"
        os.environ['VISDOM_PORT'] = '8096'

    def setUp(self):
        set_seed(1)
        self.model = MLP(196, 64, 10)  # input 14x14
        transform = Compose([Resize(size=(14, 14)), ToTensor(),
                             Normalize(mean=(0.1307,), std=(0.3081,))])
        self.data_loader = DataLoader(MNIST_Short,
                                      batch_size=128,
                                      transform=transform)

        self.optimizer = torch.optim.Adam(
            filter(lambda param: param.requires_grad, self.model.parameters()),
            lr=1e-3,
            weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer)

    def test_TrainerGrad_MutualInfo(self):
        set_seed(2)
        mi_history = []

        class MutualInfoNPEET_Test(MutualInfoNPEET):
            def plot(self_mi, viz):
                mi_history.append(self_mi.information['mlp.2'])
                super().plot(viz)

        data_loader = DataLoader(MNIST, eval_size=1_000,
                                 transform=self.data_loader.transform)
        trainer = TrainerGrad(self.model,
                              criterion=nn.CrossEntropyLoss(),
                              data_loader=data_loader,
                              optimizer=self.optimizer,
                              mutual_info=MutualInfoNPEET_Test(
                                  data_loader=data_loader, pca_size=None),
                              scheduler=self.scheduler)
        trainer.env_name = "pytorch-mighty tests"
        trainer.monitor.advanced_monitoring(level=MonitorLevel.FULL)
        trainer.timer.batches_in_epoch = 5
        loss_epochs = trainer.train(n_epochs=1, mutual_info_layers=1)
        assert_array_almost_equal(loss_epochs, [0.35], decimal=2)
        mi_history = np.vstack(mi_history)
        assert_array_less(0, mi_history)
        x_last, y_last = mi_history[-1]

        # The model is well trained, however the estimate size is small.
        # In theory, I(.;Y) <= I(.;X) must be satisfied.
        self.assertGreater(y_last, 2.5)
        self.assertGreater(x_last, 1.3)
        self.assertLessEqual(x_last, y_last)

        trainer.save()
        trainer.restore()

    def test_TrainerEmbedding(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        set_seed(3)
        criterion = TripletLossSampler(nn.TripletMarginLoss())
        trainer = TrainerEmbedding(self.model,
                                   criterion=criterion,
                                   data_loader=self.data_loader,
                                   optimizer=self.optimizer,
                                   scheduler=self.scheduler)
        trainer.env_name = "pytorch-mighty tests"
        loss_epochs = trainer.train(n_epochs=1, mutual_info_layers=0)
        assert_array_almost_equal(loss_epochs, [0.58], decimal=1)
        self.assertEqual(trainer.timer.n_epochs, 1)
        self.assertEqual(trainer.timer.epoch, 1)

    def test_TrainerEmbedding_cached(self):
        set_seed(3)
        criterion = TripletLossSampler(nn.TripletMarginLoss())
        trainer = TrainerEmbedding(self.model,
                                   criterion=criterion,
                                   data_loader=self.data_loader,
                                   optimizer=self.optimizer,
                                   scheduler=self.scheduler,
                                   accuracy_measure=AccuracyEmbedding(
                                       cache=True))
        trainer.env_name = "pytorch-mighty tests"
        trainer.open_monitor(offline=True)

        # TripletLoss is not deterministic; fix the seed
        set_seed(4)
        loss_cached = trainer.full_forward_pass(train=True)
        accuracy_cached = trainer.update_accuracy(train=True)

        trainer.accuracy_measure.cache = False
        trainer.accuracy_measure.reset()

        set_seed(4)
        loss = trainer.full_forward_pass(train=True)
        accuracy = trainer.update_accuracy(train=True)

        self.assertAlmostEqual(loss_cached.item(), loss.item())
        self.assertAlmostEqual(accuracy_cached, accuracy)

    def test_TrainerAutoencoder(self):
        set_seed(4)
        model = AutoencoderLinear(196, 64)
        trainer = TrainerAutoencoder(model,
                                     criterion=nn.BCEWithLogitsLoss(),
                                     data_loader=self.data_loader,
                                     optimizer=self.optimizer,
                                     scheduler=self.scheduler)
        trainer.env_name = "pytorch-mighty tests"
        loss_epochs = trainer.train(n_epochs=1, mutual_info_layers=0)
        assert_array_almost_equal(loss_epochs, [0.705], decimal=3)

    @unittest.skip("Accuracy mismatch, fix it!")
    def test_TrainerAutoencoder_cached(self):
        set_seed(3)
        model = AutoencoderLinear(196, 64)
        trainer = TrainerAutoencoder(model,
                                     criterion=nn.BCEWithLogitsLoss(),
                                     data_loader=self.data_loader,
                                     optimizer=self.optimizer,
                                     scheduler=self.scheduler,
                                     accuracy_measure=AccuracyEmbedding(
                                         cache=True))
        trainer.open_monitor(offline=True)

        # TripletLoss is not deterministic; fix the seed
        set_seed(4)
        loss_cached = trainer.full_forward_pass(train=True)
        accuracy_cached = trainer.update_accuracy(train=True)

        trainer.accuracy_measure.cache = False
        trainer.accuracy_measure.reset()

        set_seed(4)
        loss = trainer.full_forward_pass(train=True)
        accuracy = trainer.update_accuracy(train=True)

        self.assertAlmostEqual(loss_cached.item(), loss.item())
        self.assertAlmostEqual(accuracy_cached, accuracy)


if __name__ == '__main__':
    unittest.main()
