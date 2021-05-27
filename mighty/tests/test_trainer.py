import os
import unittest

import numpy as np
import torch
import torch.nn as nn
from numpy.testing import assert_array_almost_equal, assert_array_less
from torchvision.datasets import MNIST

from mighty.loss import *
from mighty.models import MLP, AutoencoderLinear
from mighty.monitor.accuracy import AccuracyEmbedding
from mighty.monitor.mutual_info import MutualInfoNPEET
from mighty.monitor.monitor import MonitorLevel
from mighty.trainer import *
from mighty.utils.common import set_seed
from mighty.utils.data import TransformDefault, DataLoader


class TrainerTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.environ['VISDOM_SERVER'] = "http://85.217.171.57"
        os.environ['VISDOM_PORT'] = '8096'

    def setUp(self):
        set_seed(1)
        self.model = MLP(784, 64, 10)
        self.data_loader = DataLoader(MNIST, eval_size=10000,
                                      transform=TransformDefault.mnist())

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
        trainer.env_name = f"pytorch-mighty tests"
        loss_epochs = trainer.train(n_epochs=1, mutual_info_layers=0,
                                    mask_explain_params=dict())
        assert_array_almost_equal(loss_epochs, [0.219992])
        trainer.save()
        trainer.restore()

    def test_TrainerGrad_MutualInfo(self):
        set_seed(2)
        mi_history = []

        class MutualInfoNPEETDebug(MutualInfoNPEET):
            def plot(self_mi, viz):
                mi_history.append(self_mi.information['mlp.2'])
                super().plot(viz)

        data_loader = DataLoader(MNIST, eval_size=100,
                                 transform=TransformDefault.mnist())
        trainer = TrainerGrad(self.model,
                              criterion=nn.CrossEntropyLoss(),
                              data_loader=data_loader,
                              optimizer=self.optimizer,
                              mutual_info=MutualInfoNPEETDebug(
                                  data_loader=data_loader, pca_size=None),
                              scheduler=self.scheduler)
        trainer.env_name = "pytorch-mighty tests"
        trainer.monitor.advanced_monitoring(level=MonitorLevel.FULL)
        trainer.train(n_epochs=1, mutual_info_layers=1)
        mi_history = np.vstack(mi_history)
        assert_array_less(0, mi_history)
        x_last, y_last = mi_history[-1]

        # the model is well trained
        self.assertGreater(y_last, 2.5)
        self.assertGreater(x_last, 2.2)
        self.assertLessEqual(x_last, y_last)

    def test_TrainerEmbedding(self):
        set_seed(3)
        criterion = TripletLossSampler(nn.TripletMarginLoss())
        trainer = TrainerEmbedding(self.model,
                                   criterion=criterion,
                                   data_loader=self.data_loader,
                                   optimizer=self.optimizer,
                                   scheduler=self.scheduler)
        trainer.env_name = "pytorch-mighty tests"
        loss_epochs = trainer.train(n_epochs=1, mutual_info_layers=0)
        # CircleCI outputs 0.103
        assert_array_almost_equal(loss_epochs, [0.09936], decimal=2)

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
        model = AutoencoderLinear(784, 64)
        trainer = TrainerAutoencoder(model,
                                     criterion=nn.BCEWithLogitsLoss(),
                                     data_loader=self.data_loader,
                                     optimizer=self.optimizer,
                                     scheduler=self.scheduler)
        trainer.env_name = "pytorch-mighty tests"
        loss_epochs = trainer.train(n_epochs=1, mutual_info_layers=0)
        assert_array_almost_equal(loss_epochs, [0.69737625122])

    def test_TrainerAutoencoder_cached(self):
        set_seed(3)
        model = AutoencoderLinear(784, 64)
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
