import unittest

import torch
import torch.nn as nn

from mighty.loss import ContrastiveLossRandom, ContrastiveLossPairwise, \
    TripletLoss, LossPenalty
from mighty.utils.common import set_seed


class TestPairLoss(unittest.TestCase):

    def setUp(self):
        set_seed(17)
        n_classes = 10
        labels = torch.arange(n_classes)
        self.labels = torch.cat([labels, labels])
        outputs_same = torch.randn(n_classes, 30)
        self.outputs_same = torch.cat([outputs_same, outputs_same])
        self.loss_models = (ContrastiveLossRandom(), ContrastiveLossPairwise(), TripletLoss())

    def test_same_pairs(self):
        for loss_model in self.loss_models:
            loss = loss_model(self.outputs_same, self.labels)
            self.assertEqual(loss, 0.)

    def test_rand_pairs(self):
        set_seed(17)
        outputs = torch.randn(self.labels.shape[0], 30)
        for loss_model in self.loss_models:
            loss = loss_model(outputs, self.labels)
            self.assertGreater(loss, 0.)


class TestLossPenalty(unittest.TestCase):

    def test_zero_lambda(self):
        loss_orig = nn.MSELoss()
        loss_penalty = LossPenalty(loss_orig, lambd=0.)
        x, y = torch.randn(2, 10, 20)
        z = torch.randn_like(x)  # latent variable
        self.assertAlmostEqual(loss_orig(x, y), loss_penalty(x, y, z))

    def test_loss_penalty(self):
        loss_orig = nn.MSELoss()
        lambd = 0.5
        loss_penalty = LossPenalty(loss_orig, lambd=lambd, norm=1)
        x, y = torch.randn(2, 10, 20)
        z = torch.randn_like(x)  # latent variable
        l1_norm = z.norm(p=1, dim=1).mean() / z.shape[1]
        loss_expected = loss_orig(x, y) + lambd * l1_norm
        self.assertAlmostEqual(loss_penalty(x, y, z), loss_expected)


if __name__ == '__main__':
    unittest.main()
