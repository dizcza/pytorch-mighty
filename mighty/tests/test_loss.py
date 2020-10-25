import unittest

import torch
import torch.nn as nn

from mighty.loss import ContrastiveLossSampler, TripletLossSampler, \
    LossPenalty, TripletCosineLoss
from mighty.utils.common import set_seed


class TestPairLoss(unittest.TestCase):

    def setUp(self):
        set_seed(0)
        n_classes = 3
        labels = torch.arange(n_classes)
        self.labels = torch.cat([labels, labels])
        outputs_same = torch.randn(n_classes, 30)
        self.outputs_same = torch.cat([outputs_same, outputs_same])
        self.loss_models = (
            ContrastiveLossSampler(nn.CosineEmbeddingLoss(margin=0.5)),
            TripletLossSampler(nn.TripletMarginLoss()),
            TripletLossSampler(TripletCosineLoss())
        )

    def test_same_pairs(self):
        # The loss is expected to be zero because we set a large margin.
        for loss_model in self.loss_models:
            with self.subTest(loss_model=loss_model):
                loss = loss_model(self.outputs_same, self.labels)
                self.assertEqual(loss, 0.)

    def test_rand_pairs(self):
        set_seed(1)
        outputs = torch.randn(self.labels.shape[0], 30)
        for loss_model in self.loss_models:
            with self.subTest(loss_model=loss_model):
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
