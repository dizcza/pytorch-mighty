import unittest

import torch
from numpy.testing import assert_array_almost_equal, assert_array_equal

from mighty.models import MLP
from mighty.monitor.accuracy import AccuracyArgmax
from mighty.trainer import MaskTrainer
from mighty.utils.common import set_seed


class TestMaskTrainer(unittest.TestCase):

    def setUp(self):
        set_seed(10)
        self.image = torch.rand(1, 10, 10)
        self.model = MLP(self.image.nelement(), 10)
        self.label = 3

    def test_large_learning_rate(self):
        mask_trainer = MaskTrainer(accuracy_measure=AccuracyArgmax(),
                                   image_shape=self.image.shape,
                                   learning_rate=10)
        mask_trainer.cpu()
        mask, image, loss = mask_trainer.train_mask(self.model,
                                                    image=self.image,
                                                    label_true=self.label)
        # check that some pixels are occluded with a mask
        self.assertLess(mask.mean(), 0.8)
        diff = image - self.image
        self.assertGreater(diff.abs().mean(), 0.1)

    def test_zero_learning_rate(self):
        mask_trainer = MaskTrainer(accuracy_measure=AccuracyArgmax(),
                                   image_shape=self.image.shape,
                                   learning_rate=0)
        mask_trainer.cpu()
        mask, image, loss = mask_trainer.train_mask(self.model,
                                                    image=self.image,
                                                    label_true=self.label)
        assert_array_almost_equal(image, self.image)
        assert_array_equal(mask, 1)


if __name__ == '__main__':
    unittest.main()
