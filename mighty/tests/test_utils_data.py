import unittest

import torch
from numpy.testing import assert_array_almost_equal
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Normalize

from mighty.utils.common import set_seed
from mighty.utils.data import DataLoader
from mighty.utils.data.normalize import NormalizeInverse, \
    get_normalize_inverse
from mighty.utils.data.transforms_default import TransformDefault


class TestNormalizeInverse(unittest.TestCase):
    def test_NormalizeInverse(self):
        set_seed(0)
        mean = torch.rand(3)
        std = torch.rand_like(mean)
        normalize = Normalize(mean=mean, std=std, inplace=False)
        normalize_inverse = NormalizeInverse(mean=mean, std=std)
        tensor = torch.rand(5, 3, 12, 12)
        tensor_normalized = normalize(tensor)
        tensor_restored = normalize_inverse(tensor_normalized)
        assert_array_almost_equal(tensor_restored, tensor)

    def test_get_normalize_inverse(self):
        set_seed(1)
        mean = torch.rand(3)
        std = torch.rand_like(mean)
        normalize = Normalize(mean=mean, std=std)
        normalize_inverse = NormalizeInverse(mean=mean, std=std)
        normalize_inverse2 = get_normalize_inverse(normalize)
        assert_array_almost_equal(normalize_inverse2.mean,
                                  normalize_inverse.mean)
        assert_array_almost_equal(normalize_inverse2.std,
                                  normalize_inverse.std)


class TestDefaultTransform(unittest.TestCase):
    def _test_dataset(self, loader):
        for x_tensor, y_labels in loader.eval():
            x_tensor = x_tensor.flatten(start_dim=1)
            mean = x_tensor.mean(dim=1).mean()
            std = x_tensor.std(dim=1).mean()
            assert_array_almost_equal(mean, 0., decimal=1)
            assert_array_almost_equal(std, 1., decimal=1)

    def test_mnist(self):
        set_seed(0)
        transform = TransformDefault.mnist()
        loader = DataLoader(dataset_cls=MNIST,
                            transform=transform, eval_size=10_000)
        self._test_dataset(loader)

    @unittest.skip("Skip CIFAR10 downloading")
    def test_cifar10(self):
        set_seed(0)
        transform = TransformDefault.cifar10()
        loader = DataLoader(dataset_cls=CIFAR10,
                            transform=transform)
        self._test_dataset(loader)

    def test_create_all(self):
        for transform_cls in (TransformDefault.mnist,
                              TransformDefault.cifar10,
                              TransformDefault.imagenet):
            transform_cls()


class TestDataLoader(unittest.TestCase):
    def test_data_loader(self):
        batch_size = 17
        loader = DataLoader(dataset_cls=MNIST, batch_size=batch_size)
        x, y = loader.sample()
        self.assertEqual(x.shape, (batch_size, 1, 28, 28))
        self.assertEqual(y.shape, (batch_size,))
        self.assertTrue(loader.has_labels)

    def test_normalize_inverse(self):
        loader1 = DataLoader(dataset_cls=MNIST)
        self.assertIsNone(loader1.normalize_inverse)
        loader2 = DataLoader(dataset_cls=MNIST,
                             transform=TransformDefault.mnist())
        norm_inv = NormalizeInverse(mean=(0.1307,), std=(0.3081,))
        assert_array_almost_equal(loader2.normalize_inverse.mean,
                                  norm_inv.mean)
        assert_array_almost_equal(loader2.normalize_inverse.std, norm_inv.std)

    def test_eval_size_default(self):
        loader = DataLoader(dataset_cls=MNIST)
        self.assertEqual(loader.eval_size, 60_000)

    def test_eval_size_short(self):
        eval_size = 711
        loader = DataLoader(dataset_cls=MNIST, eval_size=eval_size)
        n_eval = 0
        for x, y in loader.eval():
            n_eval += x.shape[0]
        self.assertEqual(loader.eval_size, eval_size)
        self.assertGreaterEqual(n_eval, loader.eval_size)

    def test_get_test(self):
        loader = DataLoader(dataset_cls=MNIST)
        n_samples = 0
        for x, y in loader.get(train=False):
            n_samples += x.shape[0]
        self.assertGreaterEqual(n_samples, 10_000)


if __name__ == '__main__':
    unittest.main()
