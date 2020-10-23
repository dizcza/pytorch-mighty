import unittest

import torch
from numpy.testing import assert_array_almost_equal, assert_array_equal
from torch.utils.data import TensorDataset, DataLoader

from mighty.monitor.var_online import VarianceOnline, VarianceOnlineBatch, \
    MeanOnline, MeanOnlineBatch, MeanOnlineLabels, VarianceOnlineLabels


class TestVarianceOnline(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(28)
        self.data = torch.rand(500, 5)
        self.mean_true = self.data.mean(dim=0).numpy()
        self.std_unbiased = self.data.std(dim=0, unbiased=True).numpy()
        self.std_biased = self.data.std(dim=0, unbiased=False).numpy()

    def test_mean_online(self):
        mean_online = MeanOnline()
        for sample in self.data:
            mean_online.update(sample)
        mean = mean_online.get_mean()
        assert_array_almost_equal(mean.numpy(), self.mean_true)

    def test_mean_online_batch(self):
        mean_online = MeanOnlineBatch()
        for batch_tensor in self.data.split(split_size=20):
            mean_online.update(batch_tensor)
        mean = mean_online.get_mean()
        assert_array_almost_equal(mean.numpy(), self.mean_true)

    def test_variance_online(self):
        var_online = VarianceOnline()
        for sample in self.data:
            var_online.update(sample)
        mean, std_unbiased = var_online.get_mean_std(unbiased=True)
        _, std_biased = var_online.get_mean_std(unbiased=False)
        assert_array_almost_equal(mean.numpy(), self.mean_true)
        assert_array_almost_equal(std_unbiased.numpy(), self.std_unbiased)
        assert_array_almost_equal(std_biased.numpy(), self.std_biased)

    def test_variance_online_batch(self):
        var_online_batch = VarianceOnlineBatch()
        for batch_tensor in self.data.split(split_size=20):
            var_online_batch.update(batch_tensor)
        mean, std_unbiased = var_online_batch.get_mean_std(unbiased=True)
        _, std_biased = var_online_batch.get_mean_std(unbiased=False)
        assert_array_almost_equal(mean.numpy(), self.mean_true)
        assert_array_almost_equal(std_unbiased.numpy(), self.std_unbiased)
        assert_array_almost_equal(std_biased.numpy(), self.std_biased)


class TestVarianceOnlineLabels(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(28)
        self.data = torch.randn(100, 5)
        self.labels = torch.randint(low=0, high=10, size=(self.data.shape[0],))
        self.loader = DataLoader(TensorDataset(self.data, self.labels),
                                 batch_size=5,
                                 shuffle=True)

        self.labels_unique = []
        centroids = []
        std_unbiased = []
        std_biased = []
        for label in self.labels.unique(sorted=True):
            self.labels_unique.append(label.item())
            tensor_label = self.data[self.labels == label]
            centroids.append(tensor_label.mean(dim=0))
            std_unbiased.append(tensor_label.std(dim=0, unbiased=True))
            std_biased.append(tensor_label.std(dim=0, unbiased=False))
        self.centroids = torch.stack(centroids, dim=0).numpy()
        self.std_unbiased = torch.stack(std_unbiased, dim=0).numpy()
        self.std_biased = torch.stack(std_biased, dim=0).numpy()

    def test_mean_online_labels(self):
        mean_online = MeanOnlineLabels()
        for data, labels in iter(self.loader):
            mean_online.update(data, labels)
        centroids, labels = mean_online.get_mean_labels()
        assert_array_equal(labels, self.labels_unique)
        assert_array_equal(mean_online.labels(), self.labels_unique)
        assert_array_almost_equal(centroids.numpy(), self.centroids)
        assert_array_almost_equal(mean_online.get_mean().numpy(),
                                  self.centroids)

    def test_variance_online_labels(self):
        var_online = VarianceOnlineLabels()
        for data, labels in iter(self.loader):
            var_online.update(data, labels)
        for unbiased, std_target in ((True, self.std_unbiased),
                                     (False, self.std_biased)):
            with self.subTest(unbiased=unbiased):
                centroids, std, labels = var_online.get_mean_std_labels(
                    unbiased=unbiased)
                centroids2, std2 = var_online.get_mean_std(unbiased)
                centroids3 = var_online.get_mean()
                assert_array_equal(labels, self.labels_unique)
                assert_array_equal(var_online.labels(), self.labels_unique)

                assert_array_almost_equal(centroids.numpy(), self.centroids)
                assert_array_almost_equal(centroids2.numpy(), self.centroids)
                assert_array_almost_equal(centroids3.numpy(), self.centroids)

                assert_array_almost_equal(std.numpy(), std_target)
                assert_array_almost_equal(std2.numpy(), std_target)


if __name__ == '__main__':
    unittest.main()
