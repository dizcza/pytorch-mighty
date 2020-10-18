import unittest
import math

from numpy.testing import assert_array_almost_equal, assert_array_less
from torchvision.datasets import MNIST
from torchvision.transforms import Resize, ToTensor, Compose

from mighty.models import MLP
from mighty.monitor.mutual_info import *
from mighty.monitor.viz import VisdomMighty
from mighty.utils.data import DataLoader
from mighty.utils.common import set_seed

try:
    from mighty.monitor.mutual_info.idtxl import MutualInfoIDTxl
except ImportError:
    from mighty.monitor.mutual_info import MutualInfoStub as MutualInfoIDTxl


class TestMutualInfoNPEET(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        imsize = 5
        transform = Compose([Resize(imsize), ToTensor()])
        cls.data_loader = DataLoader(dataset_cls=MNIST, transform=transform,
                                     batch_size=20, eval_size=1000)
        cls.model = MLP(imsize ** 2, 10)
        cls.viz = VisdomMighty(env='test', offline=True)

    def setUp(self):
        set_seed(0)
        self.mi_instance = self._init_mutual_info(pca_size=None)
        self.mi_instance.prepare(self.model)

    def _init_mutual_info(self, pca_size):
        return MutualInfoNPEET(data_loader=self.data_loader,
                               pca_size=pca_size, debug=True)

    def _test_estimated_values(self, mi_pca, mi_no_pca):
        assert_array_almost_equal(mi_pca, mi_no_pca, decimal=0)

    def test_force_update(self):
        set_seed(1)
        mi_instance_pca = self._init_mutual_info(pca_size=20)
        mi_instance_pca.prepare(self.model)

        self.mi_instance.force_update(self.model)
        mi_instance_pca.force_update(self.model)
        mi_no_pca = self.mi_instance.information
        mi_pca = mi_instance_pca.information
        self.assertEqual(mi_pca.keys(), mi_no_pca.keys())
        for mi_layers_dict in (mi_pca, mi_no_pca):
            for mi_x, mi_y in mi_layers_dict.values():
                # MI must be non-negative
                self.assertGreater(mi_x, 0)
                self.assertGreater(mi_y, 0)
        for layer_name in mi_pca.keys():
            self._test_estimated_values(mi_pca[layer_name],
                                        mi_no_pca[layer_name])
        mi_instance_pca.plot(self.viz)
        mi_instance_pca.plot_activations_hist(self.viz)


class TestMutualInfoGCMI(TestMutualInfoNPEET):
    def _init_mutual_info(self, pca_size):
        return MutualInfoGCMI(data_loader=self.data_loader,
                              pca_size=pca_size, debug=True)

    def _test_estimated_values(self, mi_pca, mi_no_pca):
        print(mi_pca, mi_no_pca)
        # w/ PCA and w/o PCA values differ.
        # The current implementation overestimates the true values.
        # Check for being larger than '3', although any reference value in
        # range [0, 5] can be used here.
        assert_array_less(3, mi_pca)
        assert_array_less(3, mi_no_pca)


@unittest.skipIf(MutualInfoIDTxl is MutualInfoStub, "idtxl is not installed")
class TestMutualInfoIDTxl(TestMutualInfoNPEET):
    def _init_mutual_info(self, pca_size):
        return MutualInfoIDTxl(data_loader=self.data_loader,
                               pca_size=pca_size, debug=True)


class TestMutualInfoKMeans(TestMutualInfoNPEET):
    def _init_mutual_info(self, pca_size):
        return MutualInfoKMeans(data_loader=self.data_loader, n_bins=5,
                                debug=True)


class TestMutualInfoNeuralEstimation(TestMutualInfoNPEET):
    def _init_mutual_info(self, pca_size):
        return MutualInfoNeuralEstimation(data_loader=self.data_loader,
                                          pca_size=pca_size, debug=True)

    # Don't double-test the same functional
    def test_estimate_accuracy(self):
        n_classes = self.mi_instance.accuracy_estimator.n_classes
        info_y_layerA = 0.
        info_y_layerB = math.log2(n_classes)
        self.mi_instance.information = dict(layerA=(0., info_y_layerA), layerB=(0., info_y_layerB))
        accuracies = self.mi_instance.estimate_accuracy()
        # low I(X; T) corresponds to random accuracy
        self.assertAlmostEqual(accuracies['layerA'], 1 / n_classes, places=1)
        # the largest I(Y; T) is log2(n_classes) that corresponds to 100 %
        # accuracy
        self.assertAlmostEqual(accuracies['layerB'], 1.0, places=1)


if __name__ == '__main__':
    unittest.main()
