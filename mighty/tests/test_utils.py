import os

import torch
import torch.nn as nn
import unittest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from mighty.utils.common import find_layers, find_named_layers, input_from_batch, clone_cpu, set_seed
from mighty.utils.hooks import get_layers_ordered, DumpActivationsHook
from mighty.models import MLP, AutoencoderLinear
from mighty.utils.prepare import prepare_eval, ModelMode
from mighty.utils.stub import OptimizerStub
from mighty.utils.signal import exponential_moving_average, compute_sparsity, peak_to_signal_noise_ratio, compute_distance


class TestCommonUtils(unittest.TestCase):
    def test_find_layers(self):
        mlp = MLP(14, 5)
        autoenc = AutoencoderLinear(11, 3)
        self.assertEqual(list(find_layers(mlp, layer_class=MLP)), [mlp])
        model1 = nn.Sequential(mlp, nn.Linear(12, 4))
        self.assertEqual(list(find_layers(model1, layer_class=MLP)), [mlp])
        model2 = nn.Sequential(model1, autoenc, nn.Sequential(mlp))
        self.assertEqual(
            list(find_layers(model2, layer_class=AutoencoderLinear)),
            [autoenc]
        )

    def test_find_named_layers(self):
        mlp = MLP(14, 5)
        autoenc = AutoencoderLinear(11, 3)
        self.assertEqual(list(find_named_layers(mlp, layer_class=MLP)),
                         [('', mlp)])
        model1 = nn.Sequential(mlp, nn.Linear(12, 4))
        self.assertEqual(list(find_named_layers(model1, layer_class=MLP)),
                         [('0', mlp)])
        model2 = nn.Sequential(model1, autoenc, nn.Sequential(mlp))
        self.assertEqual(
            list(find_named_layers(model2, layer_class=AutoencoderLinear)),
            [('1', autoenc)]
        )

    def test_input_from_batch(self):
        x, y = torch.rand(5, 1, 28, 28), torch.arange(5)
        self.assertIs(input_from_batch(x), x)
        self.assertIs(input_from_batch((x, y)), x)

    def test_clone_cpu(self):
        x = torch.rand(5, 2)
        x_clone = clone_cpu(x)
        self.assertIsNot(x, x_clone)
        assert_array_almost_equal(x, x_clone)

    def test_get_layers_ordered(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        c, h, w = 3, 5, 5
        mlp = MLP(c * h * w, 12, 19)
        ordered = get_layers_ordered(mlp, input_sample=torch.rand(c, h, w),
                                     ignore_layers=(nn.ReLU, nn.Sequential))
        self.assertEqual(len(ordered), 2)
        self.assertIsInstance(ordered[0], nn.Linear)
        self.assertIsInstance(ordered[1], nn.Linear)

        linear1 = nn.Linear(12, 16)
        linear2 = nn.Linear(16, 4)
        model = nn.Sequential(linear1, linear2)
        ordered = get_layers_ordered(model, input_sample=torch.rand(12))
        self.assertEqual(ordered, [linear1, linear2])

    def test_DumpActivationsHook(self):
        model = nn.Sequential(nn.Linear(4, 12), nn.Linear(12, 5))
        dumper = DumpActivationsHook(model)
        tensor = torch.rand(10, 4)
        model(tensor)
        dumper.remove_hooks()

    def test_prepare_eval(self):
        mlp = MLP(12, 3)
        conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        linear = nn.Linear(4, 5)
        mlp.requires_grad_(False)
        model = nn.Sequential(mlp, conv, linear)
        model_mode = prepare_eval(model)
        self.assertIsInstance(model_mode, ModelMode)
        self.assertTrue(model_mode.mode)
        self.assertEqual(model_mode.requires_grad, {
            '0.mlp.0.weight': False,
            '0.mlp.0.bias': False,
            '1.weight': True,
            '1.bias': True,
            '2.weight': True,
            '2.bias': True
        })

    def test_OptimizerStub(self):
        optimizer = OptimizerStub()
        optimizer.step()
        self.assertEqual(optimizer.state_dict(), {})
        optimizer.load_state_dict({})

    def test_exponential_moving_average_psnr(self):
        set_seed(1)
        noise = torch.rand(100)
        smoothed = exponential_moving_average(noise, window=3)
        psnr = peak_to_signal_noise_ratio(noise, smoothed)
        self.assertGreaterEqual(psnr, 15.7)
        self.assertEqual(smoothed.shape, noise.shape)
        self.assertLess(smoothed.std(), noise.std())

    def test_peak_to_signal_noise_ratio(self):
        tensor = torch.rand(3, 20)
        psnr_inf = peak_to_signal_noise_ratio(tensor, tensor)
        # PSNR is a quality measure between the corrupted and original signals.
        # If the signals match precisely, it's a lossless encoder, and psnr
        # evaluates to +inf.
        self.assertTrue(torch.isinf(psnr_inf))
        tensor = torch.ones(3, 20)
        psnr_nan = peak_to_signal_noise_ratio(tensor, torch.rand_like(tensor))
        self.assertTrue(torch.isnan(psnr_nan))
