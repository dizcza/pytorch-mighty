import unittest

import torch
import torch.nn as nn
from numpy.testing import assert_array_equal

from mighty.models import *
from mighty.utils.common import set_seed


class TestModels(unittest.TestCase):

    def test_AutoEncoderLinear(self):
        set_seed(0)
        in_features, hidden_features = 16, 64
        batch_size = 5
        model = AutoencoderLinear(in_features, hidden_features)
        self.assertEqual(model.encoding_dim, hidden_features)
        self.assertIsInstance(model.encoder, nn.Module)
        self.assertIsInstance(model.decoder, nn.Module)
        tensor = torch.rand(batch_size, in_features)
        output = model(tensor)
        self.assertIsInstance(output, AutoencoderOutput)
        self.assertEqual(output.latent.shape, (batch_size, hidden_features))
        self.assertEqual(output.reconstructed.shape, (batch_size, in_features))

    def test_flatten(self):
        set_seed(1)
        model = Flatten()
        tensor = torch.rand(5, 1, 28, 28)
        output = model(tensor)
        assert_array_equal(output, tensor.flatten(start_dim=1))

    def test_reshape(self):
        set_seed(2)
        model = Reshape(height=10, width=12)
        tensor = torch.rand(5, 3 * model.height * model.width)
        output = model(tensor)
        assert_array_equal(output,
                           tensor.view(5, 3, model.height, model.width))

    def test_mlp(self):
        set_seed(3)
        in_features, hidden_features, output_features = 32, 12, 17
        model = MLP(in_features, hidden_features, output_features)
        tensor = torch.rand(5, in_features)
        output = model(tensor)
        self.assertEqual(output.shape, (tensor.shape[0], output_features))

    def test_SerializableModule(self):
        model = SerializableModule()
        tensor = torch.arange(5)
        model.register_buffer(name="t1", tensor=tensor)
        model.state_attr = ["t1"]
        state_dict = model.state_dict()
        self.assertTrue("t1" in state_dict)
        assert_array_equal(state_dict["t1"], tensor)


if __name__ == '__main__':
    unittest.main()
