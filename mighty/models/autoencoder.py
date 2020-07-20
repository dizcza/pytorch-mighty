import torch.nn as nn


class AutoencoderLinear(nn.Module):
    """
    The simplest linear AutoEncoder.

    Parameters
    ----------
    fc_sizes: int
        The sizes of fully connected layers of a resulting AutoEncoder.
        Starts with the input dimension, ends with the embedding dimension.

    Examples
    --------
    >>> AutoencoderLinear(784, 128)
    AutoencoderLinear(
      (encoder): Sequential(
        (0): Linear(in_features=784, out_features=128, bias=True)
        (1): ReLU(inplace=True)
      )
      (decoder): Linear(in_features=128, out_features=784, bias=True)
    )
    >>> AutoencoderLinear(784, 256, 128)
    AutoencoderLinear(
      (encoder): Sequential(
        (0): Linear(in_features=784, out_features=256, bias=True)
        (1): ReLU(inplace=True)
        (2): Linear(in_features=256, out_features=128, bias=True)
        (3): ReLU(inplace=True)
      )
      (decoder): Linear(in_features=128, out_features=784, bias=True)
    )

    """

    def __init__(self, *fc_sizes):
        super().__init__()
        encoder = []
        for in_features, out_features in zip(fc_sizes[:-1], fc_sizes[1:]):
            encoder.append(nn.Linear(in_features, out_features))
            encoder.append(nn.ReLU(inplace=True))

        input_dim = fc_sizes[0]
        self.encoding_dim = fc_sizes[-1]

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Linear(self.encoding_dim, input_dim)

    def forward(self, x):
        input_shape = x.shape
        x = x.flatten(start_dim=1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.view(*input_shape)
        return encoded, decoded
