import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Creates sequential fully-connected layers FC_1->FC_2->...->FC_N.

    Parameters
    ----------
    fc_sizes : int
        Fully connected sequential layer sizes.
    """

    def __init__(self, *fc_sizes: int):
        super().__init__()
        fc_sizes = list(fc_sizes)
        n_classes = fc_sizes.pop()
        classifier = []
        for in_features, out_features in zip(fc_sizes[:-1], fc_sizes[1:]):
            classifier.append(nn.Linear(in_features, out_features))
            classifier.append(nn.ReLU(inplace=True))
        classifier.append(nn.Linear(in_features=fc_sizes[-1],
                                    out_features=n_classes))
        self.mlp = nn.Sequential(*classifier)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.mlp(x)
        return x


class DropConnect(nn.Module):
    __constants__ = ["p"]

    def __init__(self, linear: nn.Linear, p=0.5):
        super(DropConnect, self).__init__()
        self.linear = linear
        self.p = p

    def forward(self, x):
        if self.training:
            mask = torch.bernoulli(torch.ones_like(self.linear.weight) * (1 - self.p))
            weights = self.linear.weight * mask
            return F.linear(x, weights, self.linear.bias)
        return self.linear(x)

    def extra_repr(self):
        return f"p={self.p}"


class MLPDropout(nn.Module):
    """
    Creates sequential fully-connected layers FC_1->FC_2->...->FC_N with `nn.Dropout` in-between.

    Parameters
    ----------
    fc_sizes : int
        Fully connected sequential layer sizes.
    p, p_input : float, optional
        Dropout probabilities for hidden and input layers.
    """

    def __init__(self, *fc_sizes: int, p=0.5, p_input=0.25, p_connect=0.5):
        super().__init__()
        fc_sizes = list(fc_sizes)
        n_classes = fc_sizes.pop()
        classifier = []
        for in_features, out_features in zip(fc_sizes[:-1], fc_sizes[1:]):
            p_layer = p_input if in_features == fc_sizes[0] else p
            if p_layer is not None and p_layer > 0:
                classifier.append(nn.Dropout(p=p_layer))
            linear = nn.Linear(in_features, out_features)
            if p_connect is not None and p_connect > 0:
                linear = DropConnect(linear, p=p_connect)
            classifier.append(linear)
            classifier.append(nn.ReLU(inplace=True))
        if p is not None and p > 0:
            classifier.append(nn.Dropout(p=p))
        classifier.append(nn.Linear(in_features=fc_sizes[-1],
                                    out_features=n_classes))
        self.mlp = nn.Sequential(*classifier)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.mlp(x)
        return x
