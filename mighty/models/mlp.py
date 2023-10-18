import torch.nn as nn


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

    def __init__(self, *fc_sizes: int, p=0.5, p_input=0.25):
        super().__init__()
        fc_sizes = list(fc_sizes)
        n_classes = fc_sizes.pop()
        classifier = []
        for in_features, out_features in zip(fc_sizes[:-1], fc_sizes[1:]):
            classifier.append(nn.Dropout(p=p_input if in_features == fc_sizes[0] else p))
            classifier.append(nn.Linear(in_features, out_features))
            classifier.append(nn.ReLU(inplace=True))
        classifier.append(nn.Dropout(p=p))
        classifier.append(nn.Linear(in_features=fc_sizes[-1],
                                    out_features=n_classes))
        self.mlp = nn.Sequential(*classifier)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.mlp(x)
        return x
