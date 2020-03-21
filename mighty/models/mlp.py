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
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x
