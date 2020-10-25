import torch
import torch.nn as nn
import torch.nn.functional as F

from mighty.loss.contrastive import PairLossSampler


class TripletCosineLoss(nn.Module):
    r"""
    Creates a criterion that measures the cosine triplet loss given an input
    tensors :math:`x1`, :math:`x2`, :math:`x3` and a margin with a value
    greater than :math:`0`. This is used for measuring a relative similarity
    between samples. A triplet is composed by `a`, `p` and `n` (i.e., `anchor`,
    `positive examples` and `negative examples` respectively). The shapes of
    all input tensors should be :math:`(N, D)`.

    The loss function for each sample in the mini-batch is:

    .. math::
        L(a, p, n) = \max \{cos(a_i, n_i) - cos(a_i, p_i) + {\rm margin}, 0\}

    """
    def __init__(self, margin=0.):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        sim_positive = F.cosine_similarity(anchor, positive, dim=1)
        sim_negative = F.cosine_similarity(anchor, negative, dim=1)
        loss = torch.relu(sim_negative - sim_positive + self.margin).mean()
        return loss


class TripletLossSampler(PairLossSampler):
    """
    TripletLoss [1]_ with random sampling of triplets out of the
    conventional :code:`(outputs, labels)` batch.

    A convenient convertor of :code:`(outputs, labels)` batch into
    :code:`(anchor, same, other)` triplets.

    Parameters
    ----------
    criterion : nn.Module
        Triplet Loss module (e.g., ``nn.TripletMarginLoss``) to compute the
        loss, followed by pairs sampling.
    pairs_multiplier : int, optional
        Defines how many pairs to create from a single sample. The typical
        range is ``[1, 10]``.
        Default: 1

    References
    ----------
    1. Schroff, F., Kalenichenko, D., & Philbin, J. (2015). Facenet: A unified
       embedding for face recognition and clustering. In Proceedings of the
       IEEE conference on computer vision and pattern recognition (pp.
       815-823).
    """

    def forward(self, outputs, labels):
        n_samples = len(outputs)
        pairs_to_sample = (self.pairs_to_sample(labels),)
        anchor = torch.randint(low=0, high=n_samples, size=pairs_to_sample,
                               device=outputs.device)
        positive = torch.randint(low=0, high=n_samples, size=pairs_to_sample,
                                 device=outputs.device)
        negative = torch.randint(low=0, high=n_samples, size=pairs_to_sample,
                                 device=outputs.device)

        triplets = (anchor != positive) & \
                   (labels[anchor] == labels[positive]) & \
                   (labels[anchor] != labels[negative])
        anchor = anchor[triplets]
        positive = positive[triplets]
        negative = negative[triplets]

        loss = self.criterion(outputs[anchor],
                              outputs[positive],
                              outputs[negative])
        self._check_non_nan(loss)

        return loss
