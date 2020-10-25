from abc import ABC

import torch
import torch.nn as nn
import warnings


class PairLossSampler(nn.Module, ABC):
    """
    A base class for TripletLossSampler and ContrastiveLossSampler.

    Parameters
    ----------
    criterion : nn.Module
        A criterion module to compute the loss, followed by pairs sampling.
    pairs_multiplier : int, optional
        Defines how many pairs to create from a single sample. The typical
        range is ``[1, 10]``.
        Default: 1
    """

    def __init__(self, criterion: nn.Module, pairs_multiplier: int = 1):
        super().__init__()
        self.criterion = criterion
        self.pairs_multiplier = pairs_multiplier

    def extra_repr(self):
        margin = getattr(self.criterion, 'margin')
        margin_str = '' if margin is None else f", criterion.margin={margin}"
        return f"pairs_multiplier={self.pairs_multiplier}{margin_str}"

    def pairs_to_sample(self, labels):
        """
        Estimates how many pairs to sample to a batch of `labels`.

        The probability of two random samples having the same class is
        :code:`1/n_classes`. On average, each single sample in a batch
        produces :code:`1/n_classes` pairs or
        :code:`1/n_classes * (1 - 1/n_classes)` triplets.

        Parameters
        ----------
        labels : (B,) torch.LongTensor
            A batch of labels.

        Returns
        -------
        n_random_pairs : int
            An estimated number of random permutations to sample to get the
            desired number of pairs or triplets.
        """
        batch_size = len(labels)
        n_unique = len(labels.unique(sorted=False))
        n_random_pairs = self.pairs_multiplier * n_unique * batch_size
        return n_random_pairs

    def _check_non_nan(self, loss: torch.Tensor):
        if torch.isnan(loss):
            warnings.warn("Loss evaluated to NaN probably because there were "
                          "no pairs to sample. Increase the "
                          "'pairs_multiplier'.")

    def forward(self, outputs, labels):
        """
        Converts the input batch into pairs or triplets and computes
        Contrastive or Triplet Loss.

        Parameters
        ----------
        outputs : (B, N) torch.Tensor
            The output of a model.
        labels : (B,) torch.LongTensor
            A batch of the true labels.

        Returns
        -------
        loss : torch.Tensor
            Loss scalar tensor.
        """
        raise NotImplementedError


class ContrastiveLossSampler(PairLossSampler):
    """
    Contrastive Loss [1]_ with random sampling of vector pairs out of the
    conventional :code:`(outputs, labels)` batch.

    A convenient convertor of :code:`(outputs, labels)` batch into
    :code:`(outputs, target)` same-same and same-other pairs.

    Parameters
    ----------
    criterion : nn.Module
        Contrastive Loss module (e.g., ``nn.CosineEmbeddingLoss``) to compute
        the loss, followed by pairs sampling.
    pairs_multiplier : int, optional
        Defines how many pairs to create from a single sample. The typical
        range is ``[1, 10]``.
        Default: 1

    References
    ----------
    1. Hadsell, R., Chopra, S., & LeCun, Y. (2006, June). Dimensionality
       reduction by learning an invariant mapping. In 2006 IEEE Computer
       Society Conference on Computer Vision and Pattern Recognition (CVPR'06)
       (Vol. 2, pp. 1735-1742). IEEE.
    """

    def forward(self, outputs, labels):
        n_samples = len(outputs)
        pairs_to_sample = (self.pairs_to_sample(labels),)
        left_indices = torch.randint(low=0, high=n_samples,
                                     size=pairs_to_sample,
                                     device=outputs.device)
        right_indices = torch.randint(low=0, high=n_samples,
                                      size=pairs_to_sample,
                                      device=outputs.device)

        # exclude (a, a) pairs
        indices_different = left_indices != right_indices
        left_indices = left_indices[indices_different]
        right_indices = right_indices[indices_different]

        # exclude [(a, b), (b, a)] duplicate pairs
        left_indices, right_indices = torch.stack(
            [left_indices, right_indices], dim=1).sort(dim=1).values.sort(
            dim=0).values.unique(sorted=False, dim=0).t()

        is_same = labels[left_indices] == labels[right_indices]
        y_target = 2 * is_same - 1
        loss = self.criterion(outputs[left_indices],
                              outputs[right_indices],
                              y_target)
        self._check_non_nan(loss)

        return loss
