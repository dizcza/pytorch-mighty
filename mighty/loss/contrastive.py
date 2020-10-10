from abc import ABC

import numpy as np
import torch
import torch.nn as nn

from mighty.monitor.batch_timer import timer
from mighty.utils.algebra import compute_distance


class PairLoss(nn.Module, ABC):
    """
    A base class for TripletLoss and ContrastiveLoss.

    Parameters
    ----------
    metric : {'cosine', 'l1', 'l2'}, optional
        The metric to use to calculate the distance between two vectors.
        Default: 'cosine'
    margin : float or None, optional
        The margin to have between intra- and inner-samples. Large values
        promote better generalization in cost of lower accuracy. If None is
        passed, the margin will be set to :code:`0.5` if the metric is 'cosine'
        and :code:`1.0` otherwise.
        Default: None
    pairs_multiplier : int, optional
        Defines how many pairs create from a single sample.
        Default: 1
    leave_hardest : float, optional
        Defines the hard negative and positive mining.
        If less than :code:`1.0`, take the "hardest" pairs only that a
        discriminator failed to resolve most. If set to :code:`1.0`, all
        pairs are returned.
        Default: 1.0
    """

    def __init__(self, metric='cosine', margin: float = None,
                 pairs_multiplier: int = 1, leave_hardest: float = 1.0):
        assert 0 < leave_hardest <= 1, "Value should be in (0, 1]"
        super().__init__()
        self.metric = metric
        if margin is None:
            if self.metric == 'cosine':
                margin = 0.5
            else:
                margin = 1.0
        self.margin = margin
        self.leave_hardest = leave_hardest
        self.pairs_multiplier = pairs_multiplier

    @property
    def power(self):
        """
        The metric norm.

        Returns
        -------
        float
            :code:`1` for 'l1' and :code:`2` for 'l2'.
        """
        if self.metric == 'l1':
            return 1
        elif self.metric == 'l2':
            return 2
        else:
            raise NotImplementedError

    def extra_repr(self):
        return f'metric={self.metric}, margin={self.margin}, ' \
               f'pairs_multiplier={self.pairs_multiplier}, ' \
               f'leave_hardest={self.leave_hardest}'

    def _filter_nonzero(self, outputs, labels, normalize: bool):
        # filters out zero vectors
        nonzero = (outputs != 0).any(dim=1)
        outputs = outputs[nonzero]
        labels = labels[nonzero]
        if normalize and self.metric != 'cosine':
            # sparsity changes with epoch but margin stays the same
            outputs = outputs / outputs.norm(p=self.power, dim=1).mean()
        return outputs, labels

    def distance(self, input1, input2):
        """
        Computes the distance between two batches of vectors.

        Parameters
        ----------
        input1, input2 : (B, N) torch.Tensor
            Input and target vector batches.

        Returns
        -------
        torch.Tensor
            A tensor of length B, containing the distances.
        """
        return compute_distance(input1=input1, input2=input2,
                                metric=self.metric, dim=1)


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
        random_pairs_shape : tuple
            A tuple containing one element - the estimated number of random
            permutations to sample to get the desired number of pairs or
            triplets.
        """
        batch_size = len(labels)
        n_unique = len(labels.unique(sorted=False))
        random_pairs_shape = (self.pairs_multiplier * n_unique * batch_size,)
        return random_pairs_shape

    def take_hardest(self, distances):
        """
        Sort the `distances` in descending order and return top
        `self.leave_hardest`.

        Parameters
        ----------
        distances : (B,) torch.Tensor
            Computed distances batch.

        Returns
        -------
        distances : torch.Tensor
            A subsample of the input `distances`. If `self.leave_hardest` is
            :code:`1.0`, do nothing - return the input `distances`.
        """
        if self.leave_hardest < 1.0:
            distances, _unused = distances.sort(descending=True)
            distances = distances[: int(len(distances) * self.leave_hardest)]
        return distances

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


class ContrastiveLossRandom(PairLoss):
    """
    Contrastive Loss [1]_ with random sampling of vector pairs out of the
    conventional :code:`(outputs, labels)` batch.

    A convenient convertor of :code:`(outputs, labels)` batch into
    :code:`(outputs, target)` same-same and same-other pairs.

    Parameters
    ----------
    metric : {'cosine', 'l1', 'l2'}, optional
        The metric to use to calculate the distance between two vectors.
        Default: 'cosine'
    margin : float or None, optional
        The margin to have between intra- and inner-samples. Large values
        promote better generalization in cost of lower accuracy. If None is
        passed, the margin will be set to :code:`0.5` if the metric is 'cosine'
        and :code:`1.0` otherwise.
        Default: None
    pairs_multiplier : int, optional
        Defines how many pairs create from a single sample.
        Default: 1
    leave_hardest : float, optional
        Defines the hard negative and positive mining.
        If less than :code:`1.0`, take the "hardest" pairs only that a
        discriminator failed to resolve most. If set to :code:`1.0`, all
        pairs are returned.
        Default: 1.0
    margin_same : float, optional
        Defines the margin for same-same pairs below which the loss is
        truncated to zero - it does not make sense to penalize for non-exact
        matches.
        Default: 1.0

    References
    ----------
    1. Hadsell, R., Chopra, S., & LeCun, Y. (2006, June). Dimensionality
       reduction by learning an invariant mapping. In 2006 IEEE Computer
       Society Conference on Computer Vision and Pattern Recognition (CVPR'06)
       (Vol. 2, pp. 1735-1742). IEEE.
    """

    def __init__(self, metric='cosine', margin: float = None,
                 pairs_multiplier: int = 1, leave_hardest: float = 1.0,
                 margin_same: float = 0.1):
        super().__init__(metric=metric, margin=margin,
                         pairs_multiplier=pairs_multiplier,
                         leave_hardest=leave_hardest)
        self.margin_same = margin_same

    def extra_repr(self):
        return f'{super().extra_repr()}, margin_same={self.margin_same}'

    def _forward_contrastive(self, outputs, labels):
        return self._forward_random(outputs, labels)

    def _forward_random(self, outputs, labels):
        n_samples = len(outputs)
        pairs_to_sample = self.pairs_to_sample(labels)
        left_indices = torch.randint(low=0, high=n_samples,
                                     size=pairs_to_sample,
                                     device=outputs.device)
        right_indices = torch.randint(low=0, high=n_samples,
                                      size=pairs_to_sample,
                                      device=outputs.device)
        dist = self.distance(outputs[left_indices], outputs[right_indices])
        is_same = labels[left_indices] == labels[right_indices]

        dist_same = dist[is_same]
        dist_other = dist[~is_same]

        return dist_same, dist_other

    def forward(self, outputs, labels):
        outputs, labels = self._filter_nonzero(outputs, labels, normalize=True)
        if timer.is_epoch_finished():
            # if an epoch is finished, use random pairs no matter what the mode is
            dist_same, dist_other = self._forward_random(outputs, labels)
        else:
            dist_same, dist_other = self._forward_contrastive(outputs, labels)

        dist_same = self.take_hardest(dist_same)
        loss_same = torch.relu(dist_same - self.margin_same).mean()

        loss_other = self.margin - dist_other
        loss_other = self.take_hardest(loss_other)
        loss_other = torch.relu(loss_other).mean()

        loss = loss_same + loss_other

        return loss


class ContrastiveLossPairwise(ContrastiveLossRandom):
    """
    Contrastive Loss that samples all possible B x B combinations of pairs,
    given the input batch of B samples.

    Performs worse and slower than `ContrastiveLossRandom` equivalent.

    For the input parameters description, refer to
    :func:`ContrastiveLossRandom`.
    """

    def _forward_contrastive(self, outputs, labels):
        dist_same = []
        dist_other = []
        labels_unique = labels.unique()
        outputs_sorted = {}
        for label in labels_unique:
            outputs_sorted[label.item()] = outputs[labels == label]
        for label_id, label_same in enumerate(labels_unique):
            outputs_same_label = outputs_sorted[label_same.item()]
            n_same = len(outputs_same_label)
            if n_same > 1:
                upper_triangle_idx = np.triu_indices(n=n_same, k=1)
                upper_triangle_idx = torch.as_tensor(upper_triangle_idx,
                                                     device=outputs_same_label.device)
                same_left, same_right = outputs_same_label[upper_triangle_idx]
                dist_same.append(self.distance(same_left, same_right))

            for label_other in labels_unique[label_id + 1:]:
                outputs_other_label = outputs_sorted[label_other.item()]
                n_other = len(outputs_other_label)
                n_max = max(n_same, n_other)
                idx_same = torch.arange(n_max) % n_same
                idx_other = torch.arange(n_max) % n_other
                dist = self.distance(outputs_other_label[idx_other],
                                     outputs_same_label[idx_same])
                dist_other.append(dist)

        dist_same = torch.cat(dist_same)
        dist_other = torch.cat(dist_other)

        return dist_same, dist_other
