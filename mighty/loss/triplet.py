import torch
import torch.nn.functional as F

from mighty.loss.contrastive import PairLoss


class TripletLoss(PairLoss):
    """
    TripletLoss [1]_.

    A convenient convertor of :code:`(outputs, labels)` batch into
    :code:`(anchor, same, other)` triplets.

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

    References
    ----------
    1. Schroff, F., Kalenichenko, D., & Philbin, J. (2015). Facenet: A unified
       embedding for face recognition and clustering. In Proceedings of the
       IEEE conference on computer vision and pattern recognition (pp.
       815-823).
    """

    def forward(self, outputs, labels):
        outputs, labels = self._filter_nonzero(outputs, labels, normalize=True)
        n_samples = len(outputs)
        pairs_to_sample = self.pairs_to_sample(labels)
        anchor = torch.randint(low=0, high=n_samples, size=pairs_to_sample, device=outputs.device)
        same = torch.randint(low=0, high=n_samples, size=pairs_to_sample, device=outputs.device)
        other = torch.randint(low=0, high=n_samples, size=pairs_to_sample, device=outputs.device)

        triplets = (labels[anchor] == labels[same]) & (labels[anchor] != labels[other])
        anchor = anchor[triplets]
        same = same[triplets]
        other = other[triplets]

        if self.metric == 'cosine':
            dist_same = self.distance(outputs[anchor], outputs[same])
            dist_other = self.distance(outputs[anchor], outputs[other])
            loss = dist_same - dist_other + self.margin
            loss = torch.relu(loss)
        else:
            loss = F.triplet_margin_loss(outputs[anchor], outputs[same], outputs[other], margin=self.margin,
                                         p=self.power, reduction='none')
        loss = self.take_hardest(loss).mean()

        return loss
