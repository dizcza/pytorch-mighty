"""
Mean and variance online measures
---------------------------------

.. autosummary::
    :toctree: toctree/monitor
    
    MeanOnline
    MeanOnlineBatch
    SumOnlineBatch
    MeanOnlineLabels
    VarianceOnline
    VarianceOnlineBatch
    VarianceOnlineLabels
"""

from collections import defaultdict

import torch
import torch.utils.data


class MeanOnline:
    """
    Online sample mean aggregate. Works with scalars, vectors, and
    n-dimensional tensors.
    
    Parameters
    ----------
    tensor : torch.Tensor or None
        The initial tensor, if provided.
    """

    def __init__(self, tensor=None):
        self.mean = None
        self.count = 0
        self.is_active = True
        if tensor is not None:
            self.update(tensor)

    def activate(self, is_active):
        """
        Activates or deactivates the updates.
        
        Parameters
        ----------
        is_active : bool
            New state.
        """
        self.is_active = is_active

    def update(self, tensor):
        """
        Update sample mean (and variance) from a batch of new values.
        
        Parameters
        ----------
        tensor : torch.Tensor
            Next tensor sample.
        """
        if not self.is_active:
            return
        self.count += 1
        if self.mean is None:
            self.mean = tensor.clone()
        else:
            self.mean += (tensor - self.mean) / self.count

    def get_mean(self):
        """
        Returns
        -------
        torch.Tensor
            The mean of all tensors.
        """
        if self.mean is None:
            return None
        else:
            return self.mean.clone()

    def reset(self):
        """
        Reset the mean and the count.
        """
        self.mean = None
        self.count = 0


class VarianceOnline(MeanOnline):
    """
    Welford's online algorithm for population mean and variance estimation.
    """

    def __init__(self, tensor=None):
        self.M2 = None
        super().__init__(tensor)

    def update(self, tensor):
        if not self.is_active:
            return
        self.count += 1
        if self.mean is None:
            self.mean = torch.zeros_like(tensor)
            self.M2 = torch.zeros_like(tensor)
        delta_var = tensor - self.mean
        self.mean += delta_var / self.count
        delta_var.mul_(tensor - self.mean)
        self.M2 += delta_var

    def get_mean_std(self, unbiased=True):
        """
        Return mean and std of all samples.

        Parameters
        ----------
        unbiased : bool, optional
            Biased (False) or unbiased (True) variance estimate.
            Default: True

        Returns
        -------
        mean : torch.Tensor
            The mean of all samples.
        std : torch.Tensor
            The std of all samples.

        """
        if self.mean is None:
            return None, None
        if self.count > 1:
            count = self.count - 1 if unbiased else self.count
            std = torch.sqrt(self.M2 / count)
        else:
            # with 1 update both biased & unbiased sample variance is zero
            std = torch.zeros_like(self.mean)
        return self.mean.clone(), std

    def reset(self):
        super().reset()
        self.var = None


class MeanOnlineBatch(MeanOnline):
    """
    Online mean measure that updates 1d vector mean from a batch of vectors
    (2d tensor).
    """

    def update(self, tensor):
        if not self.is_active:
            return
        batch_size = tensor.shape[0]
        self.count += batch_size
        if self.mean is None:
            self.mean = tensor.mean(dim=0)
        else:
            self.mean += (tensor.sum(dim=0) -
                          self.mean * batch_size) / self.count


class SumOnlineBatch:
    """
    Online sum measure.
    """
    def __init__(self):
        self.sum = None
        self.count = 0
        self.is_active = True

    def activate(self, is_active):
        self.is_active = is_active

    def update(self, tensor: torch.Tensor):
        if not self.is_active:
            return
        self.count += tensor.shape[0]
        if self.sum is None:
            self.sum = tensor.sum(dim=0)
        else:
            self.sum += tensor.sum(dim=0)

    def get_sum(self):
        if self.sum is None:
            return None
        return self.sum.clone()

    def reset(self):
        self.sum = None
        self.count = 0


class VarianceOnlineBatch(VarianceOnline):
    """
    Welford's online algorithm for population mean and variance estimation
    from batches of 1d vectors.
    """

    def update(self, tensor):
        if not self.is_active:
            return
        batch_size = tensor.shape[0]
        self.count += batch_size
        if self.mean is None:
            self.mean = torch.zeros_like(tensor[0])
            self.M2 = torch.zeros_like(tensor[0])
        delta_var = tensor - self.mean
        delta_mean = tensor.sum(dim=0).sub_(self.mean * batch_size
                                                ).div_(self.count)
        self.mean.add_(delta_mean)
        delta_var.mul_(tensor - self.mean)
        self.M2 += torch.sum(delta_var, dim=0)


class MeanOnlineLabels:
    """
    Keep track of population mean for each unique class label.

    Parameters
    ----------
    cls : type, optional
        The generator class of online mean: either :class:`MeanOnline` or
        :class`MeanOnlineBatch`.
        Default: MeanOnlineBatch
    """

    def __init__(self, cls=MeanOnlineBatch):
        self.online = defaultdict(cls)
        self.is_active = True

    def __len__(self):
        return len(self.online)

    def activate(self, is_active: bool):
        """
        Activates or deactivates the updates.

        Parameters
        ----------
        is_active : bool
            New state.
        """
        self.is_active = is_active

    def labels(self):
        """
        Returns
        -------
        list
            Unique sorted class labels.
        """
        return sorted(self.online.keys())

    def update(self, tensor, labels):
        """
        Update sample mean (and variance) from a batch of new values, split
        by labels.

        Parameters
        ----------
        tensor : (B, V) torch.Tensor
            A tensor sample.
        labels : (B,) torch.Tensor
            Batch labels.
        """
        if not self.is_active:
            return
        for label in labels.unique(sorted=False):
            self.online[label.item()].update(tensor[labels == label])

    def get_mean_labels(self):
        """
        Returns
        -------
        mean_sorted : (C, V) torch.Tensor
            Mean tensor for each of `C` unique class labels.
        labels_sorted : (C,) torch.Tensor
            Class labels, associated with `mean_sorted`.
        """
        if len(self) == 0:
            # no updates yet
            return None, None
        labels_sorted = self.labels()
        mean_sorted = [self.online[label].get_mean() for label in
                       labels_sorted]
        mean_sorted = torch.stack(mean_sorted, dim=0)
        return mean_sorted, labels_sorted

    def get_mean(self):
        """
        Returns
        -------
        mean_sorted : (C, V) torch.Tensor
            Mean tensor for each of `C` unique class labels.
        """
        mean_sorted, _ = self.get_mean_labels()
        return mean_sorted

    def reset(self):
        """
        Reset the mean and the count.
        """
        self.online.clear()


class VarianceOnlineLabels(MeanOnlineLabels):
    """
    Keep track of population mean and std for each unique class label.
    """

    def __init__(self):
        super().__init__(cls=VarianceOnlineBatch)

    def get_mean_std_labels(self, unbiased=True):
        """
        Return the mean and std for each unique label individually.

        Parameters
        ----------
        unbiased : bool, optional
            Biased (False) or unbiased (True) variance estimate.
            Default: True

        Returns
        -------
        mean_sorted : (C, V) torch.Tensor
            Mean tensor for each of `C` unique class labels.
        std_sorted : (C, V) torch.Tensor
            Std tensor for each of `C` unique class labels.
        labels_sorted : (C,) torch.Tensor
            Class labels, associated with `mean_sorted` and `std_sorted`.

        """
        if len(self) == 0:
            # no updates yet
            return None, None, None
        labels_sorted = self.labels()
        mean_std = [self.online[label].get_mean_std(unbiased) for label in
                    labels_sorted]
        mean_sorted, std_sorted = zip(*mean_std)
        mean_sorted = torch.stack(mean_sorted, dim=0)
        std_sorted = torch.stack(std_sorted, dim=0)
        return mean_sorted, std_sorted, labels_sorted

    def get_mean_std(self, unbiased=True):
        """
        Return the mean and std for each unique label individually without
        the labels themselves.

        Parameters
        ----------
        unbiased : bool, optional
            Biased (False) or unbiased (True) variance estimate.
            Default: True

        Returns
        -------
        mean_sorted : (C, V) torch.Tensor
            Mean tensor for each of `C` unique class labels.
        std_sorted : (C, V) torch.Tensor
            Std tensor for each of `C` unique class labels.

        """
        mean_sorted, std_sorted, _ = self.get_mean_std_labels(unbiased)
        return mean_sorted, std_sorted
