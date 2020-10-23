from collections import defaultdict

import torch
import torch.utils.data


class MeanOnline:
    """
    Online updating sample mean.
    Works with scalars, vectors, and n-dimensional tensors.
    """

    def __init__(self, tensor=None):
        self.mean = None
        self.count = 0
        self.is_active = True
        if tensor is not None:
            self.update(new_tensor=tensor)

    def activate(self, is_active: bool):
        self.is_active = is_active

    def update(self, new_tensor):
        if not self.is_active:
            return
        self.count += 1
        if self.mean is None:
            self.mean = new_tensor.clone()
        else:
            self.mean += (new_tensor - self.mean) / self.count

    def get_mean(self) -> torch.Tensor:
        if self.mean is None:
            return None
        else:
            return self.mean.clone()

    def reset(self):
        self.mean = None
        self.count = 0


class VarianceOnline(MeanOnline):
    """
    Welford's online algorithm of estimating population mean and variance.
    """

    def __init__(self, tensor=None):
        self.M2 = None
        super().__init__(tensor)

    def update(self, new_tensor):
        if not self.is_active:
            return
        self.count += 1
        if self.mean is None:
            self.mean = torch.zeros_like(new_tensor)
            self.M2 = torch.zeros_like(new_tensor)
        delta_var = new_tensor - self.mean
        self.mean += delta_var / self.count
        delta_var.mul_(new_tensor - self.mean)
        self.M2 += delta_var

    def get_mean_std(self, unbiased=True):
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
    Updates 1d vector mean from a batch of vectors (2d tensor).
    """

    def update(self, new_tensor):
        if not self.is_active:
            return
        batch_size = new_tensor.shape[0]
        self.count += batch_size
        if self.mean is None:
            self.mean = new_tensor.mean(dim=0)
        else:
            self.mean += (new_tensor.sum(dim=0) -
                          self.mean * batch_size) / self.count


class SumOnlineBatch:
    def __init__(self):
        self.sum = None
        self.count = 0
        self.is_active = True

    def activate(self, is_active):
        self.is_active = is_active

    def update(self, new_tensor: torch.Tensor):
        if not self.is_active:
            return
        self.count += new_tensor.shape[0]
        if self.sum is None:
            self.sum = new_tensor.sum(dim=0)
        else:
            self.sum += new_tensor.sum(dim=0)

    def get_sum(self):
        if self.sum is None:
            return None
        return self.sum.clone()

    def reset(self):
        self.sum = None
        self.count = 0


class VarianceOnlineBatch(VarianceOnline):
    """
    Welford's online algorithm of estimating population mean and variance.
    """

    def update(self, new_tensor):
        if not self.is_active:
            return
        batch_size = new_tensor.shape[0]
        self.count += batch_size
        if self.mean is None:
            self.mean = torch.zeros_like(new_tensor[0])
            self.M2 = torch.zeros_like(new_tensor[0])
        delta_var = new_tensor - self.mean
        delta_mean = new_tensor.sum(dim=0).sub_(self.mean * batch_size
                                                ).div_(self.count)
        self.mean.add_(delta_mean)
        delta_var.mul_(new_tensor - self.mean)
        self.M2 += torch.sum(delta_var, dim=0)


class MeanOnlineLabels:

    def __init__(self, cls=MeanOnlineBatch):
        self.online = defaultdict(cls)
        self.is_active = True

    def __len__(self):
        return len(self.online)

    def activate(self, is_active: bool):
        self.is_active = is_active

    def labels(self):
        return sorted(self.online.keys())

    def update(self, new_tensor, labels):
        if not self.is_active:
            return
        # new_tensor:  (B, V)
        # labels: (B,)
        for label in labels.unique(sorted=False):
            self.online[label.item()].update(new_tensor[labels == label])

    def get_mean_labels(self):
        if len(self) == 0:
            # no updates yet
            return None, None
        labels_sorted = self.labels()
        mean_sorted = [self.online[label].get_mean() for label in
                       labels_sorted]
        mean_sorted = torch.stack(mean_sorted, dim=0)
        return mean_sorted, labels_sorted

    def get_mean(self):
        mean_sorted, _ = self.get_mean_labels()
        return mean_sorted

    def reset(self):
        self.online.clear()


class VarianceOnlineLabels(MeanOnlineLabels):

    def __init__(self):
        super().__init__(cls=VarianceOnlineBatch)

    def get_mean_std_labels(self, unbiased=True):
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
        mean_sorted, std_sorted, _ = self.get_mean_std_labels(unbiased)
        return mean_sorted, std_sorted
