import torch
import torch.utils.data


class MeanOnline:
    """
    Online updating sample mean.
    """

    def __init__(self, tensor=None):
        self.mean = None
        self.count = 0
        if tensor is not None:
            self.update(new_tensor=tensor)

    def update(self, new_tensor):
        self.count += 1
        if self.mean is None:
            self.mean = new_tensor.clone()
        else:
            self.mean += (new_tensor - self.mean) / self.count

    def get_mean(self):
        if self.mean is None:
            return None
        else:
            return self.mean.clone()

    def reset(self):
        self.mean = None
        self.count = 0


class MeanOnlineBatch(MeanOnline):

    def update(self, new_tensor):
        batch_size = new_tensor.shape[0]
        self.count += batch_size
        if self.mean is None:
            self.mean = new_tensor.mean(dim=0)
        else:
            self.mean += (new_tensor.sum(dim=0) -
                          self.mean * batch_size) / self.count


class VarianceOnline(MeanOnline):
    """
    Welford's online algorithm of estimating population mean and variance.
    """

    def __init__(self, tensor=None):
        self.var = None
        super().__init__(tensor)

    def update(self, new_tensor):
        super().update(new_tensor)
        if self.var is None:
            self.var = torch.zeros_like(self.mean)
        else:
            # todo check correctness
            self.var = (self.count - 2) / (self.count - 1) * self.var + \
                       torch.pow(new_tensor - self.mean, 2) / self.count

    def get_std(self):
        if self.var is None:
            return None
        else:
            return torch.sqrt(self.var)

    def get_mean_std(self):
        return self.get_mean(), self.get_std()

    def reset(self):
        super().reset()
        self.var = None
