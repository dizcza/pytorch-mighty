import numpy as np
import torch
import torch.nn.functional as F


def compute_distance(input1, input2, metric, dim=1):
    if metric == 'cosine':
        dist = 1 - F.cosine_similarity(input1, input2, dim=dim)
    elif metric == 'l1':
        dist = F.l1_loss(input1, input2, reduction='none').sum(dim=dim)
    elif metric == 'l2':
        dist = F.mse_loss(input1, input2, reduction='none').sum(dim=dim)
    else:
        raise NotImplementedError
    return dist


def exponential_moving_average(array, window: int):
    array = np.asarray(array)
    alpha = 2 / (window + 1.0)
    alpha_rev = 1 - alpha
    n = array.shape[0]

    pows = alpha_rev ** (np.arange(n + 1))

    scale_arr = 1 / pows[:-1]
    offset = array[0] * pows[1:]
    pw0 = alpha * alpha_rev ** (n - 1)

    mult = array * pw0 * scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums * scale_arr[::-1]
    return out


def onehot(y_labels):
    n_classes = len(y_labels.unique(sorted=False))
    y_onehot = torch.zeros(y_labels.shape[0], n_classes, dtype=torch.int64)
    y_onehot[torch.arange(y_onehot.shape[0]), y_labels] = 1
    return y_onehot
