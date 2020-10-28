"""
Signal processing and statistics.

.. autosummary::
    :toctree: toctree/utils/

    exponential_moving_average
    peak_to_signal_noise_ratio
    compute_sparsity

"""

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


def exponential_moving_average(tensor, window: int):
    """
    Exponential moving average in a sliding window.

    Parameters
    ----------
    tensor : (N,) torch.Tensor
        Input tensor.
    window : int
        Sliding window width.

    Returns
    -------
    out : (N,) torch.Tensor
        Filtered array of the same length.
    """
    tensor = torch.as_tensor(tensor)
    alpha = 2 / (window + 1.0)
    alpha_rev = 1 - alpha
    n = tensor.shape[0]

    pows = torch.pow(alpha_rev, torch.arange(n + 1))

    scale_arr = 1 / pows[:-1]
    offset = tensor[0] * pows[1:]
    pw0 = alpha * alpha_rev ** (n - 1)

    mult = tensor * pw0 * scale_arr
    cumsums = mult.cumsum(dim=0)
    out = offset + cumsums * reversed(scale_arr)
    return out


def to_onehot(y_labels, n_classes=None):
    y_labels = torch.as_tensor(y_labels, dtype=torch.long)
    if n_classes is None:
        n_classes = len(y_labels.unique(sorted=False))
    y_onehot = torch.zeros(y_labels.shape[0], n_classes, dtype=torch.int64)
    y_onehot[torch.arange(y_onehot.shape[0]), y_labels] = 1
    return y_onehot


def peak_to_signal_noise_ratio(signal_orig, signal_estimated):
    """
    Computes the Peak signal-to-noise ratio between two signals.

    Parameters
    ----------
    signal_orig, signal_estimated : torch.Tensor
        A vector or a batch of vectors or images.

    Returns
    -------
    psnr : torch.Tensor
        A scalar tensor that holds (mean) PSNR value.

    """
    signal_orig = signal_orig.detach()
    signal_estimated = signal_estimated.detach()
    signal_orig = torch.atleast_2d(signal_orig)
    signal_estimated = torch.atleast_2d(signal_estimated)
    signal_orig = signal_orig.flatten(start_dim=1)
    signal_estimated = signal_estimated.flatten(start_dim=1)
    if signal_orig.shape != signal_estimated.shape:
        raise ValueError("Input signals must have the same shape.")

    dynamic_range = signal_orig.max(dim=1).values - \
                    signal_orig.min(dim=1).values

    # filter out pairs with zero dynamic range
    mask_valid = (dynamic_range != 0.)
    if not mask_valid.any():
        return torch.tensor(float('NaN'), device=signal_orig.device)

    signal_orig = signal_orig[mask_valid]
    signal_estimated = signal_estimated[mask_valid]
    dynamic_range = dynamic_range[mask_valid]

    mse_val = F.mse_loss(signal_orig, signal_estimated,
                         reduction='none').mean(dim=1)
    psnr = 10 * torch.log10(dynamic_range ** 2 / mse_val).mean()
    return psnr.squeeze()


def compute_sparsity(tensor):
    """
    Compute L1 sparsity of the input tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        (N,) or (B, N) tensor array.

    Returns
    -------
    sparsity : torch.Tensor
        Mean L1 sparsity of `tensor`, scalar.

    """
    if tensor.ndim == 1:
        # make a batch of size 1
        tensor = tensor.unsqueeze(dim=0)
    sparsity = tensor.norm(p=1, dim=1).mean() / tensor.shape[1]
    return sparsity.squeeze()
