import torch
import torch.nn as nn
import torch.nn.functional
from tqdm import trange

from mighty.monitor.accuracy import Accuracy, AccuracyArgmax


def tv_norm(mask_expanded, tv_beta: int):
    """
    Mask gradient (approximate) cost.
    """
    # (1, 1, H, W)
    mask = mask_expanded[0, 0, ::]
    row_grad = (mask[:-1, :] - mask[1:, :]).abs().pow(tv_beta).mean()
    col_grad = (mask[:, :-1] - mask[:, 1:]).abs().pow(tv_beta).mean()
    return row_grad + col_grad


def create_gaussian_filter(size: int, sigma: float, channels: int):
    linspace = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    # Create a x, y coordinate grid of shape (size, size, 2)
    x_grid = linspace.repeat(size).view(size, size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    gaussian_kernel = torch.exp(-xy_grid.pow(2).sum(dim=-1) / (2 * sigma ** 2))
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel /= gaussian_kernel.sum()
    gaussian_kernel = gaussian_kernel.expand(channels, 1,
                                             *gaussian_kernel.shape)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=size, groups=channels, bias=False)
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad_(False)
    return gaussian_filter


class MaskTrainer:
    """
    Interpretable Explanations of Black Boxes by Meaningful Perturbation [1]_.

    Train an occlusion mask that shows where a neural network "looks at" in the
    input space.

    Parameters
    ----------
    accuracy_measure : Accuracy
        Accuracy estimator.
    image_shape : tuple
        The shape of an input image.
    learning_rate : float, optional
        Optimizer learning rate.
        Default: 0.1
    show_progress : bool, optional
        Show the training progress bar or not.
        Default: False

    References
    ----------
    .. [1] Fong, R. C., & Vedaldi, A. (2017). Interpretable explanations of
       black boxes by meaningful perturbation. In Proceedings of the IEEE
       International Conference on Computer Vision (pp. 3429-3437).

    """

    tv_beta = 1
    max_iterations = 100
    l1_coeff = 0.01
    tv_coeff = 0.2
    mask_size = 10

    def __init__(self, image_shape, accuracy_measure=AccuracyArgmax(),
                 learning_rate=0.1, show_progress=False):
        self.image_shape = image_shape
        self.accuracy_measure = accuracy_measure
        self.learning_rate = learning_rate
        kernel_size = 2 * int(image_shape[1] ** 0.5 // 2) + 1
        self.gaussian_filter = create_gaussian_filter(size=kernel_size,
                                                      sigma=2*kernel_size,
                                                      channels=image_shape[0])
        self.padding = nn.modules.ReflectionPad2d(padding=kernel_size // 2)
        self.show_progress = show_progress
        if torch.cuda.is_available():
            self.gaussian_filter.cuda()
            self.padding.cuda()

    def train_mask(self, model, image, label_true):
        """
        Train a grayscale occlusion mask.

        Parameters
        ----------
        model : nn.Module
            A neural network model.
        image : (C, H, W) torch.Tensor
            An input image.
        label_true : int
            The true class label of the image.

        Returns
        -------
        mask_upsampled : torch.Tensor
            The occlusion mask.
        image_perturbed : torch.Tensor
            The input image with the mask applied.
        loss_trace : list of float
            A list of training losses.

        """
        channels, height, width = image.shape
        image = image.unsqueeze(dim=0)
        image_blurred = self.gaussian_filter(self.padding(image))

        # 1 - take input pixel
        # 0 - cover with mask
        mask = nn.Parameter(torch.ones(self.mask_size, self.mask_size,
                                       dtype=torch.float32,
                                       device=image.device))

        optimizer = torch.optim.Adam([mask], lr=self.learning_rate)
        loss_trace = []
        mask_upsampled = None
        image_perturbed = None
        for i in trange(self.max_iterations, desc="Training a mask",
                        disable=not self.show_progress, leave=False):
            mask_upsampled = mask.expand(1, channels, *mask.shape)
            mask_upsampled = nn.functional.interpolate(mask_upsampled,
                                                       size=(height, width),
                                                       mode='bilinear',
                                                       align_corners=True)
            optimizer.zero_grad()
            noise = torch.randn_like(image) * 0.2
            image_perturbed = mask_upsampled * image + (
                    1 - mask_upsampled) * image_blurred
            outputs = model(image_perturbed + noise)
            proba = self.get_probability(outputs=outputs, label=label_true)
            loss = self.l1_coeff * (1 - mask_upsampled).abs().mean() + \
                   self.tv_coeff * tv_norm(mask_upsampled, self.tv_beta) + \
                   proba
            loss.backward()
            optimizer.step()
            mask_upsampled.data.clamp_(0, 1)
            loss_trace.append(loss.item())
        mask_upsampled = mask_upsampled[0].detach()
        image_perturbed = image_perturbed[0].detach()
        return mask_upsampled, image_perturbed, loss_trace

    def get_probability(self, outputs, label):
        """
        Returns the probability of the `label` class of the outputs.

        Parameters
        ----------
        outputs : torch.Tensor
            The output of a model.
        label : int
            The true class label.

        Returns
        -------
        float
            The probability of ``outputs[label]``.

        """
        proba = self.accuracy_measure.predict_proba(outputs)[0, label]
        return proba

    def __repr__(self):
        return f"{self.__class__.__name__}(mask_size={self.mask_size}, " \
               f"gaussian_filter={self.gaussian_filter})"
