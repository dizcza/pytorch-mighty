import torch
import torch.nn as nn


class LossPenalty(nn.Module):
    def __init__(self, criterion, lambd=0.5, norm=1, latent_grad=False):
        super().__init__()
        self.criterion = criterion
        self.lambd = lambd
        self.norm = norm
        self.latent_grad = latent_grad

    def extra_repr(self):
        # self.criterion will be populated here by pytorch
        return f"lambd={self.lambd}, norm={self.norm}, " \
               f"latent_grad={self.latent_grad}"

    def forward(self, reconstructed, input, latent):
        assert latent.ndim == 2
        loss = self.criterion(reconstructed, input)
        if not self.latent_grad:
            latent = latent.detach()
        lambd = self.lambd
        if isinstance(lambd, torch.Tensor):
            # the penalty coefficient must be non-negative
            lambd = lambd.relu().mean()

        # the norm is divided by the dimensionality of the embedding space;
        # this allows us to represent 'lambd' as the penalty relative to
        # the main cost.
        l1_norm = latent.norm(p=self.norm, dim=1).mean() / latent.shape[1]

        penalty = lambd * l1_norm
        return loss + penalty
