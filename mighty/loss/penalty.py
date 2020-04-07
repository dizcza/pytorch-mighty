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
        loss = self.criterion(reconstructed, input)
        if not self.latent_grad:
            latent = latent.detach()
        penalty = self.lambd * torch.norm(latent, p=self.norm, dim=1).mean()
        return loss + penalty
