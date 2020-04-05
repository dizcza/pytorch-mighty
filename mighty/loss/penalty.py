import torch
import torch.nn as nn


class LossPenalty(nn.Module):
    def __init__(self, criterion, lambd=0.5, norm=1):
        super().__init__()
        self.criterion = criterion
        self.lambd = lambd
        self.norm = norm

    def extra_repr(self):
        return f"{self.criterion}, lambd={self.lambd}, norm={self.norm}"

    def forward(self, reconstructed, input, latent):
        loss = self.criterion(reconstructed, input)
        penalty = self.lambd * torch.norm(latent, p=self.norm, dim=1).mean()
        return loss + penalty
