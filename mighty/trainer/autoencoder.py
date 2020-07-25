from typing import Union

import torch
import torch.nn as nn
import torch.utils.data
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer

from mighty.loss import LossPenalty
from mighty.monitor.monitor import MonitorAutoencoder
from mighty.monitor.var_online import MeanOnline
from mighty.utils.algebra import compute_psnr
from mighty.utils.common import input_from_batch, batch_to_cuda
from mighty.utils.data import DataLoader
from .embedding import TrainerEmbedding


class TrainerAutoencoder(TrainerEmbedding):

    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 data_loader: DataLoader,
                 optimizer: Optimizer,
                 scheduler: Union[_LRScheduler, ReduceLROnPlateau] = None,
                 **kwargs):
        super().__init__(model, criterion=criterion, data_loader=data_loader,
                         optimizer=optimizer, scheduler=scheduler, **kwargs)

    def _init_monitor(self, mutual_info) -> MonitorAutoencoder:
        monitor = MonitorAutoencoder(
            accuracy_measure=self.accuracy_measure,
            mutual_info=mutual_info,
            normalize_inverse=self.data_loader.normalize_inverse
        )
        return monitor

    def _init_online_measures(self):
        online = super()._init_online_measures()

        # peak signal-to-noise ratio
        online['psnr-train'] = MeanOnline()
        online['psnr-test'] = MeanOnline()

        return online

    def _get_loss(self, batch, output):
        input = input_from_batch(batch)
        latent, reconstructed = output
        if isinstance(self.criterion, LossPenalty):
            loss = self.criterion(reconstructed, input, latent)
        else:
            loss = self.criterion(reconstructed, input)
        return loss

    def _on_forward_pass_batch(self, batch, output, train):
        input = input_from_batch(batch)
        latent, reconstructed = output
        if isinstance(self.criterion, nn.BCEWithLogitsLoss):
            reconstructed = reconstructed.sigmoid()
        psnr = compute_psnr(input, reconstructed)
        fold = 'train' if train else 'test'
        if torch.isfinite(psnr):
            self.online[f'psnr-{fold}'].update(psnr.cpu())
        super()._on_forward_pass_batch(batch, latent, train)

    def _epoch_finished(self, loss):
        self.plot_autoencoder()
        for fold in ('train', 'test'):
            self.monitor.plot_psnr(self.online[f'psnr-{fold}'].get_mean(),
                                   mode=fold)
        super()._epoch_finished(loss)

    def plot_autoencoder(self):
        # plot AutoEncoder reconstruction
        batch = self.data_loader.sample()
        batch = batch_to_cuda(batch)
        mode_saved = self.model.training
        self.model.train(False)
        with torch.no_grad():
            latent, reconstructed = self._forward(batch)
        if isinstance(self.criterion, nn.BCEWithLogitsLoss):
            reconstructed = reconstructed.sigmoid()
        self._plot_autoencoder(batch, reconstructed)
        self.model.train(mode_saved)

    def _plot_autoencoder(self, batch, reconstructed, mode='train'):
        input = input_from_batch(batch)
        self.monitor.plot_autoencoder(input, reconstructed, mode=mode)
