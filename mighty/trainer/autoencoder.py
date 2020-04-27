from typing import Union

import torch
import torch.nn as nn
import torch.utils.data
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer

from mighty.loss import LossPenalty
from mighty.monitor import MonitorAutoenc
from mighty.monitor.accuracy import Accuracy, AccuracyAutoencoder
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
                 accuracy_measure: Accuracy = AccuracyAutoencoder(),
                 **kwargs):
        if not isinstance(accuracy_measure, AccuracyAutoencoder):
            raise ValueError("'accuracy_measure' must be of instance "
                             f"{AccuracyAutoencoder.__name__}")
        super().__init__(model, criterion=criterion, data_loader=data_loader,
                         optimizer=optimizer, scheduler=scheduler,
                         accuracy_measure=accuracy_measure, **kwargs)

    def _init_monitor(self, mutual_info) -> MonitorAutoenc:
        monitor = MonitorAutoenc(
            accuracy_measure=self.accuracy_measure,
            mutual_info=mutual_info,
            normalize_inverse=self.data_loader.normalize_inverse
        )
        return monitor

    def _init_online_measures(self):
        online = super()._init_online_measures()
        online['psnr'] = MeanOnline()  # peak signal-to-noise ratio
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
        if not train:
            super()._on_forward_pass_batch(batch, output, train)
            return
        input = input_from_batch(batch)
        latent, reconstructed = output
        if isinstance(self.criterion, nn.BCEWithLogitsLoss):
            reconstructed = reconstructed.sigmoid()
        psnr = compute_psnr(input, reconstructed)
        if torch.isfinite(psnr):
            self.online['psnr'].update(psnr.cpu())
        super()._on_forward_pass_batch(batch, latent, train)

    def _epoch_finished(self, loss):
        self.plot_autoencoder()
        self.monitor.plot_psnr(self.online['psnr'].get_mean())
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

    def _plot_autoencoder(self, batch, reconstructed):
        input = input_from_batch(batch)
        self.monitor.plot_autoencoder(input, reconstructed)
