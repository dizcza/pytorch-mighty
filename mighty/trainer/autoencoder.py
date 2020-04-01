from typing import Union

import torch
import torch.nn as nn
import torch.utils.data
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer

from mighty.monitor import MonitorAutoenc
from mighty.monitor.accuracy import Accuracy, AccuracyAutoencoder
from mighty.monitor.var_online import MeanOnline
from mighty.utils.algebra import compute_psnr
from mighty.utils.data import DataLoader, get_normalize_inverse
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
        normalize_inverse = get_normalize_inverse(self.data_loader.normalize)
        monitor = MonitorAutoenc(
            accuracy_measure=self.accuracy_measure,
            mutual_info=mutual_info,
            normalize_inverse=normalize_inverse
        )
        return monitor

    def _init_online_measures(self):
        online = super()._init_online_measures()
        online['psnr'] = MeanOnline()  # peak signal-to-noise ratio
        return online

    def _get_loss(self, input, output, labels):
        latent, reconstructed = output
        return self.criterion(reconstructed, input)

    def _on_forward_pass_batch(self, input, output, labels):
        latent, reconstructed = output
        if isinstance(self.criterion, nn.BCEWithLogitsLoss):
            reconstructed = reconstructed.sigmoid()
        psnr = compute_psnr(input, reconstructed)
        self.online['psnr'].update(psnr.cpu())
        super()._on_forward_pass_batch(input, latent, labels)

    def _epoch_finished(self, epoch, loss):
        self.plot_autoencoder()
        self.monitor.plot_psnr(self.online['psnr'].get_mean())
        super()._epoch_finished(epoch, loss)

    def plot_autoencoder(self):
        input, labels = next(iter(self.data_loader.eval))
        if torch.cuda.is_available():
            input = input.cuda()
        mode_saved = self.model.training
        self.model.train(False)
        with torch.no_grad():
            latent, reconstructed = self.model(input)
        if isinstance(self.criterion, nn.BCEWithLogitsLoss):
            reconstructed = reconstructed.sigmoid()
        self.monitor.plot_autoencoder(input, reconstructed)
        self.model.train(mode_saved)
