from typing import Union

import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer

from mighty.monitor.accuracy import Accuracy, AccuracyEmbedding
from mighty.monitor.monitor import MonitorEmbedding
from mighty.utils.var_online import MeanOnline, VarianceOnlineLabels
from mighty.utils.signal import compute_sparsity
from mighty.utils.data import DataLoader
from .gradient import TrainerGrad


__all__ = [
    "TrainerEmbedding"
]


class TrainerEmbedding(TrainerGrad):
    """
    An (unsupervised) trainer that transforms input data into
    linearly-separable embedding vectors that form clusters.

    Parameters
    ----------
    model : nn.Module
        A neural network to train.
    criterion : nn.Module
        A loss function.
    data_loader : DataLoader
        A data loader.
    optimizer : Optimizer
        An optimizer (Adam, SGD, etc.).
    scheduler : _LRScheduler or ReduceLROnPlateau, or None
        A learning rate scheduler.
    accuracy_measure : AccuracyEmbedding
        Calculates the accuracy of embedding vectors.
        Default: AccuracyEmbedding()
    **kwargs
        Passed to the base class.
    """

    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 data_loader: DataLoader,
                 optimizer: Optimizer,
                 scheduler: Union[_LRScheduler, ReduceLROnPlateau] = None,
                 accuracy_measure: Accuracy = AccuracyEmbedding(),
                 **kwargs):
        if not isinstance(accuracy_measure, AccuracyEmbedding):
            raise ValueError("'accuracy_measure' must be of instance "
                             f"{AccuracyEmbedding.__name__}")
        super().__init__(model=model,
                         criterion=criterion,
                         data_loader=data_loader,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         accuracy_measure=accuracy_measure,
                         **kwargs)

    def _init_monitor(self, mutual_info):
        monitor = MonitorEmbedding(
            mutual_info=mutual_info,
            normalize_inverse=self.data_loader.normalize_inverse
        )
        return monitor

    def _init_online_measures(self):
        online = super()._init_online_measures()
        online['sparsity'] = MeanOnline()  # scalar
        online['l1_norm'] = MeanOnline()  # (V,) vector
        online['clusters'] = VarianceOnlineLabels()  # (C, V) tensor
        return online

    def _on_forward_pass_batch(self, batch, output, train):
        if train:
            sparsity = compute_sparsity(output)
            self.online['sparsity'].update(sparsity.cpu())
            self.online['l1_norm'].update(output.abs().mean(dim=0).cpu())
            if self.data_loader.has_labels:
                # supervised
                input, labels = batch
                self.online['clusters'].update(output, labels)
        super()._on_forward_pass_batch(batch, output, train)

    def _epoch_finished(self, loss):
        self.monitor.update_sparsity(self.online['sparsity'].get_mean(),
                                     mode='train')
        self.monitor.update_l1_neuron_norm(self.online['l1_norm'].get_mean())
        # mean and std can be Nones
        mean, std = self.online['clusters'].get_mean_std()
        self.monitor.clusters_heatmap(mean=mean, std=std)
        self.monitor.embedding_hist(activations=mean)
        super()._epoch_finished(loss)
