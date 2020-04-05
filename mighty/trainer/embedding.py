from typing import Union

import torch.nn as nn
import torch.utils.data
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer

from mighty.monitor import MonitorEmbedding
from mighty.monitor.accuracy import Accuracy, AccuracyEmbedding
from mighty.monitor.var_online import MeanOnline, VarianceOnline, VarianceOnlineLabels
from mighty.utils.algebra import compute_sparsity
from mighty.utils.data import DataLoader, get_normalize_inverse
from .gradient import TrainerGrad


class TrainerEmbedding(TrainerGrad):
    """
    Operates on embedding vectors.
    """

    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 data_loader: DataLoader,
                 optimizer: Optimizer,
                 scheduler: Union[_LRScheduler, ReduceLROnPlateau] = None,
                 accuracy_measure: Accuracy = AccuracyEmbedding(),
                 **kwargs):
        """
        :param model: NN model
        :param criterion: loss function
        :param dataset_name: one of "MNIST", "CIFAR10", "Caltech256"
        :param optimizer: gradient-based optimizer (SGD, Adam)
        :param scheduler: learning rate scheduler
        :param kwta_scheduler: kWTA sparsity and hardness scheduler
        """
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
        normalize_inverse = get_normalize_inverse(self.data_loader.normalize)
        monitor = MonitorEmbedding(
            accuracy_measure=self.accuracy_measure,
            mutual_info=mutual_info,
            normalize_inverse=normalize_inverse
        )
        return monitor

    def _init_online_measures(self):
        online = super()._init_online_measures()
        online['sparsity'] = MeanOnline()  # scalar
        online['l1_norm'] = MeanOnline()  # (V,) vector
        online['clusters'] = VarianceOnlineLabels()  # (C, V) tensor
        return online

    def _on_forward_pass_batch(self, input, output, labels):
        super()._on_forward_pass_batch(input, output, labels)
        sparsity = compute_sparsity(output)
        self.online['sparsity'].update(sparsity.cpu())
        self.online['l1_norm'].update(output.abs().mean(dim=0).cpu())
        self.online['clusters'].update(output, labels)

    def _epoch_finished(self, epoch, loss):
        self.monitor.update_sparsity(self.online['sparsity'].get_mean(),
                                     mode='train')
        self.monitor.update_l1_norm(self.online['l1_norm'].get_mean())
        self.monitor.clusters_heatmap(*self.online['clusters'].get_mean_std())
        super()._epoch_finished(epoch, loss)