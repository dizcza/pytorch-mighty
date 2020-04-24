from typing import Union

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer

from mighty.monitor.accuracy import AccuracyArgmax
from mighty.utils.common import batch_to_cuda
from mighty.utils.data import DataLoader
from .trainer import Trainer


class TrainerGrad(Trainer):
    """
    Default gradient descent trainer with full float precision.
    """

    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 data_loader: DataLoader,
                 optimizer: Optimizer,
                 scheduler: Union[_LRScheduler, ReduceLROnPlateau] = None,
                 **kwargs):
        """
        :param model: NN model
        :param criterion: loss function
        :param dataset_name: one of "MNIST", "CIFAR10", "Caltech256"
        :param optimizer: gradient-based optimizer (SGD, Adam)
        :param scheduler: learning rate scheduler
        """
        super().__init__(model, criterion=criterion, data_loader=data_loader, **kwargs)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self._labels = {
            "predicted": [],
            "true": []
        }

    def monitor_functions(self):
        super().monitor_functions()

        def learning_rate(viz):
            viz.line_update(y=[group['lr'] for group in self.optimizer.param_groups], opts=dict(
                xlabel='Epoch',
                ylabel='Learning rate',
                title='Learning rate',
                ytype='log',
            ))

        if self.scheduler is not None:
            self.monitor.register_func(learning_rate)

    def log_trainer(self):
        super().log_trainer()
        optimizer_str = f"Optimizer {self.optimizer.__class__.__name__}:"
        for group_id, group in enumerate(self.optimizer.param_groups):
            optimizer_str += f"\n\tgroup {group_id}: lr={group['lr']}, weight_decay={group['weight_decay']}"
        self.monitor.log(optimizer_str)

    def train_batch(self, batch):
        self.optimizer.zero_grad()
        outputs = self._forward(batch)
        loss = self._get_loss(batch, outputs)
        loss.backward()
        self.optimizer.step(closure=None)
        return loss

    def _on_forward_pass_batch(self, batch, output):
        if not self.data_loader.has_labels:
            # unsupervised, no labels
            return
        _, labels = batch
        self._labels['true'].append(labels)
        if isinstance(self.accuracy_measure, AccuracyArgmax):
            # softmax
            predicted = self.accuracy_measure.predict(output)
            self._labels['predicted'].append(predicted)
        self.accuracy_measure.partial_fit(output, labels)

    def _get_loss(self, batch, output):
        input, labels = batch
        return self.criterion(output, labels)

    def update_accuracy(self):
        if len(self._labels['true']) == 0:
            # unsupervised, no labels
            return
        labels_full = torch.cat(self._labels['true'], dim=0)

        if len(self._labels['predicted']) > 0:
            # softmax
            labels_pred = torch.cat(self._labels['predicted'], dim=0)
        elif getattr(self.accuracy_measure, 'cache', False):
            labels_pred = self.accuracy_measure.predict_cached()
        else:
            labels_pred = []
            with torch.no_grad():
                for batch in self.data_loader.eval():
                    batch = batch_to_cuda(batch)
                    output = self._forward(batch)
                    labels_pred.append(self.accuracy_measure.predict(output))
            labels_pred = torch.cat(labels_pred, dim=0)

        self.monitor.update_accuracy_epoch(labels_pred, labels_full,
                                           mode='train')

    def _epoch_finished(self, loss):
        self.update_accuracy()
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(metrics=loss)
        elif isinstance(self.scheduler, _LRScheduler):
            self.scheduler.step()
        self._labels['true'].clear()
        self._labels['predicted'].clear()
        super()._epoch_finished(loss)

    def state_dict(self):
        state = super().state_dict()
        state['optimizer'] = self.optimizer.state_dict()
        state['criterion'] = self.criterion.state_dict()
        if self.scheduler is not None:
            state['scheduler'] = self.scheduler.state_dict()
        return state

    def restore(self, checkpoint_path=None, strict=True):
        checkpoint_state = super().restore(checkpoint_path=checkpoint_path,
                                           strict=strict)
        try:
            if checkpoint_state is not None:
                self.optimizer.load_state_dict(checkpoint_state['optimizer'])
                self.criterion.load_state_dict(checkpoint_state['criterion'])
                scheduler_state = checkpoint_state.get('scheduler')
                if self.scheduler is not None and scheduler_state is not None:
                    self.scheduler.load_state_dict(scheduler_state)
        except Exception as exception:
            print("Couldn't restore the trained state: ", exception)
        return checkpoint_state
