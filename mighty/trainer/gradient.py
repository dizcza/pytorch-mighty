from typing import Union

import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer

from mighty.utils.data import DataLoader
from .trainer import Trainer


__all__ = [
    "TrainerGrad"
]


class TrainerGrad(Trainer):
    """
    The default gradient descent trainer.

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
    **kwargs
        Passed to the base class.
    """

    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 data_loader: DataLoader,
                 optimizer: Optimizer,
                 scheduler: Union[_LRScheduler, ReduceLROnPlateau] = None,
                 **kwargs):
        super().__init__(model, criterion=criterion, data_loader=data_loader, **kwargs)
        self.optimizer = optimizer
        self.scheduler = scheduler

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

    def _epoch_finished(self, loss):
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(metrics=loss)
        elif isinstance(self.scheduler, _LRScheduler):
            self.scheduler.step()
        super()._epoch_finished(loss)

    def state_dict(self):
        state = super().state_dict()
        state['optimizer'] = self.optimizer.state_dict()
        state['criterion'] = self.criterion.state_dict()
        if self.scheduler is not None:
            state['scheduler'] = self.scheduler.state_dict()
        return state

    def restore(self, checkpoint_path=None, best=False, strict=True):
        checkpoint_state = super().restore(checkpoint_path, best=best,
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
