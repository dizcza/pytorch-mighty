from typing import Union, Optional

import torch.nn as nn
import torch.utils.data
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from models.kwta import KWinnersTakeAllSoft, KWinnersTakeAll, SynapticScaling
from trainer.gradient import TrainerGrad
from trainer.mask import MaskTrainerIndex
from utils.layers import find_layers, find_named_layers
from utils.prepare import prepare_eval


class KWTAScheduler:
    """
    KWinnersTakeAll sparsity scheduler.
    """

    def __init__(self, model: nn.Module, step_size: int, gamma_sparsity=0.5, min_sparsity=0.05,
                 gamma_hardness=2.0, max_hardness=10):
        self.kwta_layers = tuple(find_layers(model, layer_class=KWinnersTakeAll))
        self.step_size = step_size
        self.gamma_sparsity = gamma_sparsity
        self.min_sparsity = min_sparsity
        self.gamma_hardness = gamma_hardness
        self.max_hardness = max_hardness
        self.epoch = 0
        self.last_epoch_update = -1

    def need_update(self):
        return self.epoch >= self.last_epoch_update + self.step_size

    def step(self, epoch=None):
        if epoch is not None:
            self.epoch = epoch
        if self.need_update():
            for layer in self.kwta_layers:
                layer.clear_lateral()
                layer.sparsity = max(layer.sparsity * self.gamma_sparsity, self.min_sparsity)
                if isinstance(layer, KWinnersTakeAllSoft):
                    layer.hardness = min(layer.hardness * self.gamma_hardness, self.max_hardness)
            self.last_epoch_update = self.epoch
        self.epoch += 1

    def extra_repr(self):
        return f"step_size={self.step_size}, Sparsity(gamma={self.gamma_sparsity}, min={self.min_sparsity}), " \
               f"Hardness(gamma={self.gamma_hardness}, max={self.max_hardness})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.extra_repr()})"


class TrainerGradKWTA(TrainerGrad):
    watch_modules = TrainerGrad.watch_modules + (KWinnersTakeAll, SynapticScaling)

    """
    If the model does not have any KWinnerTakeAll layers, it works just as TrainerGrad.
    """

    def __init__(self, model: nn.Module, criterion: nn.Module, dataset_name: str,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Union[_LRScheduler, ReduceLROnPlateau, None] = None,
                 kwta_scheduler: Optional[KWTAScheduler] = None,
                 **kwargs):
        super().__init__(model=model, criterion=criterion, dataset_name=dataset_name, optimizer=optimizer,
                         scheduler=scheduler, **kwargs)
        self.kwta_scheduler = kwta_scheduler
        self.mask_trainer_kwta = MaskTrainerIndex(image_shape=self.mask_trainer.image_shape)

    def monitor_functions(self):
        super().monitor_functions()
        kwta_named_layers = tuple(find_named_layers(self.model, layer_class=KWinnersTakeAll))

        def lateral_weights(viz):
            for name, layer in kwta_named_layers:
                if layer.weight_lateral is not None:
                    title = f"kWTA {name}.lateral_weights"
                    viz.heatmap(layer.weight_lateral, win=title, opts=dict(
                        xlabel='Neuron id',
                        ylabel='Neuron id',
                        title=title,
                    ))

        self.monitor.register_func(lateral_weights)

        if self.kwta_scheduler is not None:
            kwta_named_layers_soft = tuple(find_named_layers(self.model, layer_class=KWinnersTakeAllSoft))

            def sparsity(viz):
                layers_sparsity = []
                names = []
                for name, layer in kwta_named_layers:
                    layers_sparsity.append(layer.sparsity)
                    names.append(name)
                viz.line_update(y=layers_sparsity, opts=dict(
                    xlabel='Epoch',
                    ylabel='sparsity',
                    title='KWinnersTakeAll.sparsity',
                    legend=names,
                    ytype='log',
                ))

            def hardness(viz):
                layers_hardness = []
                names = []
                for name, layer in kwta_named_layers_soft:
                    layers_hardness.append(layer.hardness)
                    names.append(name)
                viz.line_update(y=layers_hardness, opts=dict(
                    xlabel='Epoch',
                    ylabel='hardness',
                    title='KWinnersTakeAllSoft.hardness',
                    legend=names,
                    ytype='log',
                ))

            self.monitor.register_func(sparsity, hardness)

        synaptic_scale_named_layers = tuple(find_named_layers(self.model, SynapticScaling))
        if len(synaptic_scale_named_layers) > 0:

            def monitor_synaptic_scaling(viz):
                rownames = []
                frequency = []
                for name, layer in synaptic_scale_named_layers:
                    rownames.append(name)
                    frequency.append(layer.frequency)
                frequency = torch.stack(frequency, dim=0)
                title = "Synaptic scaling plasticity"
                viz.heatmap(frequency, win=title, opts=dict(
                    xlabel="Embedding dimension",
                    ylabel="Neuron fires frequency",
                    title=title,
                    rownames=rownames,
                ))

            self.monitor.register_func(monitor_synaptic_scaling)

    def log_trainer(self):
        super().log_trainer()
        self.monitor.log(f"KWTA scheduler: {self.kwta_scheduler}")

    @property
    def env_name(self) -> str:
        env_name = super().env_name
        if not any(find_layers(self.model, layer_class=KWinnersTakeAll)):
            env_name += " (TrainerGrad)"
        return env_name

    def _epoch_finished(self, epoch, outputs, labels):
        loss = super()._epoch_finished(epoch, outputs, labels)
        if self.kwta_scheduler is not None:
            self.kwta_scheduler.step(epoch=epoch)
        return loss

    def train_mask(self):
        image, label = super().train_mask()
        mode_saved = prepare_eval(self.model)
        with torch.no_grad():
            output = self.model(image.unsqueeze(dim=0))
        output_sorted, argsort = output[0].sort(dim=0, descending=True)
        neurons_check = min(5, len(argsort))
        for i in range(neurons_check):
            neuron_max = argsort[i]
            self.monitor.plot_mask(self.model, mask_trainer=self.mask_trainer_kwta, image=image, label=neuron_max,
                                   win_suffix=i)
        mode_saved.restore(self.model)
        return image, label
