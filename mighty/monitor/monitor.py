from collections import UserDict

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from sklearn.metrics import confusion_matrix
from typing import Callable, Optional

from mighty.monitor.accuracy import calc_accuracy, Accuracy
from mighty.monitor.batch_timer import timer, ScheduleExp
from mighty.monitor.mutual_info.stub import MutualInfoStub
from mighty.monitor.var_online import VarianceOnline
from mighty.monitor.viz import VisdomMighty
from mighty.utils.common import clone_cpu
from mighty.utils.domain import AdversarialExamples, MonitorLevel


class ParamRecord:
    """
    Tracks gradient variance, sign flips of param's data.
    ParamRecords are created by Monitor.
    """

    def __init__(self, param: nn.Parameter,
                 monitor_level: MonitorLevel = MonitorLevel.DISABLED):
        self.param = param
        self.monitor_level = monitor_level
        self.grad_variance = VarianceOnline()
        self.variance = VarianceOnline()
        self.prev_sign = None
        self.initial_data = None
        self.initial_norm = param.data.norm(p=2).item()

    def update_signs(self) -> float:
        param = self.param
        new_data = clone_cpu(param.data)
        if self.prev_sign is None:
            self.prev_sign = new_data
        sign_flips = (new_data * self.prev_sign < 0).sum().item()
        self.prev_sign = new_data
        return sign_flips

    def update_grad_variance(self):
        if self.param.grad is not None:
            self.grad_variance.update(self.param.grad.data.cpu())

    def reset(self):
        self.grad_variance.reset()
        self.variance.reset()


class ParamsDict(UserDict):
    def __init__(self):
        super().__init__()
        self.sign_flips = 0
        self.n_updates = 0

    def batch_finished(self):
        def filter_ge(level: MonitorLevel):
            # filter by greater or equal to the Monitor level
            return (precord for precord in self.values()
                    if precord.monitor_level.value >= level.value)

        self.n_updates += 1
        for param_record in filter_ge(MonitorLevel.SIGNAL_TO_NOISE):
            param_record.update_grad_variance()
        for param_record in filter_ge(MonitorLevel.FULL):
            self.sign_flips += param_record.update_signs()
            param_record.variance.update(param_record.param.data.cpu())

    def plot_sign_flips(self, viz: VisdomMighty):
        viz.line_update(y=self.sign_flips / self.n_updates, opts=dict(
            xlabel='Epoch',
            ylabel='Sign flips',
            title="Sign flips after optimizer.step()",
        ))
        self.sign_flips = 0
        self.n_updates = 0


class Monitor:
    n_classes_format_ytickstep_1 = 10

    def __init__(self, accuracy_measure: Accuracy, mutual_info=None,
                 normalize_inverse=None):
        """
        :param accuracy_measure: argmax or centroid embeddings accuracy measure
        :param mutual_info: mutual information estimator
        """
        self.timer = timer
        self.viz = None
        self._advanced_monitoring_level = MonitorLevel.DISABLED
        self.accuracy_measure = accuracy_measure
        self.normalize_inverse = normalize_inverse
        self.param_records = ParamsDict()
        if mutual_info is None:
            mutual_info = MutualInfoStub()
        self.mutual_info = mutual_info
        self.functions = []

    @property
    def is_active(self):
        return self.viz is not None

    def advanced_monitoring(self, level: MonitorLevel = MonitorLevel.DISABLED):
        # Note: advanced monitoring is memory consuming
        self._advanced_monitoring_level = level
        for param_record in self.param_records.values():
            param_record.monitor_level = level

    def open(self, env_name: str):
        """
        :param env_name: Visdom environment name
        """
        self.viz = VisdomMighty(env=env_name)

    def log_model(self, model: nn.Module, space='-'):
        lines = []
        for line in repr(model).splitlines():
            n_spaces = len(line) - len(line.lstrip())
            line = space * n_spaces + line
            lines.append(line)
        lines = '<br>'.join(lines)
        self.log(lines)

    def log_self(self):
        self.log(f"{self.__class__.__name__}("
                 f"level={self._advanced_monitoring_level})")

    def log(self, text: str):
        self.viz.log(text)

    def batch_finished(self, model: nn.Module):
        self.param_records.batch_finished()
        self.timer.tick()
        if self.timer.epoch == 0:
            self.batch_finished_first_epoch(model)

    @ScheduleExp()
    def batch_finished_first_epoch(self, model):
        # inspect the very beginning of the training progress
        self.mutual_info.force_update(model)
        self.update_mutual_info()
        self.update_gradient_signal_to_noise_ratio()

    def update_loss(self, loss: Optional[torch.Tensor], mode='batch'):
        if loss is None:
            return
        self.viz.line_update(loss.item(), opts=dict(
            xlabel='Epoch',
            ylabel='Loss',
            title=f'Loss'
        ), name=mode)

    def update_accuracy(self, accuracy: float, mode='batch'):
        self.viz.line_update(accuracy, opts=dict(
            xlabel='Epoch',
            ylabel='Accuracy',
            title=f'Accuracy'
        ), name=mode)

    def clear(self):
        self.viz.close()

    def register_func(self, func: Callable):
        self.functions.append(func)

    def update_weight_histogram(self):
        for name, param_record in self.param_records.items():
            param_data = param_record.param.data.cpu()
            if param_data.numel() == 1:
                # not used since biases aren't tracked anymore
                self.viz.line_update(y=param_data.item(), opts=dict(
                    xlabel='Epoch',
                    ylabel='Value',
                    title='Layer biases',
                    name=name,
                ))
            else:
                self.viz.histogram(X=param_data.view(-1), win=name, opts=dict(
                    xlabel='Param norm',
                    ylabel='# bins (distribution)',
                    title=name,
                ))

    def update_weight_trace_signal_to_noise_ratio(self):
        # if weight mean / std is large, the network is confident
        # in which direction "to move"
        # if weight mean / std is small, the network makes random walk
        for name, param_record in self.param_records.items():
            mean, std = param_record.variance.get_mean_std()
            snr = mean / std
            if not torch.isfinite(snr).all():
                continue
            snr.pow_(2)
            snr.log10_().mul_(10)
            name = f"Weight SNR {name}"
            self.viz.histogram(X=snr.flatten(), win=name, opts=dict(
                xlabel='10 log10[(mean/std)^2], db',
                ylabel='# params (distribution)',
                title=name,
            ))

    def update_gradient_signal_to_noise_ratio(self):
        if self._advanced_monitoring_level.value < \
                MonitorLevel.SIGNAL_TO_NOISE.value:
            # SNR is not monitored
            return
        snr = []
        legend = []
        for name, param_record in self.param_records.items():
            param = param_record.param
            if param.grad is None:
                continue
            mean, std = param_record.grad_variance.get_mean_std()
            param_norm = param.data.norm(p=2).cpu()

            # matrix Frobenius norm is L2 norm
            mean = mean.norm(p=2) / param_norm
            std = std.norm(p=2) / param_norm

            snr.append(mean / std)
            legend.append(name)
            if self._advanced_monitoring_level is MonitorLevel.FULL:
                if (std == 0).all():
                    # skip the first update
                    continue
                self.viz.line_update(y=[mean, std], opts=dict(
                    xlabel='Epoch',
                    ylabel='Normalized Mean and STD',
                    title=f'Gradient Mean and STD: {name}',
                    legend=['||Mean(∇Wi)||', '||STD(∇Wi)||'],
                    xtype='log',
                    ytype='log',
                ))
        if np.isfinite(snr).all():
            snr.append(torch.tensor(1.))
            legend.append('phase-transition')
            self.viz.line_update(y=snr, opts=dict(
                xlabel='Epoch',
                ylabel='||Mean(∇Wi)|| / ||STD(∇Wi)||',
                title='Gradient Signal to Noise Ratio',
                legend=legend,
                xtype='log',
                ytype='log',
            ))

    def update_accuracy_epoch(self, labels_pred, labels_true, mode):
        self.update_accuracy(
            accuracy=calc_accuracy(labels_true, labels_pred), mode=mode)
        title = f"Confusion matrix '{mode}'"
        if len(labels_true.unique()) <= self.n_classes_format_ytickstep_1:
            # don't plot huge matrices
            confusion = confusion_matrix(labels_true, labels_pred)
            self.viz.heatmap(confusion, win=title, opts=dict(
                title=title,
                xlabel='Predicted label',
                ylabel='True label',
            ))

    def plot_adversarial_examples(self, model: nn.Module,
                                  adversarial_examples: AdversarialExamples,
                                  n_show=10):
        images_orig, images_adv, labels_true = adversarial_examples
        saved_mode = model.training
        model.eval()
        with torch.no_grad():
            outputs_orig = model(images_orig)
            outputs_adv = model(images_adv)
        model.train(saved_mode)
        accuracy_orig = calc_accuracy(
            labels_true=labels_true,
            labels_predicted=self.accuracy_measure.predict(outputs_orig))
        accuracy_adv = calc_accuracy(
            labels_true=labels_true,
            labels_predicted=self.accuracy_measure.predict(outputs_adv))
        self.update_accuracy(accuracy=accuracy_orig, mode='batch')
        self.update_accuracy(accuracy=accuracy_adv, mode='adversarial')
        images_stacked = []
        images_orig, images_adv = images_orig.cpu(), images_adv.cpu()
        for images in (images_orig, images_adv):
            n_show = min(n_show, len(images))
            images = images[: n_show]
            if self.normalize_inverse is not None:
                images = list(map(self.normalize_inverse, images))
                images = torch.cat(images, dim=2)
            images_stacked.append(images)
        adv_noise = images_stacked[1] - images_stacked[0]
        adv_noise -= adv_noise.min()
        adv_noise /= adv_noise.max()
        images_stacked.insert(1, adv_noise)
        images_stacked = torch.cat(images_stacked, dim=1)
        images_stacked.clamp_(0, 1)
        self.viz.image(images_stacked, win='Adversarial examples',
                       opts=dict(title='Adversarial examples'))

    def plot_mask(self, model: nn.Module, mask_trainer, image, label,
                  win_suffix=''):
        def forward_probability(image_example):
            with torch.no_grad():
                outputs = model(image_example.unsqueeze(dim=0))
            proba = mask_trainer.get_probability(outputs=outputs, label=label)
            return proba

        mask, loss_trace, image_perturbed = mask_trainer.train_mask(
            model=model, image=image, label_true=label)
        proba_original = forward_probability(image)
        proba_perturbed = forward_probability(image_perturbed)
        image, mask, image_perturbed = image.cpu(), mask.cpu(), \
                                       image_perturbed.cpu()
        image = self.normalize_inverse(image)
        image_perturbed = self.normalize_inverse(image_perturbed)
        image_masked = mask * image
        images_stacked = torch.stack(
            [image, mask, image_masked, image_perturbed], dim=0)
        images_stacked.clamp_(0, 1)
        self.viz.images(images_stacked, nrow=len(images_stacked),
                        win=f'masked images {win_suffix}', opts=dict(
                title=f"Masked image decreases neuron '{label}' probability "
                      f"{proba_original:.4f} -> {proba_perturbed:.4f}"
            ))
        self.viz.line(Y=loss_trace, X=np.arange(1, len(loss_trace) + 1),
                      win=f'mask loss {win_suffix}', opts=dict(
                xlabel='Iteration',
                title='Mask loss'
            ))

    def update_mutual_info(self):
        # for layer_name, estimated_accuracy in self.mutual_info.estimate_accuracy().items():
        #     self.update_accuracy(accuracy=estimated_accuracy, mode=layer_name)
        self.mutual_info.plot(self.viz)

    def epoch_finished(self):
        self.update_mutual_info()
        for monitored_function in self.functions:
            monitored_function(self.viz)
        self.update_grad_norm()
        self.update_gradient_signal_to_noise_ratio()
        if self._advanced_monitoring_level is MonitorLevel.FULL:
            self.param_records.plot_sign_flips(self.viz)
            self.update_initial_difference()
            self.update_weight_trace_signal_to_noise_ratio()
            self.update_weight_histogram()
        self.reset()

    def reset(self):
        for precord in self.param_records.values():
            precord.reset()

    def register_layer(self, layer: nn.Module, prefix: str):
        self.mutual_info.register(layer, name=prefix)
        for name, param in layer.named_parameters(prefix=prefix):
            if param.requires_grad and not name.endswith('.bias'):
                self.param_records[name] = ParamRecord(
                    param,
                    monitor_level=self._advanced_monitoring_level
                )

    def update_initial_difference(self):
        legend = []
        dp_normed = []
        for name, precord in self.param_records.items():
            legend.append(name)
            if precord.initial_data is None:
                precord.initial_data = precord.param.data.cpu().clone()
            dp = precord.param.data.cpu() - precord.initial_data
            dp = dp.norm(p=2) / precord.initial_norm
            dp_normed.append(dp)
        self.viz.line_update(y=dp_normed, opts=dict(
            xlabel='Epoch',
            ylabel='||W - W_initial|| / ||W_initial||',
            title='How far are the current weights from the initial?',
            legend=legend,
        ))

    def update_grad_norm(self):
        grad_norms = []
        legend = []
        for name, param_record in self.param_records.items():
            grad = param_record.param.grad
            if grad is not None:
                grad_norms.append(grad.norm(p=2).cpu())
                legend.append(name)
        if len(grad_norms) > 0:
            self.viz.line_update(y=grad_norms, opts=dict(
                xlabel='Epoch',
                ylabel='Gradient norm, L2',
                title='Gradient norm',
                legend=legend,
            ))

    def plot_psnr(self, psnr, mode='train'):
        self.viz.line_update(y=psnr, opts=dict(
            xlabel='Epoch',
            ylabel='PSNR',
            title='Peak signal-to-noise ratio',
        ), name=mode)


class MonitorEmbedding(Monitor):

    def update_sparsity(self, sparsity: float, mode: str):
        # L1 sparsity
        self.viz.line_update(y=sparsity, opts=dict(
            xlabel='Epoch',
            ylabel='L1 norm / size',
            title='Output L1 sparsity',
        ), name=mode)

    def clusters_heatmap(self, mean, std, save=False):
        """
        Cluster centers distribution heatmap.

        Parameters
        ----------
        mean, std : torch.Tensor
            Tensors of shape `(C, V)`.
            The mean and standard deviation of `C` clusters (vectors of size
            `V`).

        """
        if mean is None:
            return
        if mean.shape != std.shape:
            raise ValueError("The mean and std must have the same shape and"
                             "come from VarianceOnline.get_mean_std().")

        def compute_manhattan_dist(tensor: torch.Tensor):
            l1_dist = tensor.unsqueeze(dim=1) - tensor.unsqueeze(dim=0)
            l1_dist = l1_dist.norm(p=1, dim=2)
            upper_triangle_idx = l1_dist.triu_(1).nonzero(as_tuple=True)
            l1_dist = l1_dist[upper_triangle_idx].mean()
            return l1_dist

        n_classes = mean.shape[0]
        win = "Last layer activations heatmap"
        opts = dict(
            title=f"{win}. Epoch {self.timer.epoch}",
            xlabel='Embedding dimension',
            ylabel='Label',
            rownames=list(map(str, range(n_classes))),
        )
        if n_classes <= self.n_classes_format_ytickstep_1:
            opts.update(ytickstep=1)
        self.viz.heatmap(mean.cpu(), win=win, opts=opts)
        if save:
            self.viz.heatmap(mean.cpu(),
                             win=f"{win}. Epoch {self.timer.epoch}",
                             opts=opts)
        normalizer = mean.norm(p=1, dim=1).mean()
        outer_distance = compute_manhattan_dist(mean) / normalizer
        std = std.norm(p=1, dim=1).mean() / normalizer
        self.viz.line_update(y=[outer_distance.item(), std.item()], opts=dict(
            xlabel='Epoch',
            ylabel='Mean pairwise distance (normalized)',
            legend=['inter-distance', 'intra-STD'],
            title='How much do patterns differ in L1 measure?',
        ))

    def update_l1_neuron_norm(self, l1_norm: torch.Tensor):
        # Neuron L1 norm, normalized by the batch size, heatmap
        l1_norm = l1_norm.unsqueeze(dim=0)
        title = 'Neuron L1 norm'
        self.viz.heatmap(l1_norm, win=title, opts=dict(
            title=title,
            xlabel='Neuron ID',
            rownames=['Last layer'],
            width=None,
            height=200,
        ))


class MonitorAutoencoder(MonitorEmbedding):

    def plot_autoencoder(self, images, reconstructed, *tensors, labels=(),
                         normalize_inverse=True, n_show=10, mode='train'):
        if images.shape != reconstructed.shape:
            raise ValueError("Input & reconstructed image shapes differ")
        n_take = 10 if n_show == 'all' else n_show
        n_take = min(images.shape[0], n_take)
        combined = [images, reconstructed, *tensors]
        combined = [t.cpu() for t in combined]
        if n_show == 'all':
            combined = [t.split(n_take) for t in combined]
        else:
            combined = [[t[:n_show]] for t in combined]
        images_stacked = []
        for batch in zip(*combined):
            for tensor in batch:
                if normalize_inverse and self.normalize_inverse is not None:
                    tensor = self.normalize_inverse(tensor)
                images_stacked.append(tensor)
            empty_space = torch.ones_like(images_stacked[0])
            images_stacked.append(empty_space)
        images_stacked.pop()  # remove the last empty batch pad
        images_stacked = torch.cat(images_stacked, dim=0)
        images_stacked.clamp_(0, 1)
        labels = [f'[{mode.upper()}] Original (Top)', 'Reconstructed', *labels]
        self.viz.images(images_stacked,
                        nrow=n_take, win=f'autoencoder {mode}', opts=dict(
                title=' | '.join(labels),
                width=1000,
                height=None,
            ))

    def plot_reconstruction_error(self, pixel_missed, thresholds,
                                  optimal_id=None):
        title = "Reconstruction error"
        self.viz.line(Y=pixel_missed, X=thresholds, win=title, opts=dict(
            title=title,
            xlabel="reconstruct threshold",
            ylabel="#incorrect_pixels"
        ))
        if optimal_id is None:
            optimal_id = pixel_missed.argmin()
        self.viz.line_update(pixel_missed[optimal_id], opts=dict(
            title="Reconstruction error lowest",
            xlabel="Epoch",
            ylabel="min_thr #incorrect_pixels"
        ))
        self.viz.line_update(thresholds[optimal_id].item(), opts=dict(
            title="Reconstruction threshold",
            xlabel="Epoch",
            ylabel="Thr. that minimizes the error"
        ))
