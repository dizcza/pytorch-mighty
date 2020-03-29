import os
import subprocess
from collections import UserDict
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from sklearn.metrics import confusion_matrix, pairwise

from mighty.monitor.accuracy import calc_accuracy, Accuracy, \
    AccuracyArgmax
from mighty.monitor.batch_timer import timer, ScheduleStep
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
        self.viz.prepare()

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
                 f"accuracy_measure={self.accuracy_measure}, "
                 f"level={self._advanced_monitoring_level})")
        self.log(repr(self.mutual_info))
        fwd_size = os.environ.get('FULL_FORWARD_PASS_SIZE', '(all samples)')
        self.log(f"FULL_FORWARD_PASS_SIZE: {fwd_size}")
        self.log(f"BATCH_SIZE: {os.environ.get('BATCH_SIZE', '(default)')}")
        self.log(f"Batches in epoch: {self.timer.batches_in_epoch}")
        self.log(f"Start epoch: {self.timer.epoch}")
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'],
                                stdout=subprocess.PIPE,
                                universal_newlines=True)
        self.log(f"Git commit: {commit.stdout}")

    def log(self, text: str):
        self.viz.log(text)

    def batch_finished(self, model: nn.Module):
        self.param_records.batch_finished()
        self.timer.tick()
        if self.timer.epoch == 0:
            self.mutual_info.force_update(model)
            self.update_mutual_info()

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
        self.viz.prepare()

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
        for name, param_record in self.param_records.items():
            name = f"Weight SNR {name}"
            param_norm = param_record.param.data.norm(p=2).cpu()
            mean, std = param_record.variance.get_mean_std()
            param_record.variance.reset()
            snr = mean / (std + 1e-6)
            snr /= param_norm
            snr.pow_(2)
            self.viz.histogram(X=snr.view(-1), win=name, opts=dict(
                xlabel='(mean/std)^2',
                ylabel='# bins (distribution)',
                title=name,
                xtype='log',
            ))

    def update_gradient_signal_to_noise_ratio(self):
        snr = []
        legend = []
        for name, param_record in self.param_records.items():
            param = param_record.param
            if param.grad is None:
                continue
            mean, std = param_record.grad_variance.get_mean_std()
            param_record.grad_variance.reset()
            param_norm = param.data.norm(p=2).cpu()

            # matrix Frobenius norm is L2 norm
            mean = mean.norm(p=2) / param_norm
            std = std.norm(p=2) / param_norm

            snr.append(mean / std)
            legend.append(name)
            if self._advanced_monitoring_level is MonitorLevel.FULL:
                self.viz.line_update(y=[mean, std], opts=dict(
                    xlabel='Epoch',
                    ylabel='Normalized Mean and STD',
                    title=f'Gradient Mean and STD: {name}',
                    legend=['||Mean(∇Wi)||', '||STD(∇Wi)||'],
                    xtype='log',
                    ytype='log',
                ))
        self.viz.line_update(y=snr, opts=dict(
            xlabel='Epoch',
            ylabel='||Mean(∇Wi)|| / ||STD(∇Wi)||',
            title='Grad Signal to Noise Ratio',
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
        if self._advanced_monitoring_level.value >= \
                MonitorLevel.SIGNAL_TO_NOISE.value:
            self.update_gradient_signal_to_noise_ratio()
        if self._advanced_monitoring_level is MonitorLevel.FULL:
            self.param_records.plot_sign_flips(self.viz)
            self.update_initial_difference()
            self.update_weight_trace_signal_to_noise_ratio()
            self.update_weight_histogram()

    def register_layer(self, layer: nn.Module, prefix: str):
        self.mutual_info.register(layer, name=prefix)
        for name, param in layer.named_parameters(prefix=prefix):
            if param.requires_grad and not name.endswith('.bias'):
                self.param_records[name] = ParamRecord(param,
                    monitor_level=self._advanced_monitoring_level)

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
            title='How far the current weights are from the initial?',
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

    def plot_psnr(self, psnr):
        self.viz.line_update(y=psnr, opts=dict(
            xlabel='Epoch',
            ylabel='PSNR',
            title='Peak signal-to-noise ratio',
        ))