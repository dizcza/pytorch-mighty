"""
Monitors
--------

.. autosummary::
    :toctree: toctree/monitor

    Monitor
    MonitorEmbedding
    MonitorAutoencoder


Monitor Parameter Records
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: toctree/monitor

    ParamRecord
    ParamsDict

"""


from collections import UserDict, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from torcheval.metrics.functional import binary_precision_recall_curve, auc, binary_auprc

from mighty.monitor.accuracy import calc_accuracy, Accuracy
from mighty.monitor.batch_timer import timer, ScheduleExp
from mighty.monitor.mutual_info.stub import MutualInfoStub
from mighty.utils.var_online import VarianceOnline
from mighty.monitor.viz import VisdomMighty
from mighty.utils.common import clone_cpu
from mighty.utils.domain import MonitorLevel


__all__ = [
    "ParamRecord",
    "ParamsDict",
    "Monitor",
    "MonitorEmbedding",
    "MonitorAutoencoder"
]


class ParamRecord:
    """
    A parameter record, created by a monitor, that tracks parameter statistics
    like gradient variance, sign flips on update step, etc.

    Parameters
    ----------
    param : nn.Parameter
        Model parameter.
    monitor_level : MonitorLevel, optional
        The extent of keeping the statistics.
        Default: MonitorLevel.DISABLED

    """

    def __init__(self, param, monitor_level=MonitorLevel.DISABLED):
        self.param = param
        self.monitor_level = monitor_level
        self.grad_variance = VarianceOnline()
        self.variance = VarianceOnline()
        self.prev_data = None
        if monitor_level & MonitorLevel.SIGN_FLIPS:
            self.prev_data = clone_cpu(param.data)
        self.initial_data = None
        if monitor_level & MonitorLevel.WEIGHT_INITIAL_DIFFERENCE:
            self.initial_data = clone_cpu(param.data)
        self.initial_norm = param.data.float().norm(p=2).item()

    @staticmethod
    def count_sign_flips(new_data, prev_data):
        """
        Count the no. of sing flips in `prev_data -> new_data`.

        The default implementation is

        .. code-block:: python

            sum(new_data * prev_data)

        Parameters
        ----------
        new_data, prev_data : torch.Tensor
            New and previous tensors.

        Returns
        -------
        sign_flips : int
            The no. of signs flipped.
        """
        sign_flips = (new_data * prev_data < 0).sum().item()
        return sign_flips

    def update_signs(self):
        """
        Updates the number of sign flips by comparing with the previously
        stored tensor.

        Returns
        -------
        sign_flips : float
            Normalized number of sign flips in range ``[0, 1]``.

        """
        new_data = clone_cpu(self.param.data)
        if self.prev_data is None:
            # The user might turn on the advanced monitoring in the middle.
            self.prev_data = new_data
        sign_flips = self.count_sign_flips(new_data, self.prev_data)
        self.prev_data = new_data
        return sign_flips

    def update_grad_variance(self):
        """
        Updates the gradient variance, need for Signal-to-Noise ratio
        estimation.
        """
        if self.param.grad is not None:
            self.grad_variance.update(self.param.grad.data.cpu())

    def reset(self):
        """
        Resets the current state and all saved variables.
        """
        self.grad_variance.reset()
        self.variance.reset()


class ParamsDict(UserDict):
    """
    A dictionary that holds named `ParamRecord`s.
    """
    def __init__(self):
        super().__init__()
        self.sign_flips = defaultdict(int)
        self.n_updates = 0

    def reset(self):
        self.sign_flips.clear()
        self.n_updates = 0

    def batch_finished(self):
        """
        Batch finished callback that triggers `update*()` methods of the
        stored param records.
        """
        def filter_level(level: MonitorLevel):
            # filter by greater or equal to the Monitor level
            return (precord for precord in self.values()
                    if precord.monitor_level & level)

        self.n_updates += 1
        for precord in filter_level(MonitorLevel.SIGNAL_TO_NOISE):
            precord.update_grad_variance()
        for name, precord in self.items():
            if precord.monitor_level & MonitorLevel.SIGN_FLIPS:
                self.sign_flips[name] += precord.update_signs()
        for precord in filter_level(MonitorLevel.WEIGHT_SNR_TRACE):
            precord.variance.update(precord.param.data.cpu())

    def plot_sign_flips(self, viz, total=True):
        """
        Plots the no. of weight sign flips after a batch update. Refer to
        :func:`ParamRecord.update_signs`.

        Parameters
        ----------
        viz : VisdomMighty
            Visdom server instance.
        total : bool, optional
            Whether to sum across all the parameters (True) or plot individual
            parameter sign flips counts (False).
            Default: True
        """
        opts = dict(
            xlabel='Epoch',
            ylabel='Sign flips',
            title="Sign flips after optimizer.step()",
        )
        if total:
            sign_flips = sum(self.sign_flips.values()) / self.n_updates
        else:
            labels, sign_flips = zip(*self.sign_flips.items())
            sign_flips = np.divide(sign_flips, self.n_updates)
            opts['legend'] = list(labels)
        viz.line_update(y=sign_flips, opts=opts)


class Monitor:
    """
    Generic Monitor that provides meaningful statistics in interactive Visdom
    plots throughout training.

    Parameters
    ----------
    mutual_info : MutualInfo or None, optional
        The Mutual Information estimator (the same as in a trainer).
        Default: None
    normalize_inverse : NormalizeInverse or None, optional
        The inverse normalization transform, taken from a data loader.
        Default: None
    """

    n_classes_format_ytickstep_1 = 15

    def __init__(self, mutual_info=None, normalize_inverse=None):
        self.timer = timer
        self.viz = None
        self._advanced_monitoring_level = MonitorLevel.DISABLED
        self.normalize_inverse = normalize_inverse
        self.param_records = ParamsDict()
        if mutual_info is None:
            mutual_info = MutualInfoStub()
        self.mutual_info = mutual_info
        self.functions = []

    @property
    def is_active(self):
        """
        Returns
        -------
        bool
            Indicator whether a Visdom server is initialized or not.
        """
        return self.viz is not None

    def advanced_monitoring(self, level=MonitorLevel.DEFAULT):
        """
        Sets the extent of monitoring.

        Parameters
        ----------
        level : MonitorLevel, optional
            New monitoring level to apply.
            * DISABLED - only basic metrics are computed (memory tolerable)
            * SIGNAL_TO_NOISE - track SNR of the gradients
            * FULL - SNR, sign flips, weight hist, weight diff
            Default: MonitorLevel.NORMAL

        Notes
        -----
        Advanced monitoring is memory consuming.

        """
        self._advanced_monitoring_level = level
        for param_record in self.param_records.values():
            param_record.monitor_level = level

    def open(self, env_name: str, offline=False):
        """
        Opens a Visdom server.

        Parameters
        ----------
        env_name : str
            Environment name.
        offline : bool
            Offline mode (True) or online (False).

        """
        self.viz = VisdomMighty(env=env_name, offline=offline)

    def log_model(self, model, space='-'):
        """
        Logs the model.

        Parameters
        ----------
        model : nn.Module
            A PyTorch model.
        space : str, optional
            A space substitution to correctly parse HTML later on.
            Default: '-'
        """
        lines = []
        for line in repr(model).splitlines():
            n_spaces = len(line) - len(line.lstrip())
            line = space * n_spaces + line
            lines.append(line)
        lines = '<br>'.join(lines)
        self.log(lines)

    def log_self(self):
        """
        Logs the monitor itself.
        """
        self.log(f"{self.__class__.__name__}("
                 f"level={self._advanced_monitoring_level})")

    def log(self, text):
        """
        Logs the text.

        Parameters
        ----------
        text : str
            Log text.
        """
        self.viz.log(text)

    def batch_finished(self, model):
        """
        Batch finished monitor callback.

        Parameters
        ----------
        model : nn.Module
            A model that has been trained the last epoch.

        """
        self.param_records.batch_finished()
        self.timer.tick()
        if self.timer.epoch == 0:
            self.batch_finished_first_epoch(model)

    @ScheduleExp()
    def batch_finished_first_epoch(self, model):
        """
        First batch finished monitor callback.

        Parameters
        ----------
        model : nn.Module
            A model that has been trained the last epoch.

        """
        # inspect the very beginning of the training progress
        self.mutual_info.force_update(model)
        self.update_mutual_info()
        self.update_gradient_signal_to_noise_ratio()

    def update_precision_recall(self, proba, labels_true, train=True):
        if proba.shape[1] == 2:
            mode = 'Train' if train else 'Test'
            proba = proba[:, 1]
            prec, recall, thr = binary_precision_recall_curve(proba.cuda(), labels_true.cuda())
            self.viz.line(Y=torch.stack([prec[:-1], thr], dim=1), X=recall[:-1],
                          win=f'prec-recall-{mode}', opts=dict(
                    xlabel='Recall',
                    ylabel='Precision',
                    title=f'Precision Recall {mode}',
                    legend=['precision', 'threshold']
                ))
            self.viz.line_update(auc(recall, prec, reorder=True).item(), opts=dict(
                xlabel='Epoch',
                ylabel='Area Under Curve',
                title='Precision Recall AUC'
            ), name=mode.lower())

    def update_loss(self, loss, mode='batch'):
        """
        Update the loss plot with a new value.

        Parameters
        ----------
        loss : torch.Tensor
            Loss tensor. If None, do noting.
        mode : {'batch', 'epoch'}, optional
            The update mode.
            Default: 'batch'

        """
        if loss is None:
            return
        self.viz.line_update(loss.item(), opts=dict(
            xlabel='Epoch',
            title='Loss'
        ), name=mode)

    def update_accuracy(self, accuracy, mode='batch'):
        """
        Update the accuracy plot with a new value.

        Parameters
        ----------
        accuracy : torch.Tensor or float
            Accuracy scalar.
        mode : {'batch', 'epoch'}, optional
            The update mode.
            Default: 'batch'

        """
        title = 'Accuracy'
        if isinstance(self, MonitorAutoencoder):
            # the ability to identify class ID given an embedding vector
            title = f"{title} embedding"
        self.viz.line_update(accuracy, opts=dict(
            xlabel='Epoch',
            ylabel='Accuracy',
            title=title
        ), name=mode)

    def clear(self):
        """
        Clear out all Visdom plots.
        """
        self.viz.close()

    def register_func(self, func):
        """
        Register a plotting function to call on the end of each epoch.

        The `func` must have only one argument `viz`, a Visdom instance.

        Parameters
        ----------
        func : callable
            User-provided plot function with one argument `viz`.

        """
        self.functions.append(func)

    def update_weight_histogram(self):
        """
        Update the model weights histogram.
        """
        if not self._advanced_monitoring_level & MonitorLevel.WEIGHT_HISTOGRAM:
            return
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
                    xlabel='Param value',
                    ylabel='# bins (distribution)',
                    title=name,
                ))

    def update_weight_trace_signal_to_noise_ratio(self):
        """
        Update the SNR, mean divided by std, of the model weights.

        If mean / std is large, the network is confident in which direction
        to "move". If mean / std is small, the network is making random walk.
        """
        if not self._advanced_monitoring_level & MonitorLevel.WEIGHT_SNR_TRACE:
            return
        for name, param_record in self.param_records.items():
            mean, std = param_record.variance.get_mean_std()
            snr = mean / std
            snr = snr[torch.isfinite(snr)]
            if snr.nelement() == 0:
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
        """
        Update the SNR, mean divided by std, of the model weight gradients.

        Similar to :func:`Monitor.update_weight_trace_signal_to_noise_ratio`
        but on a smaller time scale.
        """
        if not self._advanced_monitoring_level & MonitorLevel.SIGNAL_TO_NOISE:
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
            if self._advanced_monitoring_level == MonitorLevel.FULL:
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
        if any(snr) and np.isfinite(snr).all():
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
        """
        The callback to calculate and update the epoch accuracy from a batch
        of predicted and true class labels.

        Parameters
        ----------
        labels_pred, labels_true : (N,) torch.Tensor
            Predicted and true class labels.
        mode : str
            Update mode: 'batch' or 'epoch'.

        Returns
        -------
        accuracy : torch.Tensor
            A scalar tensor with one value - accuracy.
        """
        logscale = getattr(self, "logscale", False)
        accuracy = calc_accuracy(labels_true, labels_pred)
        self.update_accuracy(accuracy=accuracy, mode=mode)
        title = f"Confusion matrix '{mode}'"
        if len(labels_true.unique()) <= self.n_classes_format_ytickstep_1:
            # don't plot huge matrices
            if labels_true.ndim == 2:
                onehot = F.one_hot(labels_pred, num_classes=labels_true.shape[1])
                labels_true = (labels_true * onehot).argmax(dim=1)
            confusion = confusion_matrix(labels_true, labels_pred)
            if logscale:
                confusion = confusion.astype(np.float32)
                confusion[confusion == 0] = np.nan
                confusion = np.log(confusion)
            self.viz.heatmap(confusion, win=title, opts=dict(
                title=title,
                xlabel='Predicted label',
                ylabel='True label',
            ))
        return accuracy

    def plot_explain_input_mask(self, model, mask_trainer, image, label,
                                win_suffix=''):
        """
        Plot the mask where the model is "looking at" to make decisions about
        the class label. Based on [1]_.

        Parameters
        ----------
        model : nn.Module
            The model.
        mask_trainer : MaskTrainer
            The instance of :class:`MaskTrainer`.
        image : torch.Tensor
            The input image to investigate and plot the mask on.
        label : int
            The class label to investigate.
        win_suffix : str, optional
            The unique window suffix to distinguish different scenarios.
            Default: ''

        References
        ----------
        .. [1] Fong, R. C., & Vedaldi, A. (2017). Interpretable explanations of
           black boxes by meaningful perturbation. In Proceedings of the IEEE
           International Conference on Computer Vision (pp. 3429-3437).

        """
        def forward_probability(image_example):
            with torch.no_grad():
                outputs = model(image_example.unsqueeze(dim=0))
            proba = mask_trainer.get_probability(outputs=outputs, label=label)
            return proba

        mask, image_perturbed, loss_trace = mask_trainer.train_mask(
            model=model, image=image, label_true=label)
        proba_original = forward_probability(image)
        proba_perturbed = forward_probability(image_perturbed)
        image, mask, image_perturbed = image.cpu(), mask.cpu(), \
                                       image_perturbed.cpu()
        if self.normalize_inverse is not None:
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
        """
        Update the mutual info.
        """
        self.mutual_info.plot(self.viz)

    def epoch_finished(self):
        """
        Epoch finished callback.
        """
        self.update_mutual_info()
        for monitored_function in self.functions:
            monitored_function(self.viz)
        self.update_grad_norm()
        self.update_gradient_signal_to_noise_ratio()
        if self._advanced_monitoring_level & MonitorLevel.SIGN_FLIPS:
            self.param_records.plot_sign_flips(self.viz)
        self.param_records.reset()
        self.update_initial_difference()
        self.update_weight_trace_signal_to_noise_ratio()
        self.update_weight_histogram()
        self.reset()

    def reset(self):
        """
        Reset the parameter records statistics.
        """
        for precord in self.param_records.values():
            precord.reset()

    def register_layer(self, layer: nn.Module, prefix: str):
        """
        Register a layer to monitor.

        Parameters
        ----------
        layer : nn.Module
            A model layer.
        prefix : str
            The layer name.
        """
        for name, param in layer.named_parameters(prefix=prefix):
            if param.requires_grad and not name.endswith('.bias'):
                self.param_records[name] = ParamRecord(
                    param,
                    monitor_level=self._advanced_monitoring_level
                )

    def update_initial_difference(self):
        """
        Update the L1 normalized difference between the current and starting
        weights (before training).
        """
        if not self._advanced_monitoring_level \
                & MonitorLevel.WEIGHT_INITIAL_DIFFERENCE:
            return
        legend = []
        dp_normed = []
        for name, precord in self.param_records.items():
            legend.append(name)
            if precord.initial_data is None:
                precord.initial_data = precord.param.data.cpu().clone()
            dp = precord.param.data.cpu() - precord.initial_data
            dp = dp.float().norm(p=2) / precord.initial_norm
            dp_normed.append(dp)
        self.viz.line_update(y=dp_normed, opts=dict(
            xlabel='Epoch',
            ylabel='||W - W_initial|| / ||W_initial||',
            title='How far are the current weights from the initial?',
            legend=legend,
        ))

    def update_grad_norm(self):
        """
        Update the parameters gradient norm.
        """
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
        """
        If given, plot the Peak Signal to Noise Ratio.

        Used in training autoencoders.

        Parameters
        ----------
        psnr : torch.Tensor or float
            The Peak Signal to Noise Ratio scalar.
        mode : {'train', 'test'}, optional
            The update mode.
            Default: 'train'
        """
        self.viz.line_update(y=psnr, opts=dict(
            xlabel='Epoch',
            ylabel='PSNR',
            title='Peak signal-to-noise ratio',
        ), name=mode)


class MonitorEmbedding(Monitor):
    """
    A monitor for :class:`mighty.trainer.TrainerEmbedding`.
    """

    def update_sparsity(self, sparsity, mode):
        """
        Update the L1 sparsity of the hidden layer activations.

        Parameters
        ----------
        sparsity : torch.Tensor or float
            Sparsity scalar.
        mode : {'train', 'test'}
            The update mode.

        """
        # L1 sparsity
        self.viz.line_update(y=sparsity, opts=dict(
            xlabel='Epoch',
            ylabel='L1 norm / size',
            title='Output L1 sparsity',
        ), name=mode)

    def embedding_hist(self, activations):
        """
        Plots embedding activations histogram.

        Parameters
        ----------
        activations : (N,) torch.Tensor
            The averaged embedding vector.
        """
        if activations is None:
            return
        title = "Embedding activations hist"
        self.viz.histogram(X=activations.flatten(), win=title, opts=dict(
            xlabel='Neuron value',
            ylabel='Count',
            title=title,
            ytype='log',
        ))

    def update_pairwise_dist(self, mean, std):
        r"""
        Updates L1 mean pairwise distance and intra-STD of embedding centroids.

        .. math::
            \text{inter-distance} = \frac{pdist(\text{mean}, p=1)}
            {||\text{mean}||_1}

            \text{intra-STD} = \frac{||\text{std}||_1}{||\text{mean}||_1},

        where :math:`||x||_1 = \frac{\sum_{ij}{|x_{ij}|}}{C}`.

        When these two metrics equalize, the feature vectors, embeddings,
        become indistinguishable. We want :math:`\text{inter-distance}` to be
        high and :math:`\text{intra-STD}` to keep low.

        Parameters
        ----------
        mean, std : torch.Tensor
            Tensors of shape `(C, V)`.
            The mean and standard deviation of `C` clusters (vectors of size
            `V`).

        Returns
        -------
        torch.Tensor
            A scalar, a proxy to signal-to-noise ratio defined as

            .. math::
                \frac{\text{inter-distance}}{\text{intra-STD}}

        """
        if mean is None:
            return
        if mean.shape != std.shape:
            raise ValueError("The mean and std must have the same shape and "
                             "come from VarianceOnline.get_mean_std().")

        l1_norm = mean.norm(p=1, dim=1).mean()
        pdist = torch.pdist(mean, p=1).mean() / l1_norm
        std = std.norm(p=1, dim=1).mean() / l1_norm
        self.viz.line_update(y=[pdist.item(), std.item()], opts=dict(
            xlabel='Epoch',
            ylabel='Mean pairwise distance (normalized) and STD',
            legend=['inter-distance', 'intra-STD'],
            title='How much do patterns differ in L1 measure?',
        ))
        return pdist / std

    def clusters_heatmap(self, mean, title=None, save=False):
        """
        Cluster centers distribution heatmap.

        Parameters
        ----------
        mean : (C, V) torch.Tensor
            The mean of `C` clusters (vectors of size `V`).
        title : str or None, optional
            An optional title to the plot.
            If None, set to "Embedding activations heatmap".
            Default: None
        save : bool, optional
            Save the heatmap plot in a separate fixed window.
            Default: False

        """
        if mean is None:
            return

        n_classes = mean.shape[0]
        if title is None:
            win = "Embedding activations heatmap"
            title = f"{win}. Epoch {self.timer.epoch}"
        else:
            win = title
        opts = dict(
            title=title,
            xlabel='Neuron',
            ylabel='Label',
            rownames=list(map(str, range(n_classes))),
        )
        if n_classes <= self.n_classes_format_ytickstep_1:
            opts.update(ytickstep=1)
        if save:
            win = f"{win}. Epoch {self.timer.epoch}"
        self.viz.heatmap(mean.cpu(), win=win, opts=opts)

    def update_l1_neuron_norm(self, l1_norm: torch.Tensor):
        """
        Update the L1 neuron norm heatmap, normalized by the batch size.

        Useful to explore which neurons are "dead" and which - "super active".

        Parameters
        ----------
        l1_norm : (V,) torch.Tensor
            L1 norm per neuron in a hidden layer.
        """
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
    """
    A monitor for :class:`mighty.trainer.TrainerAutoencoder`.
    """

    def plot_autoencoder(self, images, reconstructed, *tensors, labels=(),
                         normalize_inverse=True, n_show=10, mode='train'):
        """
        Plot autoencoder reconstructed samples.

        Parameters
        ----------
        images, reconstructed : (B, C, H, W) torch.Tensor
            A batch of input and reconstructed images.
        *tensors : (B, C, H, W) torch.Tensor
            Other tensors of the same size, showing intermediate steps.
        labels : tuple of str, optional
            The labels of additional `tensors`.
        normalize_inverse : bool, optional
            Perform inverse normalization to the input samples to show images
            in the original input domain rather than zero-mean unit-variance
            normalized representation.
            Default: True
        n_show : int, optional
            The number of samples to show.
            Default: 10
        mode : {'train', 'test'}
            The update mode.
            Default: 'train'

        """
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
        images_stacked = torch.cat(images_stacked, dim=0).squeeze_()
        if images_stacked.ndim == 2:
            # (2xB, L) time series
            images, reconstructed = images_stacked.tensor_split(len(combined))[:2]
            images = images.numpy()
            reconstructed = reconstructed.numpy()
            for i, (im, rec) in enumerate(zip(images, reconstructed)):
                ts = np.c_[im, rec]
                title = f"Reconstructed {i}"
                self.viz.line(Y=ts, X=np.arange(len(ts)), win=title, opts=dict(
                    title=title,
                    label=['orig', 'reconstructed']
                ))
                self.viz.update_window_opts(win=title, opts=dict(legend=[], title=title))
        else:
            # (2xB, H, W) images
            images_stacked.clamp_(0, 1)
            if images_stacked.ndim == 3:
                images_stacked.unsqueeze_(1)  # (B, 1, H, W)
            labels = [f'[{mode.upper()}] Original (Top)', 'Reconstructed', *labels]
            self.viz.images(images_stacked,
                            nrow=n_take, win=f'autoencoder {mode}', opts=dict(
                    title=' | '.join(labels),
                    width=1000,
                    height=None,
                ))

    def plot_reconstruction_error(self, pixel_missed, thresholds,
                                  optimal_id=None):
        """
        Plot the reconstruction pixel-wise error, depending on the threshold.

        When the input images can be considered as binary, like MNIST, this
        plot helps to choose the correct threshold that minimizes incorrectly
        reconstructed binary pixels count.

        Parameters
        ----------
        pixel_missed : (N,) torch.Tensor
            Incorrectly reconstructed pixels count.
        thresholds : (N,) torch.Tensor
            Used thresholds.
        optimal_id : int or None, optional
            The optimal threshold ID used as the "best" threshold. If None,
            set to ``pixel_missed.argmin()``.
        """
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
