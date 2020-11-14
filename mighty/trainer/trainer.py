import subprocess
import sys
import time
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from mighty.loss import PairLossSampler, LossPenalty
from mighty.models import AutoencoderOutput, MLP
from mighty.monitor.accuracy import AccuracyEmbedding, \
    AccuracyArgmax, Accuracy
from mighty.monitor.batch_timer import timer
from mighty.monitor.monitor import Monitor
from mighty.monitor.mutual_info import MutualInfoNeuralEstimation, \
    MutualInfoStub
from mighty.monitor.mutual_info.mutual_info import MutualInfo
from mighty.monitor.var_online import MeanOnline
from mighty.trainer.mask import MaskTrainer
from mighty.utils.common import find_named_layers, batch_to_cuda, \
    input_from_batch
from mighty.utils.constants import CHECKPOINTS_DIR
from mighty.utils.data import DataLoader
from mighty.utils.prepare import prepare_eval


__all__ = [
    "Trainer"
]


class Trainer(ABC):
    """
    Trainer base class.

    Parameters
    ----------
    model : nn.Module
        A neural network to train.
    criterion : nn.Module
        Loss function.
    data_loader : DataLoader
        A data loader.
    accuracy_measure : Accuracy
        Calculates the accuracy from the last layer activations.
        If None, set to :code:`AccuracyArgmax` for a classification problem
        and :code:`AccuracyEmbedding` otherwise.
        Default: None
    mutual_info : MutualInfo
        A handle to compute the mutual information I(X; T) and I(Y; T) [1]_.
        If None, don't compute the mutual information.
        Default: None
    env_suffix : str
        The suffix to add to the current environment name.
        Default: ''
    checkpoint_dir : Path or str
        The path to store the checkpoints.
        Default: '${HOME}/.mighty/checkpoints'

    References
    ----------
    .. [1] Shwartz-Ziv, R., & Tishby, N. (2017). Opening the black box of deep
       neural networks via information. arXiv preprint arXiv:1703.00810.

    Notes
    -----
    For the choice of `mutual_info` refer to
    https://github.com/dizcza/entropy-estimators
    """

    watch_modules = (nn.Linear, nn.Conv2d, MLP)

    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 data_loader: DataLoader,
                 accuracy_measure: Accuracy = None,
                 mutual_info=None,
                 env_suffix='',
                 checkpoint_dir=CHECKPOINTS_DIR):
        if torch.cuda.is_available():
            model.cuda()
        self.model = model
        self.criterion = criterion
        self.data_loader = data_loader
        self.train_loader = data_loader.get(train=True)
        if mutual_info is None:
            mutual_info = MutualInfoNeuralEstimation(data_loader)
        self.mutual_info = mutual_info
        self.checkpoint_dir = Path(checkpoint_dir)
        self.timer = timer
        self.timer.init(batches_in_epoch=len(self.train_loader))
        self.timer.set_epoch(0)
        criterion_name = self.criterion.__class__.__name__
        if isinstance(criterion, LossPenalty):
            criterion_name = f"{criterion_name}(" \
                             f"{criterion.criterion.__class__.__name__})"
        self.env_name = f"{time.strftime('%Y.%m.%d')} " \
                        f"{model.__class__.__name__}: " \
                        f"{data_loader.dataset_cls.__name__} " \
                        f"{self.__class__.__name__} " \
                        f"{criterion_name}"
        env_suffix = env_suffix.lstrip(' ')
        if env_suffix:
            self.env_name = f'{self.env_name} {env_suffix}'
        if accuracy_measure is None:
            if isinstance(self.criterion, PairLossSampler):
                accuracy_measure = AccuracyEmbedding()
            else:
                # cross entropy loss
                accuracy_measure = AccuracyArgmax()
        self.accuracy_measure = accuracy_measure
        self.monitor = self._init_monitor(mutual_info)
        self.online = self._init_online_measures()
        self.best_score = 0.

    @property
    def epoch(self):
        """
        The current epoch, int.
        """
        return self.timer.epoch

    def checkpoint_path(self, best=False):
        """
        Get the checkpoint path, given the mode.

        Parameters
        ----------
        best : bool
            The best (True) or normal (False) mode.

        Returns
        -------
        Path
            Checkpoint path.
        """
        checkpoint_dir = self.checkpoint_dir
        if best:
            checkpoint_dir = self.checkpoint_dir / "best"
        return checkpoint_dir / (self.env_name + '.pt')

    def monitor_functions(self):
        """
        Override this method to register `Visdom` callbacks on each epoch.
        """
        pass

    def log_trainer(self):
        """
        Logs the trainer in `Visdom` text field.
        """
        self.monitor.log_model(self.model)
        self.monitor.log(f"Criterion: {self.criterion}")
        self.monitor.log(repr(self.data_loader))
        self.monitor.log_self()
        self.monitor.log(repr(self.accuracy_measure))
        self.monitor.log(repr(self.mutual_info))
        git_dir = Path(sys.argv[0]).parent
        while str(git_dir) != git_dir.root:
            commit = subprocess.run(['git', '--git-dir', str(git_dir / '.git'),
                                     'rev-parse', 'HEAD'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    universal_newlines=True)
            if commit.returncode == 0:
                self.monitor.log(f"Git location '{str(git_dir)}' "
                                 f"commit: {commit.stdout}")
                break
            git_dir = git_dir.parent

    def _init_monitor(self, mutual_info) -> Monitor:
        monitor = Monitor(
            accuracy_measure=self.accuracy_measure,
            mutual_info=mutual_info,
            normalize_inverse=self.data_loader.normalize_inverse
        )
        return monitor

    def _init_online_measures(self) -> Dict[str, MeanOnline]:
        return dict(accuracy=MeanOnline())

    @abstractmethod
    def train_batch(self, batch):
        """
        The core function of a trainer to update the model parameters, given
        a batch.

        Parameters
        ----------
        batch : torch.Tensor or tuple of torch.Tensor
            :code:`(X, Y)` or :code:`X` batch of input data.

        Returns
        -------
        loss : torch.Tensor
            The batch loss.

        """
        raise NotImplementedError()

    def update_best_score(self, score, score_type=''):
        """
        If :code:`score` is greater than the :code:`self.best_score`, save
        the model.

        Parameters
        ----------
        score : float
            The model score at the current epoch. The higher, the better.
            The simplest way to use this function is set :code:`score = -loss`.
        score_type : str
            An optional user-defined key-word to determine the criteria for the
            "best" score. For example, one may be interested in both the
            accuracy and the loss of a model as a proxy to the "best" score.
            Default: ''
        """
        if score > self.best_score:
            self.best_score = score
            self.save(best=True)

    def save(self, best=False):
        """
        Saves the trainer and the model parameters to
        :code:`self.checkpoint_path(best)`.

        Parameters
        ----------
        best : bool
            The mode (refer to :func:`Trainer.checkpoint_path`).

        See Also
        --------
        restore : restore the training progress
        """
        checkpoint_path = self.checkpoint_path(best)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(self.state_dict(), checkpoint_path)
        except PermissionError as error:
            print(error)

    def state_dict(self):
        """
        Returns
        -------
        dict
            A dict of the trainer state to be saved.
        """
        return {
            "model_state": self.model.state_dict(),
            "epoch": self.timer.epoch,
            "env_name": self.env_name,
            "best_score": self.best_score,
        }

    def restore(self, checkpoint_path=None, best=False, strict=True):
        """
        Restores the trainer progress and the model from the path.

        Parameters
        ----------
        checkpoint_path : Path or None
            Trainer checkpoint path to restore. If None, the default path
            :code:`self.checkpoint_path()` is used.
            Default: None
        best : bool
            The mode (refer to :func:`Trainer.checkpoint_path`).
        strict : bool
            Strict model loading or not.

        Returns
        -------
        checkpoint_state : dict
            The loaded state of a trainer.
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path(best)
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print(f"Checkpoint '{checkpoint_path}' doesn't exist. "
                  f"Nothing to restore.")
            return None
        map_location = None
        if not torch.cuda.is_available():
            map_location = 'cpu'
        checkpoint_state = torch.load(checkpoint_path,
                                      map_location=map_location)
        try:
            self.model.load_state_dict(checkpoint_state['model_state'],
                                       strict=strict)
        except RuntimeError as error:
            print(f"Restoring {checkpoint_path} raised {error}")
            return None
        self.env_name = checkpoint_state['env_name']
        self.timer.set_epoch(checkpoint_state['epoch'])
        self.best_score = checkpoint_state['best_score']
        self.monitor.open(env_name=self.env_name)
        print(f"Restored model state from {checkpoint_path}.")
        return checkpoint_state

    def _get_loss(self, batch, output):
        # In case of unsupervised learning, '_get_loss' is overridden
        # accordingly.
        input, labels = batch
        return self.criterion(output, labels)

    def _on_forward_pass_batch(self, batch, output, train):
        if self.is_unsupervised():
            # unsupervised, no labels
            return
        _, labels = batch
        if train:
            self.accuracy_measure.partial_fit(output, labels)
        else:
            self.accuracy_measure.true_labels_cached.append(labels.cpu())
            labels_pred = self.accuracy_measure.predict(output).cpu()
            self.accuracy_measure.predicted_labels_cached.append(labels_pred)

    def _forward(self, batch):
        input = input_from_batch(batch)
        return self.model(input)

    def is_unsupervised(self):
        """
        Returns
        -------
        bool
            True, if the training is unsupervised and False otherwise.
        """
        return not self.data_loader.has_labels

    def full_forward_pass(self, train=True):
        """
        Fixes the model weights, evaluates the epoch score and updates the
        monitor.

        Parameters
        ----------
        train : bool
            Either train (True) or test (False) batches to run. In both cases,
            the model is set to the evaluation regime via `self.model.eval()`.

        Returns
        -------
        loss : torch.Tensor
            The loss of a full forward pass.

        """
        mode_saved = self.model.training
        self.model.train(False)
        self.accuracy_measure.reset_labels()
        loss_online = MeanOnline()

        if train:
            loader = self.data_loader.eval(
                description="Full forward pass (eval)")
            self.mutual_info.start_listening()
        else:
            loader = self.data_loader.get(train)

        with torch.no_grad():
            for batch in loader:
                batch = batch_to_cuda(batch)
                output = self._forward(batch)
                loss = self._get_loss(batch, output)
                self._on_forward_pass_batch(batch, output, train)
                loss_online.update(loss)

        self.mutual_info.finish_listening()
        self.model.train(mode_saved)

        loss = loss_online.get_mean()
        self.monitor.update_loss(loss, mode='train' if train else 'test')
        self.update_accuracy(train=train)

        return loss

    def _epoch_finished(self, loss):
        self.save()
        for online_measure in self.online.values():
            online_measure.reset()
        self.accuracy_measure.reset()

    def update_accuracy(self, train=True):
        """
        Updates the accuracy of the model.

        Parameters
        ----------
        train : bool
            Either train (True) or test (False) mode.

        Returns
        -------
        accuracy : torch.Tensor
            A scalar with the accuracy value.

        """
        if self.is_unsupervised():
            return None
        labels_true = torch.cat(self.accuracy_measure.true_labels_cached)
        if not train or isinstance(self.accuracy_measure, AccuracyArgmax):
            labels_pred = torch.cat(
                self.accuracy_measure.predicted_labels_cached)
        elif getattr(self.accuracy_measure, 'cache', False):
            labels_pred = self.accuracy_measure.predict_cached()
        else:
            labels_pred = []
            with torch.no_grad():
                for batch in self.data_loader.eval():
                    batch = batch_to_cuda(batch)
                    output = self._forward(batch)
                    if isinstance(output, AutoencoderOutput):
                        output = output.latent
                    labels_pred.append(
                        self.accuracy_measure.predict(output).cpu())
            labels_pred = torch.cat(labels_pred, dim=0)

        if labels_true.is_cuda:
            warnings.warn("labels_true is cuda")
            labels_true = labels_true.cpu()
        if labels_pred.is_cuda:
            warnings.warn("labels_pred is cuda")
            labels_pred = labels_pred.cpu()

        accuracy = self.monitor.update_accuracy_epoch(
            labels_pred, labels_true, mode='train' if train else 'test')
        self.update_best_score(accuracy, score_type='accuracy')

        return accuracy

    def train_mask(self, mask_explain_params=dict()):
        """
        Train mask to see what part of an image is crucial from the network
        perspective (saliency map).

        Parameters
        ----------
        mask_explain_params : dict, optional
            `MaskTrainer` keyword arguments.
        """
        images, labels = next(iter(self.train_loader))
        mask_trainer = MaskTrainer(self.accuracy_measure,
                                   image_shape=images[0].shape,
                                   **mask_explain_params)
        mode_saved = prepare_eval(self.model)
        if torch.cuda.is_available():
            images = images.cuda()
        with torch.no_grad():
            proba = self.accuracy_measure.predict_proba(self.model(images))
        proba_max, _ = proba.max(dim=1)
        sample_max_proba = proba_max.argmax()
        image = images[sample_max_proba]
        label = labels[sample_max_proba]
        self.monitor.plot_explain_input_mask(self.model, mask_trainer=mask_trainer,
                                             image=image, label=label)
        mode_saved.restore(self.model)
        return image, label

    def train_epoch(self, epoch):
        """
        Trains an epoch.

        Parameters
        ----------
        epoch : int
            Epoch ID.

        """
        loss_online = MeanOnline()
        for batch in tqdm(self.train_loader,
                                   desc="Epoch {:d}".format(epoch),
                                   leave=False):
            batch = batch_to_cuda(batch)
            loss = self.train_batch(batch)
            loss_online.update(loss.detach().cpu())
            for name, param in self.model.named_parameters():
                if torch.isnan(param).any():
                    warnings.warn(f"NaN parameters in '{name}'")
            self.monitor.batch_finished(self.model)

        self.monitor.update_loss(loss=loss_online.get_mean(),
                                 mode='batch')

    def open_monitor(self, offline=False):
        """
        Opens a `Visdom` monitor.

        Parameters
        ----------
        offline : bool
            Online (False) or offline (True) monitoring.

        """
        # visdom can be already initialized via trainer.restore()
        if self.monitor.viz is None:
            # new environment
            self.monitor.open(env_name=self.env_name, offline=offline)
            self.monitor.clear()

    def train(self, n_epochs=10, mutual_info_layers=0,
              mask_explain_params=None):
        """
        User-entry function to train the model for :code:`n_epochs`.

        Parameters
        ----------
        n_epochs : int
            The number of epochs to run.
            Default: 10
        mutual_info_layers : int, optional
            Evaluate the mutual information [1]_ from the last
            :code:`mutual_info_layers` layers at each epoch. If set to 0,
            skip the (time-consuming) mutual information estimation.
            Default: 0
        mask_explain_params : dict or None, optional
            If not None, a dictionary with parameters for :class:`MaskTrainer`,
            that is used to show the "saliency map" [2]_.
            Default: None

        Returns
        -------
        loss_epochs : list
            A list of epoch loss.

        References
        ----------
        .. [1] Shwartz-Ziv, R., & Tishby, N. (2017). Opening the black box of deep
           neural networks via information. arXiv preprint arXiv:1703.00810.
        .. [1] Fong, R. C., & Vedaldi, A. (2017). Interpretable explanations of
           black boxes by meaningful perturbation.

        """
        print(self.model)
        self.open_monitor()
        if n_epochs == 1:
            self.monitor.viz.with_markers = True
        self.monitor_functions()
        self.log_trainer()
        for name, layer in find_named_layers(self.model,
                                             layer_class=self.watch_modules):
            self.monitor.register_layer(layer, prefix=name)

        if mutual_info_layers > 0 and not isinstance(self.mutual_info,
                                                     MutualInfoStub):
            # Mutual Information in conv layers are poorly estimated
            monitor_classes = set(self.watch_modules).difference({nn.Conv2d})
            self.mutual_info.prepare(model=self.model,
                                     monitor_layers=tuple(monitor_classes),
                                     monitor_layers_count=mutual_info_layers)

        loss_epochs = []
        for epoch in range(self.timer.epoch, self.timer.epoch + n_epochs):
            self.train_epoch(epoch=epoch)
            loss = self.full_forward_pass(train=True)
            self.full_forward_pass(train=False)
            self.monitor.epoch_finished()
            if mask_explain_params:
                self.train_mask(mask_explain_params)
            self._epoch_finished(loss)
            loss_epochs.append(loss.item())
        return loss_epochs
