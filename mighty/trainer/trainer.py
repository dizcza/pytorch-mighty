import subprocess
import time
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from mighty.loss import PairLoss
from mighty.monitor.accuracy import AccuracyEmbedding, \
    AccuracyArgmax, Accuracy
from mighty.monitor.batch_timer import timer
from mighty.monitor.monitor import Monitor
from mighty.monitor.mutual_info import MutualInfoStub
from mighty.monitor.var_online import MeanOnline
from mighty.trainer.mask import MaskTrainer
from mighty.utils.common import find_named_layers, batch_to_cuda, \
    input_from_batch
from mighty.utils.constants import CHECKPOINTS_DIR
from mighty.utils.data import DataLoader
from mighty.utils.domain import AdversarialExamples
from mighty.utils.prepare import prepare_eval


class Trainer(ABC):
    watch_modules = (nn.Linear, nn.Conv2d)

    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 data_loader: DataLoader,
                 accuracy_measure: Accuracy = None,
                 mutual_info=None,
                 env_suffix='',
                 checkpoint_dir=CHECKPOINTS_DIR):
        """
        :param model: NN model
        :param criterion: loss function
        :param dataset_name: one of "MNIST", "CIFAR10", "Caltech256"
        :param accuracy_measure: depending on the loss function, the predicted label could be either
                                 - argmax (cross-entropy loss)
                                 - closest centroid ID (triplet loss)
        :param env_suffix: monitor environment suffix
        :param checkpoint_dir: path to the directory where model checkpoints will be stored
        :param mutual_info: mutual information estimator
        """
        if torch.cuda.is_available():
            model.cuda()
        self.model = model
        self.criterion = criterion
        self.data_loader = data_loader
        self.train_loader = data_loader.get(train=True)
        if mutual_info is None:
            mutual_info = MutualInfoStub()
        self.mutual_info = mutual_info
        self.checkpoint_dir = Path(checkpoint_dir)
        self.timer = timer
        self.timer.init(batches_in_epoch=len(self.train_loader))
        self.env_name = f"{time.strftime('%Y.%m.%d')} " \
                        f"{model.__class__.__name__}: " \
                        f"{data_loader.dataset_cls.__name__} " \
                        f"{self.__class__.__name__} " \
                        f"{criterion.__class__.__name__}"
        env_suffix = env_suffix.lstrip(' ')
        if env_suffix:
            self.env_name = f'{self.env_name} {env_suffix}'
        if accuracy_measure is None:
            if isinstance(self.criterion, PairLoss):
                accuracy_measure = AccuracyEmbedding(self.criterion.metric)
            else:
                # cross entropy loss
                accuracy_measure = AccuracyArgmax()
        self.accuracy_measure = accuracy_measure
        self.monitor = self._init_monitor(mutual_info)
        self.online = self._init_online_measures()

    @property
    def checkpoint_path(self):
        return self.checkpoint_dir / (self.env_name + '.pt')

    def monitor_functions(self):
        pass

    def log_trainer(self):
        self.monitor.log_model(self.model)
        self.monitor.log(f"Criterion: {self.criterion}")
        self.monitor.log(repr(self.data_loader))
        self.monitor.log_self()
        self.monitor.log(repr(self.accuracy_measure))
        self.monitor.log(repr(self.mutual_info))
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'],
                                stdout=subprocess.PIPE,
                                universal_newlines=True)
        self.monitor.log(f"Git commit: {commit.stdout}")

    def _init_monitor(self, mutual_info) -> Monitor:
        monitor = Monitor(
            accuracy_measure=self.accuracy_measure,
            mutual_info=mutual_info,
            normalize_inverse=self.data_loader.normalize_inverse
        )
        return monitor

    def _init_online_measures(self) -> Dict[str, MeanOnline]:
        return dict()

    @abstractmethod
    def train_batch(self, batch):
        raise NotImplementedError()

    def save(self):
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(self.state_dict(), self.checkpoint_path)
        except PermissionError as error:
            print(error)

    def state_dict(self):
        return {
            "model_state": self.model.state_dict(),
            "epoch": self.timer.epoch,
            "env_name": self.env_name,
        }

    def restore(self, checkpoint_path=None, strict=True):
        """
        :param checkpoint_path: train checkpoint path to restore
        :param strict: model's load_state_dict strict argument
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path
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
        self.monitor.open(env_name=self.env_name)
        print(f"Restored model state from {checkpoint_path}.")
        return checkpoint_state

    def _get_loss(self, batch, output):
        raise NotImplementedError()

    def _on_forward_pass_batch(self, batch, output):
        pass

    def _forward(self, batch):
        input = input_from_batch(batch)
        return self.model(input)

    def full_forward_pass(self, train=True):
        mode_saved = self.model.training
        self.model.train(False)
        loss_online = MeanOnline()

        if train:
            loader = self.data_loader.eval(verbose=True)
            self.mutual_info.start_listening()
        else:
            loader = self.data_loader.get(train)

        with torch.no_grad():
            for batch in loader:
                batch = batch_to_cuda(batch)
                output = self._forward(batch)
                loss = self._get_loss(batch, output)
                if train:
                    # evaluation
                    self._on_forward_pass_batch(batch, output)
                loss_online.update(loss)

        loss = loss_online.get_mean()
        self.monitor.update_loss(loss, mode='train' if train else 'test')

        self.mutual_info.finish_listening()
        self.model.train(mode_saved)

        return loss

    def _epoch_finished(self, loss):
        self.save()
        for online_measure in self.online.values():
            online_measure.reset()
        self.accuracy_measure.reset()

    def train_mask(self):
        """
        Train mask to see what part of the image is crucial from the network perspective.
        """
        images, labels = next(iter(self.train_loader))
        mask_trainer = MaskTrainer(self.accuracy_measure,
                                   image_shape=images[0].shape)
        mode_saved = prepare_eval(self.model)
        if torch.cuda.is_available():
            images = images.cuda()
        with torch.no_grad():
            proba = self.accuracy_measure.predict_proba(self.model(images))
        proba_max, _ = proba.max(dim=1)
        sample_max_proba = proba_max.argmax()
        image = images[sample_max_proba]
        label = labels[sample_max_proba]
        self.monitor.plot_mask(self.model, mask_trainer=mask_trainer,
                               image=image, label=label)
        mode_saved.restore(self.model)
        return image, label

    def get_adversarial_examples(self, noise_ampl=100, n_iter=10):
        """
        :param noise_ampl: adversarial noise amplitude
        :param n_iter: adversarial iterations
        :return adversarial examples
        """
        images, labels = next(iter(self.train_loader))
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        images_orig = images.clone()
        images.requires_grad_(True)
        mode_saved = prepare_eval(self.model)
        for i in range(n_iter):
            images.grad = None  # reset gradients tensor
            outputs = self.model(images)
            loss = self._get_loss((images, labels), outputs)
            loss.backward()
            with torch.no_grad():
                adv_noise = noise_ampl * images.grad
                images += adv_noise
        images.requires_grad_(False)
        mode_saved.restore(self.model)
        return AdversarialExamples(original=images_orig, adversarial=images,
                                   labels=labels)

    def train_epoch(self, epoch):
        """
        :param epoch: epoch id
        :return: last batch loss
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

    def _open_monitor(self):
        if not self.monitor.is_active:
            # new environment
            self.monitor.open(env_name=self.env_name)
            self.monitor.clear()

    def train(self, n_epochs=10, epoch_update_step=1, mutual_info_layers=1,
              adversarial=False, mask_explain=False):
        """
        :param n_epochs: number of training epochs
        :param epoch_update_step: epoch step to run full evaluation
        :param mutual_info_layers: number of last layers to be monitored for mutual information;
                                   pass '0' to turn off this feature.
        :param adversarial: perform adversarial attack test?
        :param mask_explain: train the image mask that 'explains' network behaviour?
        """
        print(self.model)
        self._open_monitor()
        if n_epochs == 1:
            self.monitor.viz.with_markers = True
        self.monitor_functions()
        self.log_trainer()
        for name, layer in find_named_layers(self.model,
                                             layer_class=self.watch_modules):
            self.monitor.register_layer(layer, prefix=name)

        if mutual_info_layers > 0 and not isinstance(self.mutual_info,
                                                     MutualInfoStub):
            self.mutual_info.prepare(loader=self.data_loader.eval(),
                                     model=self.model,
                                     monitor_layers_count=mutual_info_layers)

        print(f"Training '{self.model.__class__.__name__}'")

        for epoch in range(self.timer.epoch, self.timer.epoch + n_epochs):
            self.train_epoch(epoch=epoch)
            if epoch % epoch_update_step == 0:
                loss = self.full_forward_pass(train=True)
                self.full_forward_pass(train=False)
                self.monitor.epoch_finished()
                if adversarial:
                    self.monitor.plot_adversarial_examples(
                        self.model,
                        self.get_adversarial_examples())
                if mask_explain:
                    self.train_mask()
                # TODO: epoch_finished should be called on each epoch
                # independently of the update step
                self._epoch_finished(loss)
