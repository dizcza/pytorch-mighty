import time
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from mighty.loss import PairLoss
from mighty.monitor.accuracy import AccuracyEmbedding, \
    AccuracyArgmax, Accuracy, calc_accuracy
from mighty.monitor.batch_timer import timer
from mighty.monitor.monitor import Monitor
from mighty.monitor.mutual_info import MutualInfoKMeans, MutualInfoStub
from mighty.monitor.var_online import MeanOnline
from mighty.trainer.mask import MaskTrainer
from mighty.utils.common import find_named_layers, how_many_samples_take
from mighty.utils.constants import CHECKPOINTS_DIR
from mighty.utils.data import DataLoader, get_normalize_inverse
from mighty.utils.domain import AdversarialExamples
from mighty.utils.prepare import prepare_eval


class Trainer(ABC):
    watch_modules = (nn.Linear, nn.Conv2d)

    def __init__(self, model: nn.Module, criterion: nn.Module,
                 data_loader: DataLoader, accuracy_measure: Accuracy = None,
                 mutual_info=MutualInfoKMeans(),
                 env_suffix='', checkpoint_dir=CHECKPOINTS_DIR):
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
            model = model.cuda()
        self.model = model
        self.criterion = criterion
        self.data_loader = data_loader
        self.train_loader = data_loader.get(train=True)
        self.mutual_info = mutual_info
        self.checkpoint_dir = Path(checkpoint_dir)
        self.timer = timer
        self.timer.init(batches_in_epoch=len(self.train_loader))
        self.env_name = f"{time.strftime('%Y.%m.%d')} " \
                        f"{model.__class__.__name__}: " \
                        f"{data_loader.dataset_cls.__name__} " \
                        f"{self.__class__.__name__} " \
                        f"{criterion.__class__.__name__}"
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

        images, labels = next(iter(self.train_loader))
        self.mask_trainer = MaskTrainer(accuracy_measure=self.accuracy_measure,
                                        image_shape=images[0].shape)

    @property
    def checkpoint_path(self):
        return self.checkpoint_dir / (self.env_name + '.pt')

    def monitor_functions(self):
        pass

    def log_trainer(self):
        self.monitor.log(f"Criterion: {self.criterion}")
        self.monitor.log(repr(self.mask_trainer))

    def _init_monitor(self, mutual_info) -> Monitor:
        normalize_inverse = get_normalize_inverse(self.data_loader.normalize)
        monitor = Monitor(
            accuracy_measure=self.accuracy_measure,
            mutual_info=mutual_info,
            normalize_inverse=normalize_inverse
        )
        return monitor

    @abstractmethod
    def train_batch(self, images, labels):
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

    def eval_batches(self):
        loader = self.data_loader.eval
        n_samples_take = how_many_samples_take(loader)
        n_taken = 0
        for images, labels in iter(loader):
            if n_taken > n_samples_take:
                break
            n_taken += len(labels)
            yield images, labels

    def _get_loss(self, input, output, labels):
        return self.criterion(output, labels)

    def _on_forward_pass_batch(self, input, output, labels):
        self.accuracy_measure.partial_fit(output, labels)

    def full_forward_pass(self, cache=False):
        mode_saved = self.model.training
        self.model.train(False)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.model.cuda()
        loss_online = MeanOnline()
        self.accuracy_measure.reset()

        outputs_full = []
        labels_full = []
        with torch.no_grad():
            for inputs, labels in self.eval_batches():
                if use_cuda:
                    inputs = inputs.cuda()
                outputs = self.model(inputs)
                labels_full.append(labels)
                if cache:
                    outputs_full.append(outputs.cpu())
                loss = self._get_loss(inputs, outputs, labels)
                self._on_forward_pass_batch(inputs, outputs, labels)
                loss_online.update(loss)
        labels_full = torch.cat(labels_full, dim=0)

        if cache:
            outputs_full = torch.cat(outputs_full, dim=0)
            labels_pred = self.accuracy_measure.predict(outputs_full)
        else:
            labels_pred = []
            with torch.no_grad():
                for inputs, labels in self.eval_batches():
                    if use_cuda:
                        inputs = inputs.cuda()
                    outputs = self.model(inputs)
                    labels_pred.append(self.accuracy_measure.predict(outputs))
            labels_pred = torch.cat(labels_pred, dim=0)

        loss = loss_online.get_mean()

        self.monitor.update_accuracy_epoch(labels_pred, labels_full,
                                           mode='full train')
        self.monitor.update_loss(loss, mode='full train')

        self.model.train(mode_saved)

        return loss

    def full_forward_pass_test(self):
        mode_saved = self.model.training
        self.model.train(False)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.model.cuda()
        test_loader = self.data_loader.get(train=False)

        labels_full = []
        labels_pred = []
        with torch.no_grad():
            for inputs, labels in iter(test_loader):
                if use_cuda:
                    inputs = inputs.cuda()
                outputs = self.model(inputs)
                labels_pred.append(self.accuracy_measure.predict(outputs))
                labels_full.append(labels)
        labels_pred = torch.cat(labels_pred, dim=0)
        labels_full = torch.cat(labels_full, dim=0)

        self.monitor.update_accuracy_epoch(labels_pred, labels_full,
                                           mode='full test')
        self.model.train(mode_saved)
        accuracy = calc_accuracy(labels_full, labels_pred)
        return accuracy

    def _epoch_finished(self, epoch, loss):
        self.save()

    def train_mask(self):
        """
        Train mask to see what part of the image is crucial from the network perspective.
        """
        images, labels = next(iter(self.train_loader))
        mode_saved = prepare_eval(self.model)
        if torch.cuda.is_available():
            images = images.cuda()
        with torch.no_grad():
            proba = self.accuracy_measure.predict_proba(self.model(images))
        proba_max, _ = proba.max(dim=1)
        sample_max_proba = proba_max.argmax()
        image = images[sample_max_proba]
        label = labels[sample_max_proba]
        self.monitor.plot_mask(self.model, mask_trainer=self.mask_trainer,
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
            loss = self._get_loss(images, outputs, labels)
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
        use_cuda = torch.cuda.is_available()
        for images, labels in tqdm(self.train_loader,
                                   desc="Epoch {:d}".format(epoch),
                                   leave=False):
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()

            outputs, loss = self.train_batch(images, labels)
            loss_online.update(loss.detach().cpu())
            for name, param in self.model.named_parameters():
                if torch.isnan(param).any():
                    warnings.warn(f"NaN parameters in '{name}'")
            self.monitor.batch_finished(self.model)

        self.monitor.update_loss(loss=loss_online.get_mean(),
                                 mode='batch')

    def train(self, n_epoch=10, epoch_update_step=1, mutual_info_layers=1,
              adversarial=False, mask_explain=False, cache=False):
        """
        :param n_epoch: number of training epochs
        :param epoch_update_step: epoch step to run full evaluation
        :param mutual_info_layers: number of last layers to be monitored for mutual information;
                                   pass '0' to turn off this feature.
        :param adversarial: perform adversarial attack test?
        :param mask_explain: train the image mask that 'explains' network behaviour?
        """
        print(self.model)
        if not self.monitor.is_active:
            # new environment
            self.monitor.open(env_name=self.env_name)
            self.monitor.clear()
        self.monitor_functions()
        self.monitor.log_model(self.model)
        self.monitor.log_self()
        self.log_trainer()
        for name, layer in find_named_layers(self.model,
                                             layer_class=self.watch_modules):
            self.monitor.register_layer(layer, prefix=name)

        if mutual_info_layers > 0 and not isinstance(self.mutual_info,
                                                     MutualInfoStub):
            self.full_forward_pass = self.mutual_info.decorate_evaluation(
                self.full_forward_pass)
            self.mutual_info.prepare(loader=self.data_loader.eval,
                                     model=self.model,
                                     monitor_layers_count=mutual_info_layers)

        print(f"Training '{self.model.__class__.__name__}'")

        for epoch in range(self.timer.epoch, self.timer.epoch + n_epoch):
            self.train_epoch(epoch=epoch)
            if epoch % epoch_update_step == 0:
                loss = self.full_forward_pass(cache=cache)
                self.full_forward_pass_test()
                self.monitor.epoch_finished()
                if adversarial:
                    self.monitor.plot_adversarial_examples(
                        self.model,
                        self.get_adversarial_examples())
                if mask_explain:
                    self.train_mask()
                self._epoch_finished(epoch, loss)

    def run_idle(self, n_epoch=10):
        """
        Burn-out mode without returning the output.
        Useful in combination with `DumpActivationsHook`.

        :param n_epoch: number of epochs to run in idle
        """
        self.model.eval()
        for epoch in range(self.timer.epoch, self.timer.epoch + n_epoch):
            use_cuda = torch.cuda.is_available()
            for images, labels in tqdm(self.train_loader,
                                       desc="Epoch {:d}".format(epoch),
                                       leave=False):
                if use_cuda:
                    images = images.cuda()
                self.model(images)
                self.timer.tick()
        self.model.train()
