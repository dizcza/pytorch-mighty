import math
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

from mighty.utils.common import clone_cpu
from mighty.utils.data import DataLoader
from mighty.utils.hooks import get_layers_ordered


class AccuracyFromMutualInfo:
    def __init__(self, n_classes: int, resolution_bins=100):
        self.n_classes = n_classes
        self.accuracy_binned = np.linspace(start=1 / n_classes, stop=1,
                                           num=resolution_bins, endpoint=False)
        entropy_correct = self.accuracy_binned * np.log2(
            1 / self.accuracy_binned)
        entropy_incorrect = (1 - self.accuracy_binned) * np.log2(
            (n_classes - 1) / (1 - self.accuracy_binned))
        entropy_class_given_activations = entropy_correct + entropy_incorrect
        entropy_classes = np.log2(n_classes)
        self.mutual_info_binned = entropy_classes - \
                                  entropy_class_given_activations

    def estimate_accuracy(self, mutual_info_bits: float) -> float:
        """
        :param mutual_info_bits: mutual information between the hidden layer T and the true class Y
        :return: estimated layer accuracy
        """
        bin_id = np.digitize(mutual_info_bits, bins=self.mutual_info_binned)
        bin_id = min(bin_id, len(self.accuracy_binned) - 1)
        accuracy_estimated = self.accuracy_binned[bin_id]
        return accuracy_estimated


class MutualInfo(ABC):
    log2e = math.log2(math.e)
    ignore_layers = (nn.Conv2d,)  # poor estimate

    def __init__(self, data_loader: DataLoader, debug=False):
        """
        :param debug: plot bins distribution?
        """
        self.data_loader = data_loader
        self.debug = debug
        self.activations = defaultdict(list)
        self.quantized = {}
        self.information = {}
        self.is_active = False
        self.is_updating = False
        self.layer_to_name = {}
        self.accuracy_estimator = None

    def __repr__(self):
        return f"{self.__class__.__name__}({self.extra_repr()})"

    def extra_repr(self):
        return ""

    def register(self, layer: nn.Module, name: str):
        if not isinstance(layer, self.ignore_layers):
            self.layer_to_name[layer] = name

    def force_update(self, model: nn.Module):
        if not self.is_active:
            return
        self.start_listening()
        use_cuda = torch.cuda.is_available()
        with torch.no_grad():
            for images, labels in self.data_loader.eval():
                if use_cuda:
                    images = images.cuda()
                # the output of each layer is saved implicitly via hooks
                model(images)
        self.finish_listening()

    @abstractmethod
    def prepare_input(self):
        pass

    def prepare(self, model: nn.Module, monitor_layers_count=1):
        self.is_active = True  # turn on the feature
        self.prepare_input()
        images_batch, _ = self.data_loader.sample()
        image_sample = images_batch[0]

        layers_ordered = get_layers_ordered(model, image_sample)
        layers_ordered = list(layer for layer in layers_ordered
                              if layer in self.layer_to_name)
        layers_ordered = layers_ordered[-monitor_layers_count:]

        for layer in layers_ordered:
            layer.register_forward_hook(self.save_activations)

        monitored_layer_names = tuple(self.layer_to_name[layer]
                                      for layer in layers_ordered)
        print(f"Monitoring only these last layers for mutual information "
              f"estimation: {monitored_layer_names}")

    def start_listening(self):
        self.activations.clear()
        self.is_updating = True

    def finish_listening(self):
        self.is_updating = False
        for hname, activations in self.activations.items():
            self.process_activations(layer_name=hname, activations=activations)
        self.save_mutual_info()

    def save_activations(self, module: nn.Module, tensor_input, tensor_output):
        if not self.is_updating:
            return
        layer_name = self.layer_to_name[module]
        tensor_output_clone = clone_cpu(tensor_output)
        tensor_output_clone = tensor_output_clone.flatten(start_dim=1)
        self.activations[layer_name].append(tensor_output_clone)

    def plot_activations_hist(self, viz):
        for hname, activations in self.activations.items():
            title = f'Activations histogram: {hname}'
            activations = torch.cat(activations, dim=0)
            viz.histogram(activations.flatten(), win=title, opts=dict(
                xlabel='neuron value',
                ylabel='neuron counts',
                title=title,
            ))

    def _plot_debug(self, viz):
        self.plot_activations_hist(viz)

    def plot(self, viz):
        assert not self.is_updating, "Wait, not finished yet."
        if len(self.information) == 0:
            return
        if self.debug:
            self._plot_debug(viz)
        legend = []
        info_hidden_input = []
        info_hidden_output = []
        for layer_name, (info_x, info_y) in self.information.items():
            info_hidden_input.append(info_x)
            info_hidden_output.append(info_y)
            legend.append(layer_name)
        title = 'Mutual information plane'
        viz.line(Y=np.array([info_hidden_output]),
                 X=np.array([info_hidden_input]), win=title, opts=dict(
                xlabel='I(X, T), bits',
                ylabel='I(T, Y), bits',
                title=title,
                legend=legend,
            ), update='append' if viz.win_exists(title) else None)
        self.information.clear()
        self.activations.clear()

    @abstractmethod
    def process_activations(self, layer_name: str,
                            activations: List[torch.FloatTensor]):
        pass

    @abstractmethod
    def save_mutual_info(self):
        pass

    def estimate_accuracy(self):
        accuracies = {}
        for layer_name, (info_x, info_y) in self.information.items():
            accuracies[layer_name] = self.accuracy_estimator.estimate_accuracy(
                info_y)
        return accuracies
