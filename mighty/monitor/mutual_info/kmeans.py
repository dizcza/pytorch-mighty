from typing import List

import numpy as np
import torch
import torch.utils.data
import torch.utils.data
from sklearn import cluster
from sklearn.metrics import mutual_info_score
from tqdm import tqdm

from mighty.monitor.mutual_info.mutual_info import MutualInfo, AccuracyFromMutualInfo
from mighty.utils.constants import BATCH_SIZE


class MutualInfoKMeans(MutualInfo):
    """
    Classical binning approach to estimate the mutual information.
    KMeans is used to cluster the data.
    """

    def __init__(self, estimate_size=None, n_bins=20, debug=False):
        """
        :param estimate_size: number of samples to estimate mutual information from
        :param n_bins: how many bins to use? This value should be no less than the number of classes.
                       But if the estimate_size per bin (or rather class) is small, estimation suffers.
                       Setting n_bins to None means it'll be calculated as the number of distinct labels/targets.
        :param debug: plot bins distribution?
        """
        super().__init__(estimate_size=estimate_size, debug=debug)
        self.n_bins = n_bins

    def extra_repr(self):
        return f"{super().extra_repr()}, n_bins={self.n_bins}"

    def prepare_input(self):
        targets = []
        classifier = cluster.MiniBatchKMeans(n_clusters=self.n_bins,
                                             batch_size=BATCH_SIZE,
                                             compute_labels=False)
        for images, labels in tqdm(self.eval_batches(), total=len(self.eval_loader),
                                   desc="MutualInfo: quantizing input data. Stage 1"):
            images = images.flatten(start_dim=1)
            classifier.partial_fit(images, labels)
            targets.append(labels)
        targets = torch.cat(targets, dim=0)
        self.quantized['target'] = targets.numpy()

        centroids_predicted = []
        for images, _ in tqdm(self.eval_batches(), total=len(self.eval_loader),
                              desc="MutualInfo: quantizing input data. Stage 2"):
            images = images.flatten(start_dim=1)
            centroids_predicted.append(classifier.predict(images))
        self.quantized['input'] = np.hstack(centroids_predicted)

        n_classes = len(targets.unique(sorted=False))
        self.accuracy_estimator = AccuracyFromMutualInfo(n_classes=n_classes)

        if self.n_bins is None:
            unique_targets = np.unique(self.quantized['target'])
            self.n_bins = len(unique_targets)

    def process_activations(self, layer_name: str, activations: List[torch.FloatTensor]):
        assert self.n_bins is not None, "Set n_bins manually"
        activations = torch.cat(activations, dim=0)
        self.quantized[layer_name] = self.quantize(activations)

    def save_mutual_info(self):
        hidden_layers_name = set(self.quantized.keys())
        hidden_layers_name.difference_update({'input', 'target'})
        for layer_name in hidden_layers_name:
            info_x = self.compute_mutual_info(self.quantized['input'], self.quantized[layer_name])
            info_y = self.compute_mutual_info(self.quantized['target'], self.quantized[layer_name])
            self.information[layer_name] = (info_x, info_y)

    @staticmethod
    def compute_mutual_info(x, y) -> float:
        return mutual_info_score(x, y) * MutualInfo.log2e

    def quantize(self, activations: torch.FloatTensor) -> np.ndarray:
        model = cluster.MiniBatchKMeans(n_clusters=self.n_bins, batch_size=BATCH_SIZE)
        labels = model.fit_predict(activations)
        return labels

    def plot_quantized_hist(self, viz):
        """
        Plots quantized bins distribution.
        Ideally, we'd like the histogram to match a uniform distribution.
        """
        for layer_name in self.quantized.keys():
            if layer_name != 'target':
                _, counts = np.unique(self.quantized[layer_name], return_counts=True)
                n_empty_clusters = max(0, self.n_bins - len(counts))
                counts = np.r_[counts, np.zeros(n_empty_clusters, dtype=int)]
                counts.sort()
                counts = counts[::-1]
                title = f'MI quantized histogram: {layer_name}'
                viz.bar(X=counts, win=title, opts=dict(
                    xlabel='bin ID',
                    ylabel='# activation codes',
                    title=title,
                ))

    def _plot_debug(self, viz):
        super()._plot_debug(viz)
        self.plot_quantized_hist(viz)
