from abc import ABC
from collections import defaultdict

import torch
import torch.utils.data

from mighty.monitor.var_online import MeanOnlineBatch
from mighty.utils.algebra import compute_distance


def calc_accuracy(labels_true, labels_predicted) -> float:
    accuracy = (labels_true == labels_predicted).type(torch.float32).mean()
    return accuracy.item()


class Accuracy(ABC):

    def reset(self):
        pass

    def partial_fit(self, outputs_batch, labels_batch):
        """
        If accuracy measure is not argmax (if the model doesn't end with a softmax layer),
        the output is embedding vector, which has to be stored and retrieved at prediction.
        :param outputs_train: model output on the train set
        :param labels_train: train set labels
        """
        pass

    def predict(self, outputs_test):
        """
        :param outputs_test: model output on the train or test set
        :return: predicted labels of shape (N,)
        """
        return self.predict_proba(outputs_test).argmax(dim=1)

    def predict_proba(self, outputs_test):
        """
        :param outputs_test: model output on the train or test set
        :return: predicted probabilities tensor of shape (N x C),
                 where C is the number of classes
        """
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}({self.extra_repr()})"

    def extra_repr(self):
        return ''


class AccuracyArgmax(Accuracy):

    def predict(self, outputs_test):
        labels_predicted = outputs_test.argmax(dim=-1)
        return labels_predicted

    def predict_proba(self, outputs_test):
        return outputs_test.softmax(dim=1)


class AccuracyEmbedding(Accuracy):
    """
    Calculates the accuracy of embedding vectors.
    The mean embedding vector is kept for each class.
    Prediction is based on the closest centroid ID.
    """

    def __init__(self, metric='cosine', cache=False):
        self.metric = metric
        self.cache = cache
        self.input_cached = []
        self.centroids_dict = defaultdict(MeanOnlineBatch)

    @property
    def centroids(self):
        centroids = tuple(c.get_mean() for c in self.centroids_dict.values())
        centroids = torch.stack(centroids, dim=0)
        return centroids

    def reset(self):
        self.centroids_dict.clear()
        self.input_cached.clear()

    def extra_repr(self):
        return f'metric={self.metric}'

    def distances(self, outputs_test):
        """
        :param outputs_test: (B, D) embeddings tensor
        :return: (B, n_classes) distance matrix to each centroid
        """
        assert len(self.centroids_dict) > 0, "Fit the classifier first"
        centroids = torch.as_tensor(self.centroids, device=outputs_test.device)
        distances = []
        outputs_test = outputs_test.unsqueeze(dim=1)  # (B, 1, D)
        centroids = centroids.unsqueeze(dim=0)  # (1, n_classes, D)
        for centroids_chunk in centroids.split(split_size=50, dim=1):
            # memory efficient
            distances_chunk = compute_distance(input1=outputs_test,
                                               input2=centroids_chunk,
                                               metric=self.metric, dim=2)
            distances.append(distances_chunk)
        distances = torch.cat(distances, dim=1)
        return distances

    def partial_fit(self, outputs_batch, labels_batch):
        outputs_batch = outputs_batch.detach().cpu()
        for label in labels_batch.unique(sorted=True):
            self.centroids_dict[label.item()].update(
                outputs_batch[labels_batch == label]
            )
        if self.cache:
            self.input_cached.append(outputs_batch)

    def predict_cached(self):
        if not self.cache:
            raise ValueError("Caching is turned off")
        if len(self.input_cached) == 0:
            raise ValueError("Empty cached input buffer")
        input = torch.cat(self.input_cached,  dim=0)
        return self.predict(input)

    def predict(self, outputs_test):
        argmin = self.distances(outputs_test).argmin(dim=1).cpu()
        labels_stored = tuple(self.centroids_dict.keys())
        labels_stored = torch.IntTensor(labels_stored)
        labels_predicted = labels_stored[argmin]
        return labels_predicted

    def predict_proba(self, outputs_test):
        distances = self.distances(outputs_test)
        proba = 1 - distances / distances.sum(dim=1).unsqueeze(1)
        return proba
