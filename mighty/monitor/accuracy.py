"""
Accuracy measures
-----------------

.. autosummary::
    :toctree: toctree/monitor

    AccuracyArgmax
    AccuracyEmbedding
"""


from abc import ABC

import torch
import torch.utils.data

from mighty.monitor.var_online import MeanOnlineLabels
from mighty.utils.signal import compute_distance


def calc_accuracy(labels_true, labels_predicted) -> float:
    accuracy = (labels_true == labels_predicted).float().mean()
    return accuracy.item()


class Accuracy(ABC):

    def __init__(self):
        self.true_labels_cached = []
        self.predicted_labels_cached = []

    def reset(self):
        """
        Resets all cached predicted and ground truth data.
        """
        self.reset_labels()

    def reset_labels(self):
        """
        Resets predicted and ground truth **labels**.
        """
        self.true_labels_cached.clear()
        self.predicted_labels_cached.clear()

    def partial_fit(self, outputs_batch, labels_batch):
        """
        If the accuracy measure is not argmax (if the model's last layer isn't
        a softmax), the output is an embedding vector, which has to be stored
        and retrieved at prediction.

        Parameters
        ----------
        outputs_batch : torch.Tensor or tuple
            The output of a model.
        labels_batch : torch.Tensor
            True labels.
        """
        self.true_labels_cached.append(labels_batch.cpu())

    def predict(self, outputs_test):
        """
        Predict the labels, given model output.

        Parameters
        ----------
        output_test : torch.Tensor or tuple
            The output of a model.

        Returns
        -------
        torch.Tensor
            Predicted labels.
        """
        return self.predict_proba(outputs_test).argmax(dim=1)

    def predict_proba(self, outputs_test):
        """
        Compute label probabilities, given model output.

        Parameters
        ----------
        output_test : torch.Tensor or tuple
            The output of a model.

        Returns
        -------
        torch.Tensor
            The probabilities of assigning to each class of shape `(., C)`,
            where C is the number of classes.
        """
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}({self.extra_repr()})"

    def extra_repr(self):
        return ''


class AccuracyArgmax(Accuracy):
    """
    Softmax accuracy.

    The predicted labels are simply ``output.argmax(dim=-1)``.
    """

    def predict(self, outputs_test):
        labels_predicted = outputs_test.argmax(dim=-1)
        return labels_predicted

    def predict_proba(self, outputs_test):
        return outputs_test.softmax(dim=1)

    def partial_fit(self, outputs_batch, labels_batch):
        super().partial_fit(outputs_batch=outputs_batch,
                            labels_batch=labels_batch)
        labels_pred = self.predict(outputs_batch)
        self.predicted_labels_cached.append(labels_pred.cpu())


class AccuracyEmbedding(Accuracy):
    """
    Calculates the accuracy of embedding vectors.
    The mean embedding vector is kept for each class.
    Prediction is based on the closest centroid ID.

    Parameters
    ----------
    metric : str, optional
        The metric to compute pairwise distances with.
        Default: 'cosine'
    cache : bool, optional
        Cache predicted data or not.
        Default: False

    """

    def __init__(self, metric='cosine', cache=False):
        super().__init__()
        self.metric = metric
        self.cache = cache
        self.input_cached = []
        self.centroids_dict = MeanOnlineLabels()

    @property
    def centroids(self):
        """
        Returns
        -------
        torch.Tensor
            `(C, N)` mean centroids tensor, where C is the number of unique
            classes, and N is the hidden layer dimensionality.
        """
        centroids = self.centroids_dict.get_mean()
        return centroids

    @property
    def is_fit(self):
        """
        Returns
        -------
        bool
            Whether the accuracy predictor is fit with data or not.
        """
        return len(self.centroids_dict) > 0

    def reset(self):
        super().reset()
        self.centroids_dict.reset()
        self.input_cached.clear()

    def extra_repr(self):
        return f'metric={self.metric}, cache={self.cache}'

    def distances(self, outputs_test):
        """
        Returns the distances to fit centroid means.

        Parameters
        ----------
        outputs_test : (B, D) torch.Tensor
            Hidden layer activations.

        Returns
        -------
        distances : (B, C) torch.Tensor
            Distances to each class (label).
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
        super().partial_fit(outputs_batch=outputs_batch,
                            labels_batch=labels_batch)
        outputs_batch = outputs_batch.detach().cpu()
        self.centroids_dict.update(outputs_batch, labels_batch)
        if self.cache:
            self.input_cached.append(outputs_batch)

    def predict_cached(self):
        """
        Predicts the output of a model, using cached output activations.

        Returns
        -------
        torch.Tensor
            Predicted labels.
        """
        if not self.cache:
            raise ValueError("Caching is turned off")
        if len(self.input_cached) == 0:
            raise ValueError("Empty cached input buffer")
        input = torch.cat(self.input_cached,  dim=0)
        # the output device type is CPU
        return self.predict(input)

    def predict(self, outputs_test):
        argmin = self.distances(outputs_test).argmin(dim=1).cpu()
        labels_stored = self.centroids_dict.labels()
        labels_stored = torch.IntTensor(labels_stored)
        labels_predicted = labels_stored[argmin]
        return labels_predicted.to(device=outputs_test.device)

    def predict_proba(self, outputs_test):
        distances = self.distances(outputs_test)
        proba = 1 - distances / distances.sum(dim=1).unsqueeze(1)
        return proba.to(device=outputs_test.device)
