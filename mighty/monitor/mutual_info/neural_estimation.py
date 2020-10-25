"""
MINE: Mutual Information Neural Estimation
M. Belghazi et. al, 2018
https://arxiv.org/abs/1801.04062
"""

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from mighty.models import MLP
from mighty.monitor.mutual_info._pca_preprocess import MutualInfoPCA
from mighty.utils.algebra import to_onehot, exponential_moving_average
from mighty.utils.constants import BATCH_SIZE
from mighty.utils.data import DataLoader


class MINE_Net(nn.Module):

    def __init__(self, x_size: int, y_size: int, hidden_units=(100, 50)):
        """
        A network to estimate the mutual information between X and Y, I(X; Y).

        Parameters
        ----------
        x_size, y_size : int
            Number of neurons in X and Y.
        hidden_units : int or tuple of int
            Hidden layer size(s).
        """
        super().__init__()
        if isinstance(hidden_units, int):
            hidden_units = [hidden_units]
        self.fc_x = nn.Linear(x_size, hidden_units[0], bias=False)
        self.fc_y = nn.Linear(y_size, hidden_units[0], bias=False)
        self.xy_bias = nn.Parameter(torch.zeros(hidden_units[0]))
        # the output mutual info is a scalar; hence, the last dimension is 1
        self.fc_output = MLP(*hidden_units, 1)

    def forward(self, x, y):
        """
        Parameters
        ----------
        x, y : torch.Tensor
            Data batches.

        Returns
        -------
        mi : torch.Tensor
            Kullback-Leibler lower-bound estimation of I(X; Y).
        """
        hidden = F.relu(self.fc_x(x) + self.fc_y(y) + self.xy_bias,
                        inplace=True)
        mi = self.fc_output(hidden)
        return mi


class MINE_Trainer:
    """
    Parameters
    ----------
    mine_model : MINE_Net
        A network to estimate mutual information.
    learning_rate : float
        Optimizer learning rate.
    smooth_filter_size : int
        Smoothing filter size. The larger the filter, the smoother but also
        more biased towards lower values of the resulting estimate.
    """

    log2_e = np.log2(np.e)

    def __init__(self, mine_model: nn.Module, learning_rate=1e-3,
                 smooth_filter_size=30):
        if torch.cuda.is_available():
            mine_model = mine_model.cuda()
        self.mine_model = mine_model
        self.optimizer = torch.optim.Adam(self.mine_model.parameters(),
                                          lr=learning_rate,
                                          weight_decay=1e-5)
        self.smooth_filter_size = smooth_filter_size

        self.scheduler = None
        self.mi_history = None
        self.reset()

    def __repr__(self):
        return f"{MINE_Trainer.__name__}(model={self.mine_model}, " \
               f"optimizer={self.optimizer}, " \
               f"smooth_filter_size={self.smooth_filter_size})"

    def reset(self):
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.5)
        self.mi_history = [0]

    def train_batch(self, x_batch, y_batch):
        """
        Performs a single step to refine I(X; Y).

        Parameters
        ----------
        x_batch, y_batch : torch.Tensor
            A batch of multidimensional X and Y of size (B, N) to
            estimate mutual information from. N could be 1 or more.
        """
        if torch.cuda.is_available():
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
        self.optimizer.zero_grad()
        pred_joint = self.mine_model(x_batch, y_batch)
        y_batch = y_batch[
            torch.randperm(y_batch.shape[0], device=y_batch.device)]
        pred_marginal = self.mine_model(x_batch, y_batch)
        mi_lower_bound = pred_joint.mean() - pred_marginal.exp().mean().log()
        mi_bits = mi_lower_bound.item() * self.log2_e  # convert nats to bits
        self.mi_history.append(mi_bits)
        loss = -mi_lower_bound  # maximize
        loss.backward()
        self.optimizer.step()

    def smooth_history(self):
        history = torch.as_tensor(self.mi_history)
        history = history[~torch.isnan(history)]
        return exponential_moving_average(history,
                                          window=self.smooth_filter_size)

    def get_mutual_info(self):
        """
        Returns
        -------
        float
            Estimated mutual information lower bound.
        """
        return self.smooth_history().max()


class MutualInfoNeuralEstimation(MutualInfoPCA):
    """
    Mutual Information Neural Estimation [1]_, followed by PCA dimensionality
    reduction.

    Parameters
    ----------
    data_loader : DataLoader
        The data loader.
    pca_size : int, optional
        PCA dimension size.
        Default: 100
    estimate_epochs : int, optional
        The number of epochs to run.
        Default: 5
    noise_std : float, optional
        Additive noise standard deviation (to break the degeneracy).
        Default: 1e-3
    debug : bool, optional
        If True, shows more informative plots.
        Default: False

    Attributes
    ----------
    ignore_layers : tuple
        A tuple to ignore layer classes to monitor for MI.

    References
    ----------
    1. Belghazi, M. I., Baratin, A., Rajeswar, S., Ozair, S., Bengio, Y.,
       Courville, A., & Hjelm, R. D. (2018). Mine: mutual information neural
       estimation. arXiv preprint arXiv:1801.04062.
    """

    def __init__(self, data_loader: DataLoader, pca_size=100, debug=False,
                 hidden_units=(100, 50), estimate_epochs=5, noise_std=1e-3):
        super().__init__(data_loader=data_loader, pca_size=pca_size,
                         debug=debug)
        self.estimate_epochs = estimate_epochs
        self.hidden_units = hidden_units
        self.noise_sampler = torch.distributions.normal.Normal(loc=0,
                                                               scale=noise_std)
        self.trainers = {}  # MutualInformationNeuralEstimation trainers for both input X- and target Y-data
        self.input_size = None
        self.target_size = None

    def extra_repr(self):
        return f"{super().extra_repr()}; noise_variance={self.noise_sampler.variance}; "

    def _prepare_input_finished(self):
        super()._prepare_input_finished()
        self.input_size = self.quantized['input'].shape[1]
        self.target_size = len(self.quantized['target'].unique())
        # one-hot encoded labels are better fit than argmax
        self.quantized['target'] = to_onehot(self.quantized['target']).type(
            torch.float32)

    def _process_activations(self, layer_name: str,
                             activations: List[torch.FloatTensor]):
        # TODO process each batch in save_activations()
        activations = torch.cat(activations, dim=0)
        assert len(self.quantized['input']) == len(
            self.quantized['target']) == len(activations)
        embedding_size = activations.shape[1]
        if layer_name not in self.trainers:
            self.trainers[layer_name] = (
                MINE_Trainer(MINE_Net(x_size=embedding_size,
                                      y_size=self.input_size,
                                      hidden_units=self.hidden_units)),
                MINE_Trainer(MINE_Net(x_size=embedding_size,
                                      y_size=self.target_size,
                                      hidden_units=self.hidden_units)),
            )
        for mi_trainer in self.trainers[layer_name]:
            mi_trainer.reset()
        for epoch in range(self.estimate_epochs):
            permutations = torch.randperm(len(activations)).split(BATCH_SIZE)
            for batch_permutation in permutations:
                activations_batch = activations[batch_permutation]
                for data_type, trainer in zip(('input', 'target'),
                                              self.trainers[layer_name]):
                    labels_batch = self.quantized[data_type][batch_permutation]
                    labels_batch = labels_batch + self.noise_sampler.sample(
                        labels_batch.shape)
                    trainer.train_batch(x_batch=activations_batch,
                                        y_batch=labels_batch)
            for mi_trainer in self.trainers[layer_name]:
                mi_trainer.scheduler.step()

    def _save_mutual_info(self):
        for layer_name, (trainer_x, trainer_y) in self.trainers.items():
            info_x = trainer_x.get_mutual_info()
            info_y = trainer_y.get_mutual_info()
            self.information[layer_name] = (info_x, info_y)

    def plot_mine_history_loss(self, viz):
        """
        Plots the loss of a training progress with iterations.
        """
        legend = []
        info_x = []
        info_y = []
        for layer_name, (trainer_x, trainer_y) in self.trainers.items():
            info_x.append(trainer_x.smooth_history())
            info_y.append(trainer_y.smooth_history())
            legend.append(layer_name)
        for info_name, info in (('input X', info_x), ('target Y', info_y)):
            info = torch.stack(info).t().squeeze()
            title = f'MutualInfoNeuralEstimation {info_name}'
            viz.line(Y=info, X=torch.arange(len(info)), win=title, opts=dict(
                xlabel='Iteration',
                ylabel='Mutual info lower bound, bits',
                title=title,
                legend=legend,
            ))

    def _plot_debug(self, viz):
        super()._plot_debug(viz)
        self.plot_mine_history_loss(viz)
