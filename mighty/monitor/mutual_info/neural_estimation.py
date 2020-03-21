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
from mighty.utils.algebra import onehot, exponential_moving_average
from mighty.utils.constants import BATCH_SIZE


class MINE_Net(nn.Module):

    def __init__(self, x_size: int, y_size: int, hidden_units=(100, 50)):
        """
        :param x_size: hidden layer shape
        :param y_size: input/target data shape
        :param hidden_units: number of hidden units in MINE net
        """
        super().__init__()
        if isinstance(hidden_units, int):
            hidden_units = [hidden_units]
        self.fc_x = nn.Linear(x_size, hidden_units[0], bias=False)
        self.fc_y = nn.Linear(y_size, hidden_units[0], bias=False)
        self.input_bias = nn.Parameter(torch.zeros(hidden_units[0]))
        # the output is a scalar - the mutual info
        self.mlp = MLP(*hidden_units, 1)

    def forward(self, x, y):
        """
        :param x: some hidden layer batch activations of shape (batch_size, embedding_size)
        :param y: either input or target data samples of shape (batch_size, input_dimensions or 1)
        :return: mutual information I(x, y) lower bound
        """
        mi = F.relu(self.fc_x(x) + self.fc_y(y) + self.input_bias,
                    inplace=True)
        mi = self.mlp(mi)
        return mi


class MINE_Trainer:
    log2_e = np.log2(np.e)

    """
    Parameters
    ----------
    mine_model : MutualInfoNeuralEstimationNetwork
        A network to estimate mutual information.
    learning_rate : float
        Optimizer learning rate.
    """

    def __init__(self, mine_model: nn.Module, learning_rate=1e-3,
                 smooth_filter_size=30):
        if torch.cuda.is_available():
            mine_model = mine_model.cuda()
        self.mine_model = mine_model
        self.optimizer = torch.optim.Adam(self.mine_model.parameters(),
                                          lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.5)
        self.mi_history = [0.]
        self.smooth_filter_size = smooth_filter_size

    def __repr__(self):
        return f"{MINE_Trainer.__name__}(model={self.mine_model}, " \
               f"optimizer={self.optimizer})"

    def reset(self):
        self.mi_history = [0]

    def train_batch(self, data_batch, labels_batch):
        """
        Performs a single step to refine I(X; Y).

        Parameters
        ----------
        data_batch, labels_batch : torch.Tensor
            A batch of multidimensional X and Y of size (B, N) to
            estimate mutual information from. N could be 1 or more.
        """
        if torch.cuda.is_available():
            data_batch = data_batch.cuda()
            labels_batch = labels_batch.cuda()
        self.optimizer.zero_grad()
        pred_joint = self.mine_model(data_batch, labels_batch)
        labels_batch = labels_batch[
            torch.randperm(labels_batch.shape[0], device=labels_batch.device)]
        pred_marginal = self.mine_model(data_batch, labels_batch)
        mi_lower_bound = pred_joint.mean() - pred_marginal.exp().mean().log()
        mi_bits = mi_lower_bound.item() * self.log2_e  # convert nats to bits
        self.mi_history.append(mi_bits)
        loss = -mi_lower_bound  # maximize
        loss.backward()
        self.optimizer.step()

    @property
    def mi_history_smoothed(self):
        return exponential_moving_average(self.mi_history,
                                          window=self.smooth_filter_size)

    def get_mutual_info(self):
        """
        Returns
        -------
        float
            Estimated mutual information lower bound.
        """
        return self.mi_history_smoothed.max()


class MutualInfoNeuralEstimation(MutualInfoPCA):

    def __init__(self, estimate_size=None, pca_size=100, debug=False,
                 estimate_epochs=5, noise_std=1e-3):
        """
        :param estimate_size: number of samples to estimate mutual information from
        :param estimate_epochs: total estimation epochs to run
        :param pca_size: transform input data to this size;
                               pass None to use original raw input data (no transformation is applied)
        :param noise_std: how much noise to add to input and targets
        :param debug: plot MINE training curves?
        """
        super().__init__(estimate_size=estimate_size, pca_size=pca_size,
                         debug=debug)
        self.estimate_epochs = estimate_epochs
        self.noise_sampler = torch.distributions.normal.Normal(loc=0,
                                                               scale=noise_std)
        self.trainers = {}  # MutualInformationNeuralEstimation trainers for both input X- and target Y-data
        self.input_size = None
        self.target_size = None

    def extra_repr(self):
        return super().extra_repr() + f"; noise_variance={self.noise_sampler.variance}; " \
                                      f"MINETrainer(filter_size={MINE_Trainer.filter_size}, " \
                                      f"filter_rounds={MINE_Trainer.filter_rounds}, " \
                                      f"optimizer.lr={MINE_Trainer.learning_rate}); " \
                                      f"MINE(hidden_units={MINE_Net.hidden_units})"

    def prepare_input_finished(self):
        self.input_size = self.quantized['input'].shape[1]
        self.target_size = len(self.quantized['target'].unique())
        # one-hot encoded labels are better fit than argmax
        self.quantized['target'] = onehot(self.quantized['target']).type(
            torch.float32)

    def process_activations(self, layer_name: str,
                            activations: List[torch.FloatTensor]):
        # TODO process each batch in save_activations()
        activations = torch.cat(activations, dim=0)
        assert len(self.quantized['input']) == len(
            self.quantized['target']) == len(activations)
        embedding_size = activations.shape[1]
        if layer_name not in self.trainers:
            self.trainers[layer_name] = (
                MINE_Trainer(MINE_Net(x_size=embedding_size,
                                      y_size=self.input_size)),
                MINE_Trainer(MINE_Net(x_size=embedding_size,
                                      y_size=self.target_size)),
            )
        for mi_trainer in self.trainers[layer_name]:
            mi_trainer.reset()
        for epoch in range(self.estimate_epochs):
            for mi_trainer in self.trainers[layer_name]:
                mi_trainer.scheduler.step(epoch=epoch)
            permutations = torch.randperm(len(activations)).split(BATCH_SIZE)
            for batch_permutation in permutations:
                activations_batch = activations[batch_permutation]
                for data_type, trainer in zip(('input', 'target'),
                                              self.trainers[layer_name]):
                    labels_batch = self.quantized[data_type][batch_permutation]
                    labels_batch = labels_batch + self.noise_sampler.sample(
                        labels_batch.shape)
                    trainer.train_batch(data_batch=activations_batch,
                                        labels_batch=labels_batch)

    def save_mutual_info(self):
        for layer_name, (trainer_x, trainer_y) in self.trainers.items():
            info_x = trainer_x.get_mutual_info()
            info_y = trainer_y.get_mutual_info()
            self.information[layer_name] = (info_x, info_y)

    def plot_mine_history_loss(self, viz):
        legend = []
        info_x = []
        info_y = []
        for layer_name, (trainer_x, trainer_y) in self.trainers.items():
            info_x.append(trainer_x.mi_history_smoothed)
            info_y.append(trainer_y.mi_history_smoothed)
            legend.append(layer_name)
        for info_name, info in (('input X', info_x), ('target Y', info_y)):
            info = np.transpose(info)
            title = f'MutualInfoNeuralEstimation {info_name}'
            viz.line(Y=info, X=np.arange(len(info)), win=title, opts=dict(
                xlabel='Iteration',
                ylabel='Mutual info lower bound, bits',
                title=title,
                legend=legend,
            ))

    def _plot_debug(self, viz):
        super()._plot_debug(viz)
        self.plot_mine_history_loss(viz)
