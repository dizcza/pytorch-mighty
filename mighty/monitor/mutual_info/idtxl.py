import warnings
from typing import List

import numpy as np
import sklearn
import torch
import torch.utils.data
from idtxl.estimators_jidt import JidtKraskovMI
from idtxl.estimators_opencl import OpenCLKraskovMI

from mighty.monitor.mutual_info._pca_preprocess import MutualInfoPCA
from mighty.utils.data import DataLoader


class MutualInfoIDTxl(MutualInfoPCA):

    def __init__(self, data_loader: DataLoader, pca_size=50, debug=False):
        super().__init__(data_loader=data_loader, pca_size=pca_size, debug=debug)
        settings = {'kraskov_k': 4}
        try:
            self.estimator = OpenCLKraskovMI(settings=settings)
        except RuntimeError:
            warnings.warn("No OpenCL backed detected. Run "
                          "'conda install -c conda-forge pyopencl' "
                          "in a terminal.")
            self.estimator = JidtKraskovMI(settings=settings)

    def prepare_input_finished(self):
        for key in ['input', 'target']:
            self.quantized[key] = self.quantized[key].numpy().astype(np.float64)

    def process_activations(self, layer_name: str, activations: List[torch.FloatTensor]):
        pass

    def save_mutual_info(self):
        hidden_layers_name = set(self.activations.keys())
        hidden_layers_name.difference_update({'input', 'target'})
        for layer_name in hidden_layers_name:
            activations = torch.cat(self.activations[layer_name]).numpy()
            if self.pca_size is not None and activations.shape[-1] > self.pca_size:
                pca = sklearn.decomposition.PCA(n_components=self.pca_size)
                activations = pca.fit_transform(activations)
            activations = (activations - activations.mean()) / activations.std()
            activations = activations.astype(np.float64)
            info_x = self.estimator.estimate(self.quantized['input'], activations)
            info_y = self.estimator.estimate(activations, self.quantized['target'])
            self.information[layer_name] = (
                self.to_bits(float(info_x)),
                self.to_bits(float(info_y))
            )
