import pickle
from abc import ABC

import numpy as np
import sklearn.decomposition
import torch
import torch.utils.data

from mighty.monitor.mutual_info.mutual_info import MutualInfo
from mighty.utils.constants import BATCH_SIZE, DATA_DIR
from mighty.utils.data import DataLoader

PCA_DIR = DATA_DIR / "pca"


class MutualInfoPCA(MutualInfo, ABC):
    """
    A base class for Mutual Information (MI) estimation followed by PCA
    dimensionality reduction.

    Parameters
    ----------
    data_loader : DataLoader
        The data loader.
    pca_size : int, optional
        PCA dimension size.
        Default: 64
    debug : bool, optional
        If True, shows more informative plots.
        Default: False

    Attributes
    ----------
    ignore_layers : tuple
        A tuple to ignore layer classes to monitor for MI.
    """

    def __init__(self, data_loader: DataLoader, pca_size=64, debug=False):
        super().__init__(data_loader=data_loader, debug=debug)
        self.pca_size = pca_size

    def _prepare_input_raw(self):
        inputs = []
        targets = []
        for images, labels in self.data_loader.eval(
                description="MutualInfo: storing raw input data"):
            inputs.append(images.flatten(start_dim=1))
            targets.append(labels)
        self.quantized['input'] = torch.cat(inputs, dim=0)
        self.quantized['target'] = torch.cat(targets, dim=0)

    def extra_repr(self):
        return f"pca_size={self.pca_size}"

    def _prepare_input(self, verbosity=1):
        if self.pca_size is None:
            self._prepare_input_raw()
            return
        if self.data_loader.batch_size < self.pca_size:
            # Batch size has to be larger than the PCA dim in order to run
            # partial fit
            pca = self.pca_full()
        else:
            pca = self.pca_incremental(verbosity)

        inputs = []
        targets = []
        description = "MutualInfo: Applying PCA to input data. Stage 2" \
            if verbosity >= 1 else None
        for images, labels in self.data_loader.eval(description):
            images = images.flatten(start_dim=1)
            images_transformed = pca.transform(images.cpu())
            images_transformed = torch.from_numpy(images_transformed).float()
            inputs.append(images_transformed)
            targets.append(labels)
        self.quantized['target'] = torch.cat(targets, dim=0)

        self.quantized['input'] = torch.cat(inputs, dim=0)

    def pca_full(self):
        """
        Perform PCA transformation on all data at once.

        Returns
        -------
        pca: sklearn.decomposition.PCA
            Trained PCA model.
        """
        dataset_name = self.data_loader.dataset_cls.__name__
        pca_path = PCA_DIR.joinpath(dataset_name, f"dim-{self.pca_size}.pkl")
        if not pca_path.exists():
            pca_path.parent.mkdir(parents=True, exist_ok=True)
            pca = sklearn.decomposition.PCA(n_components=self.pca_size,
                                            copy=False)
            images = np.vstack([im_batch.flatten(start_dim=1)
                                for im_batch, _ in self.data_loader.eval()])
            pca.fit(images)
            with open(pca_path, 'wb') as f:
                pickle.dump(pca, f)
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
        return pca

    def pca_incremental(self, verbosity=1):
        """
        Memory efficient Incremental PCA performs the transformation batch-wise

        Returns
        -------
        pca: sklearn.decomposition.IncrementalPCA
            Trained PCA model.
        """
        pca = sklearn.decomposition.IncrementalPCA(n_components=self.pca_size,
                                                   copy=False,
                                                   batch_size=BATCH_SIZE)
        description = "MutualInfo: Applying PCA to input data. Stage 1" \
            if verbosity >= 1 else None
        for images, _ in self.data_loader.eval(description):
            if images.shape[0] < self.pca_size:
                # drop the last batch if it's smaller
                continue
            images = images.flatten(start_dim=1)
            pca.partial_fit(images.cpu())
        return pca
