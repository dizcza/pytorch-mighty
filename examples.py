import os

os.environ['FULL_FORWARD_PASS_SIZE'] = '10000'

import torch
import torch.nn as nn
import torchvision.models
from PIL import Image
from torchvision import transforms
from torchvision.datasets import MNIST

from mighty.models import MLP
from mighty.monitor.accuracy import AccuracyArgmax
from mighty.monitor.monitor import Monitor
from mighty.monitor.mutual_info import *
from mighty.trainer import TrainerGrad, MaskTrainer, Test
from mighty.utils.common import set_seed
from mighty.utils.data import NormalizeInverse, DataLoader
from mighty.utils.domain import MonitorLevel


def get_optimizer_scheduler(model: nn.Module):
    optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=1e-3,
                                 weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15,
                                                           threshold=1e-3, min_lr=1e-4)
    return optimizer, scheduler


def train_mask():
    """
    Train explainable mask for an image from ImageNet, using pretrained model.
    """
    model = torchvision.models.vgg19(pretrained=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    normalize = transforms.Normalize(
        # ImageNet normalize
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                    transforms.ToTensor(), normalize])
    accuracy_measure = AccuracyArgmax()
    monitor = Monitor(
        accuracy_measure=accuracy_measure,
        mutual_info=MutualInfoKMeans(),
        normalize_inverse=NormalizeInverse(mean=normalize.mean,
                                           std=normalize.std),
    )
    monitor.open(env_name='mask')
    image = Image.open("images/flute.jpg")
    image = transform(image)
    mask_trainer = MaskTrainer(
        accuracy_measure=accuracy_measure,
        image_shape=image.shape,
        show_progress=True
    )
    monitor.log(repr(mask_trainer))
    if torch.cuda.is_available():
        model = model.cuda()
        image = image.cuda()
    outputs = model(image.unsqueeze(dim=0))
    proba = accuracy_measure.predict_proba(outputs)
    proba_max, label_true = proba[0].max(dim=0)
    print(f"True label: {label_true} (confidence {proba_max: .5f})")
    monitor.plot_mask(model=model, mask_trainer=mask_trainer, image=image,
                      label=label_true)


def train_grad(n_epoch=10, dataset_cls=MNIST):
    model = MLP(784, 128, 10)
    optimizer, scheduler = get_optimizer_scheduler(model)
    normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    data_loader = DataLoader(dataset_cls, normalize=normalize)
    trainer = TrainerGrad(model,
                          criterion=nn.CrossEntropyLoss(),
                          data_loader=data_loader,
                          optimizer=optimizer,
                          scheduler=scheduler)
    # trainer.restore()  # uncomment to restore the saved state
    trainer.monitor.advanced_monitoring(level=MonitorLevel.FULL)
    trainer.train(n_epoch=n_epoch, mutual_info_layers=0, cache=True)


def test(model, n_epoch=500, dataset_cls=MNIST):
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    criterion = nn.CrossEntropyLoss()
    normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    data_loader = DataLoader(dataset_cls, normalize=normalize)
    trainer = Test(model=model, criterion=criterion, data_loader=data_loader)
    trainer.train(n_epoch=n_epoch, adversarial=True, mask_explain=True)


if __name__ == '__main__':
    set_seed(26)
    # torch.backends.cudnn.benchmark = True
    train_grad()
    # train_mask()
    # test()
