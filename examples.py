import torch
import torch.nn as nn
import torchvision.models
from PIL import Image
from torchvision.datasets import MNIST

from mighty.models import *
from mighty.monitor.accuracy import AccuracyArgmax, AccuracyAutoencoder
from mighty.monitor.monitor import Monitor
from mighty.monitor.mutual_info import *
from mighty.trainer import *
from mighty.utils.common import set_seed
from mighty.utils.data import get_normalize_inverse, DataLoader, \
    TransformDefault
from mighty.utils.domain import MonitorLevel


def get_optimizer_scheduler(model: nn.Module):
    optimizer = torch.optim.Adam(
        filter(lambda param: param.requires_grad, model.parameters()), lr=1e-3,
        weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.5,
                                                           patience=15,
                                                           threshold=1e-3,
                                                           min_lr=1e-4)
    return optimizer, scheduler


def train_mask():
    """
    Train explainable mask for an image from ImageNet, using pretrained model.
    """
    model = torchvision.models.vgg19(pretrained=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    transform = TransformDefault.imagenet()
    accuracy_measure = AccuracyArgmax()
    monitor = Monitor(
        accuracy_measure=accuracy_measure,
        mutual_info=MutualInfoKMeans(),
        normalize_inverse=get_normalize_inverse(transform),
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
        model.cuda()
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
    data_loader = DataLoader(dataset_cls, TransformDefault.mnist())
    trainer = TrainerGrad(model,
                          criterion=nn.CrossEntropyLoss(),
                          data_loader=data_loader,
                          optimizer=optimizer,
                          scheduler=scheduler)
    # trainer.restore()  # uncomment to restore the saved state
    trainer.monitor.advanced_monitoring(level=MonitorLevel.FULL)
    trainer.train(n_epochs=n_epoch, mutual_info_layers=0)


def test(model, n_epoch=500, dataset_cls=MNIST):
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    criterion = nn.CrossEntropyLoss()
    data_loader = DataLoader(dataset_cls, TransformDefault.mnist())
    trainer = Test(model=model, criterion=criterion, data_loader=data_loader)
    trainer.train(n_epochs=n_epoch, adversarial=True, mask_explain=True)


def train_autoencoder(n_epoch=60, dataset_cls=MNIST):
    model = AutoencoderLinear(784, 128)
    data_loader = DataLoader(dataset_cls,
                             transform=TransformDefault.mnist(
                                 normalize=None
                             ))
    criterion = nn.BCEWithLogitsLoss()
    optimizer, scheduler = get_optimizer_scheduler(model)
    trainer = TrainerAutoencoder(model,
                                 criterion=criterion,
                                 data_loader=data_loader,
                                 optimizer=optimizer,
                                 scheduler=scheduler,
                                 accuracy_measure=AccuracyAutoencoder(
                                     cache=True))
    # trainer.restore()  # uncomment to restore the saved state
    trainer.monitor.advanced_monitoring(level=MonitorLevel.SIGNAL_TO_NOISE)
    trainer.train(n_epochs=n_epoch, mutual_info_layers=0)


if __name__ == '__main__':
    set_seed(26)
    # torch.backends.cudnn.benchmark = True
    train_autoencoder()
    # train_grad()
