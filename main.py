import torch
import torch.nn as nn
from PIL import Image
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
from torchvision import models
from torchvision.models.resnet import Bottleneck, resnet50
from torchvision.transforms import Compose, ToTensor
from pytorch_lightning import LightningModule, Trainer, seed_everything

from datamodules.market1501 import Market1501, PairedMarket1501DataModule
from models.components import (
    Classifer,
    Discriminator,
    Generator,
    RelatedEncoder,
    ResnetBackbone,
    UnrelatedEncoder,
)


def main():
    seed_everything(42)

    dm = PairedMarket1501DataModule('./data')
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    pass


if __name__ == "__main__":
    main()
