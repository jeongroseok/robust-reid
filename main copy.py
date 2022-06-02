import torch
import torch.nn as nn
from PIL import Image
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
from torchvision import models
from torchvision.transforms import Compose, ToTensor
from torchvision.models.resnet import resnet50, Bottleneck

from models.components import RelatedEncoder, ResnetBackbone, UnrelatedEncoder, Generator, Discriminator, Classifer


def main():
    backbone = ResnetBackbone(pretrained=True)
    Er = RelatedEncoder(backbone.out_features)
    Eu = UnrelatedEncoder(backbone.out_features)
    C = Classifer(Er.latent_dim)
    G = Generator(Er.latent_dim, Eu.latent_dim)
    D = Discriminator()

    x = torch.randn((16, 3, 256, 128)) # resized input
    features = backbone(x)
    z_rel = Er(features[0])
    z_unrel, _, _ = Eu(features[1])
    y_hat = C(z_rel)
    x_hat = G(z_rel, z_unrel)
    d, y_hat = D(x_hat)
    pass


if __name__ == "__main__":
    main()
