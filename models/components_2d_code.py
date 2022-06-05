from copy import deepcopy
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from pl_bolts.models.gans.pix2pix.components import PatchGAN
from torch import Tensor


class Encoder:
    def encode(
        self, x: torch.Tensor
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        pass


class Disentangler:
    def disentangle(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        pass


class Generator:
    def generate(self, *features: torch.Tensor) -> torch.Tensor:
        pass


class Classifier:
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        pass


class ResnetBackbone(nn.Module):
    def __init__(
        self,
        type: str = "resnet18",
        finetuning: bool = True,
        pretrained: bool = False,
        progress: bool = True,
    ) -> None:
        super().__init__()

        self.finetuning = finetuning

        resnet = None
        if type == "resnet18":
            resnet = torchvision.models.resnet18(pretrained, progress)
        elif type == "resnet34":
            resnet = torchvision.models.resnet34(pretrained, progress)
        elif type == "resnet50":
            resnet = torchvision.models.resnet50(pretrained, progress)
        layers = list(resnet.children())[:-2]

        self.backbone = nn.Sequential(*layers[:-1])  # layer3까지만 공통으로 사용함
        self.related_branch = deepcopy(layers[-1])
        self.unrelated_branch = deepcopy(layers[-1])

        """
        # 공간정보 유지를 위해 첫번째 블럭 수정
        first_block: Union[
            torchvision.models.resnet.Bottleneck, torchvision.models.resnet.BasicBlock
        ] = self.unrelated_branch[0]

        # 블럭내의 모든 Conv2d레이어의 stride를 1로 변경
        for name, module in first_block.named_children():
            if not isinstance(module, nn.Conv2d):
                continue
            new_conv = self.__set_conv2d_attr(module, stride=1)
            setattr(first_block, name, new_conv)

        # downsample의 Conv2d레이어의 stride를 1로 변경
        first_block.downsample[0] = self.__set_conv2d_attr(
            first_block.downsample[0], stride=1
        )

        # 두번째 Conv2d레이어의 padding를 1로 변경
        if isinstance(first_block, torchvision.models.resnet.Bottleneck):
            first_block.conv2 = self.__set_conv2d_attr(
                first_block.conv2, padding=0
            )

        self.unrelated_branch.load_state_dict(layers[-1].state_dict())
        """

        for p in self.backbone.parameters():
            p.requires_grad = self.finetuning
        for p in self.related_branch[0].parameters():
            p.requires_grad = self.finetuning
        for p in self.unrelated_branch[0].parameters():
            p.requires_grad = self.finetuning

    @staticmethod
    def __set_conv2d_attr(old_conv: nn.Conv2d, stride=None, padding=None):
        return nn.Conv2d(
            in_channels=old_conv.in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=stride if stride else old_conv.stride,
            padding=padding if padding else old_conv.padding,
            bias=old_conv.bias,
        )

    def forward(self, x: Tensor):
        features = self.backbone(x)

        related_features = self.related_branch(features)
        unrelated_features = self.unrelated_branch(features)
        return related_features, unrelated_features

    @property
    def out_features(self):
        return list(self.modules())[-1].num_features


class RelatedEncoder(nn.Sequential, Encoder):
    def __init__(
        self, num_features: int = 2048, latent_dim: int = 2048
    ) -> None:
        super().__init__(
            # nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(num_features, latent_dim, 1, bias=False),
            nn.BatchNorm2d(latent_dim),
            # nn.Flatten(1),
            # nn.Linear(num_features, latent_dim, bias=False),
            # nn.BatchNorm1d(latent_dim),
        )
        self.latent_dim = latent_dim

    def encode(
        self, x: torch.Tensor
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        return super().forward(x)


class UnrelatedEncoder(nn.Module, Encoder):
    def __init__(
        self, num_features: int = 2048, latent_dim: int = 512
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim

        # self.adaptivemaxpool = nn.AdaptiveAvgPool2d(1)
        self.conv_mu = nn.Conv2d(num_features, latent_dim, 1, bias=False)
        self.conv_var = nn.Conv2d(num_features, latent_dim, 1, bias=False)

        # self.flatten = nn.Flatten(1)
        # self.linear_mu = nn.Linear(num_features, latent_dim, bias=False)
        # self.linear_var = nn.Linear(num_features, latent_dim, bias=False)

    def forward(self, x: Tensor):
        # x = self.adaptivemaxpool(x)
        # x = self.flatten(x)

        mu = self.conv_mu(x)
        log_var = self.conv_var(x)
        std = torch.exp(log_var / 2)

        p = torch.distributions.Normal(
            torch.zeros_like(mu), torch.ones_like(std)
        )
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return z, p, q

    def encode(
        self, x: torch.Tensor
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        z, _, _ = self.forward(x)
        return z


class BasicClassifer(nn.Sequential, Classifier):
    def __init__(
        self, num_classes: int = 751, num_features: int = 256
    ) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(num_features, num_classes),
        )

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        y_hat = super().forward(x)
        return y_hat


class ISGAN_Generator(nn.Sequential, Generator):
    def __init__(
        self, related_latent_dim: int = 256, unrelated_latent_dim: int = 64
    ) -> None:
        super().__init__(
            # FC
            nn.Conv2d(related_latent_dim + unrelated_latent_dim, 512, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            # nn.Unflatten(1, (-1, 1, 1)),
            # 1st block 1x1 -> 4x2
            nn.ConvTranspose2d(512, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            # 2nd block 4x2 -> 8x4
            nn.ConvTranspose2d(512, 512, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            # 3rd block 8x4 -> 16x8
            nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            # 4th block 16x8 -> 32x16
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # 5th block 16x8 -> 32x16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # 6th block 16x8 -> 64x128
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            # 7th block 16x8 -> 64x128
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, related_code: Tensor, unrelated_code: Tensor):
        z = torch.cat([related_code, unrelated_code], dim=1)
        x = super().forward(z)
        return x

    def generate(self, *features: torch.Tensor) -> torch.Tensor:
        return self.forward(*features)


class Discriminator(PatchGAN, Classifier):
    def __init__(self, num_classes: int = 751) -> None:
        super().__init__(input_channels=3)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x0 = self.d1(x)
        x1 = self.d2(x0)
        x2 = self.d3(x1)
        x3 = self.d4(x2)
        d = self.final(x3)

        x4 = F.adaptive_avg_pool2d(x3, 1)
        x4 = x4.flatten(1)
        y_hat = self.fc(x4)
        return d, y_hat

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        _, y_hat = self.forward(x)
        return y_hat
