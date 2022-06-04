from ctypes import Union
from typing import List

import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from torch import Tensor

from torchvision.utils import make_grid

from models import Encoder, Generator

def __(images):
    pass

class CodeInterpolator(Callback):
    def __init__(
        self,
        range_start: int = -5,
        range_end: int = 5,
        steps: int = 11,
        normalize: bool = True,
    ):
        """
        Args:
            range_start: default -5
            range_end: default 5
            steps: number of step between start and end
            normalize: default True (change image to (0, 1) range)
        """
        super().__init__()
        self.range_start = range_start
        self.range_end = range_end
        self.num_samples = num_samples
        self.normalize = normalize
        self.steps = steps
        """
        1. pick images
        2. interpolate
        3. make image
        4. logging
        """

    def on_fit_start(self, trainer: Trainer, pl_module):
        (a, p, n), _ = next(iter(trainer.train_dataloader))
        self.__images = {"anchor": a, "positive": p, "negative": n}

    def on_epoch_end(
        self,
        trainer: Trainer,
        pl_module: Union[LightningModule, Generator, Encoder],
    ) -> None:
        if trainer.train_dataloader is None:
            return

        images = self.__interpolate(
            pl_module, latent_dim=pl_module.hparams.latent_dim
        )
        images = torch.cat(images, dim=0)

        num_rows = self.steps
        grid = make_grid(images, nrow=num_rows, normalize=self.normalize)
        str_title = f"{pl_module.__class__.__name__}_latent_space"
        trainer.logger.experiment.add_image(
            str_title, grid, global_step=trainer.global_step
        )

    def __interpolate(
        self, pl_module: LightningModule, latent_dim: int
    ) -> List[Tensor]:
        images = []
        with torch.no_grad():
            pl_module.eval()
            for z1 in np.linspace(
                self.range_start, self.range_end, self.steps
            ):
                for z2 in np.linspace(
                    self.range_start, self.range_end, self.steps
                ):
                    # set all dims to zero
                    z = torch.zeros(
                        self.num_samples, latent_dim, device=pl_module.device
                    )

                    # set the fist 2 dims to the value
                    z[:, 0] = torch.tensor(z1)
                    z[:, 1] = torch.tensor(z2)

                    # sample
                    # generate images
                    img = pl_module(z)

                    if len(img.size()) == 2:
                        img = img.view(self.num_samples, *pl_module.img_dim)

                    img = img[0]
                    img = img.unsqueeze(0)
                    images.append(img)

        pl_module.train()
        return images
