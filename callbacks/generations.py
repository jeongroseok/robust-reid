import torch
from models import Disentangler, Generator
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.utils import make_grid


class __SampleRequiredCallback(Callback):
    def _ensure_exists_samples(self, trainer: Trainer, model: LightningModule):
        class _Samples:
            anchors: torch.Tensor = None
            positives: torch.Tensor = None
            negatives: torch.Tensor = None
            num_samples: int = 0

        (a, p, n), _ = next(iter(trainer.train_dataloader))

        self._samples = _Samples()
        self._samples.anchors = a.to(model.device)
        self._samples.positives = p.to(model.device)
        self._samples.negatives = n.to(model.device)
        self._samples.num_samples = a.shape[0]


class ReconstructionVisualizer(__SampleRequiredCallback):
    def __init__(
        self,
        normalize: bool = True,
    ):
        super().__init__()
        self.normalize = normalize

    def on_epoch_end(self, trainer: Trainer, model: LightningModule) -> None:
        if trainer.train_dataloader is None:
            return

        self._ensure_exists_samples(trainer, model)

        writer: SummaryWriter = trainer.logger.experiment

        with torch.no_grad():
            model.eval()
            encoder: Disentangler = model
            generator: Generator = model

            anc_rel, anc_unrel = encoder.disentangle(self._samples.anchors)
            reconstructed = generator.generate(anc_rel, anc_unrel)

        grid = make_grid(
            torch.cat(
                [
                    self._samples.anchors,
                    reconstructed,
                ],
                dim=2,
            ).sigmoid(),
            nrow=self._samples.num_samples
        )
        writer.add_image(
            "reconstruction", grid, global_step=trainer.global_step
        )


class CodeSwappingVisualizer(__SampleRequiredCallback):
    def __init__(
        self,
        normalize: bool = True,
    ):
        super().__init__()
        self.normalize = normalize

    def on_epoch_end(self, trainer: Trainer, model: LightningModule) -> None:
        if trainer.train_dataloader is None:
            return

        self._ensure_exists_samples(trainer, model)

        writer: SummaryWriter = trainer.logger.experiment

        with torch.no_grad():
            model.eval()
            encoder: Disentangler = model
            generator: Generator = model

            anc_rel, anc_unrel = encoder.disentangle(self._samples.anchors)
            pos_rel, pos_unrel = encoder.disentangle(self._samples.positives)
            neg_rel, neg_unrel = encoder.disentangle(self._samples.negatives)

            # self identity generation
            generated_by_self_id = generator.generate(pos_rel, anc_unrel)

            # cross identity generation
            generated_by_cross_id = generator.generate(neg_rel, anc_unrel)

        grid_self = make_grid(
            torch.cat(
                [
                    self._samples.anchors,
                    self._samples.positives,
                    generated_by_self_id,
                ],
                dim=2,
            ).sigmoid(),
            nrow=self._samples.num_samples
        )
        grid_cross = make_grid(
            torch.cat(
                [
                    self._samples.anchors,
                    self._samples.negatives,
                    generated_by_cross_id,
                ],
                dim=2,
            ).sigmoid(),
            nrow=self._samples.num_samples
        )

        writer.add_image(
            "cross_identity_generation",
            grid_cross,
            global_step=trainer.global_step,
        )
        writer.add_image(
            "self_identity_generation",
            grid_self,
            global_step=trainer.global_step,
        )


class CodeInterpolator(__SampleRequiredCallback):
    def __init__(self, steps: int = 11, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
        self.steps = steps

    def on_epoch_end(self, trainer: Trainer, model: LightningModule) -> None:
        if trainer.train_dataloader is None:
            return

        self._ensure_exists_samples(trainer, model)

        writer: SummaryWriter = trainer.logger.experiment

        with torch.no_grad():
            model.eval()
            encoder: Disentangler = model
            generator: Generator = model

            anc_rel, anc_unrel = encoder.disentangle(self._samples.anchors)
            pos_rel, pos_unrel = encoder.disentangle(self._samples.positives)
            neg_rel, neg_unrel = encoder.disentangle(self._samples.negatives)

            # related code interpolation
            imgs = []
            for weight in torch.linspace(0, 1, self.steps):
                interpolated_rel = torch.lerp(anc_rel, neg_rel, weight)
                generated = generator.generate(interpolated_rel, anc_unrel)
                imgs.append(generated)
            grid = make_grid(imgs, normalize=self.normalize)

            # unrelated code interpolation
            imgs = []
            for weight in torch.linspace(0, 1, self.steps):
                interpolated_unrel = torch.lerp(anc_unrel, neg_unrel, weight)
                generated = generator.generate(anc_rel, interpolated_unrel)
                imgs.append(generated)
            grid = make_grid(imgs, normalize=self.normalize)
