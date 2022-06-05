from pytorch_lightning import Trainer, seed_everything
from callbacks.generations import (
    CodeSwappingVisualizer,
    ReconstructionVisualizer,
)

from datamodules.market1501 import PairedMarket1501DataModule
from models import RobustReID


def main():
    seed_everything(42)

    datamodule = PairedMarket1501DataModule(
        "./data", batch_size=24, num_workers=2
    )
    model = RobustReID(
        related_latent_dim=2048,
        unrelated_latent_dim=512,
        pretraining_epochs=10,
        finetuning_backbone=True,
    )

    callbacks = [
        ReconstructionVisualizer(),
        CodeSwappingVisualizer(),
    ]

    trainer = Trainer(gpus=-1, max_epochs=100, callbacks=callbacks)
    trainer.fit(model, datamodule=datamodule)
    pass


if __name__ == "__main__":
    main()
