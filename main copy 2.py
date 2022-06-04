from pytorch_lightning import Trainer, seed_everything

from datamodules.market1501 import PairedMarket1501DataModule
from models import RobustReID


def main():
    seed_everything(42)

    datamodule = PairedMarket1501DataModule("./data", batch_size=16, num_workers=2)
    model = RobustReID()

    trainer = Trainer(gpus=-1)
    trainer.fit(model, datamodule=datamodule)
    pass


if __name__ == "__main__":
    main()
