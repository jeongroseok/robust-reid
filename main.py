import torch
from pytorch_lightning import Trainer, seed_everything
from torchvision.transforms.functional import normalize, to_pil_image
from torchvision.utils import make_grid

from datamodules.market1501 import PairedMarket1501DataModule
from datasets.market1501 import PairedMarket1501
from models import RobustReID

MODEL_PATH = rf"lightning_logs\version_4\checkpoints\epoch=7-step=6912.ckpt"


def main():
    seed_everything(42)

    datamodule = PairedMarket1501DataModule(
        "./data", batch_size=4, num_workers=0
    )
    model = RobustReID.load_from_checkpoint(MODEL_PATH)

    datamodule.setup()
    dataloader = datamodule.train_dataloader()
    (anc, pos, neg), _ = next(iter(dataloader))

    anc_related_codes, anc_unrelated_codes = model.forward(anc)
    pos_related_codes, pos_unrelated_codes = model.forward(pos)
    neg_related_codes, neg_unrelated_codes = model.forward(neg)
    reconstructed_anc = model.generate(anc_related_codes, anc_unrelated_codes)

    image = make_grid(
        torch.cat([anc, reconstructed_anc, pos], dim=2),
        nrow=datamodule.batch_size,
        normalize=True,
    )
    to_pil_image(image).show()

    pass


if __name__ == "__main__":
    main()
