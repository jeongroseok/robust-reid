from contextlib import contextmanager
from time import time

import torch
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import DataLoader
from torchmetrics.functional import (
    pairwise_euclidean_distance,
    retrieval_average_precision,
)
from torchvision import transforms
from torchvision.transforms.functional import normalize, to_pil_image
from torchvision.utils import make_grid

from datasets.market1501 import Market1501
from models import RobustReID
from models.components import Encoder

MODEL_PATH = (
    rf"lightning_logs\freezed-9epoch\checkpoints\epoch=9-step=12940.ckpt"
)
DATA_DIR = "./data"
BATCH_SIZE = 1024
NUM_WORKER = 4
DEVICE = "cuda"


@contextmanager
def timer(prefix: str):
    start_time = time()
    yield
    print(f"{prefix} took {time() - start_time:.3f} [s]")


def get_dataloader(mode: str):
    transform = transforms.Compose(
        [
            transforms.Resize((256, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )
    return DataLoader(
        Market1501(DATA_DIR, mode=mode, transform=transform),
        BATCH_SIZE,
        num_workers=NUM_WORKER,
    )


def process_data(encoder: Encoder, dataloader: DataLoader):
    codes = []
    targets = []

    for x, y in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        code = encoder.encode(x)

        codes.extend(code.tolist())
        targets.extend(y.tolist())
        
    return (
        torch.tensor(codes, device=DEVICE),
        torch.tensor(targets, device=DEVICE),
    )


@torch.no_grad()
def main():
    encoder: RobustReID = (
        RobustReID.load_from_checkpoint(MODEL_PATH).eval().to(DEVICE)
    )

    encoder = RobustReID().eval().to(DEVICE)

    with timer("query extreaction"):
        query_dataloader = get_dataloader("query")
        query_codes, query_targets = process_data(encoder, query_dataloader)

    with timer("gallery extreaction"):
        gallery_dataloader = get_dataloader("gallery")
        gallery_codes, gallery_targets = process_data(
            encoder, gallery_dataloader
        )

    with timer("distance calculation"):
        preds = pairwise_euclidean_distance(query_codes, gallery_codes)

    with timer("mAP calculation"):
        results = []
        for data in zip(preds, query_targets):
            pred, target = data
            results.append(retrieval_average_precision(-pred, target == gallery_targets))

    print(f"Total mAP: {torch.stack(results).mean()}")

    pass


if __name__ == "__main__":
    main()
