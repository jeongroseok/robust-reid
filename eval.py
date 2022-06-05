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

MODEL_PATH = rf"lightning_logs\version_22\checkpoints\epoch=23-step=22464 copy.ckpt"
DATA_DIR = "./data"
BATCH_SIZE = 768
NUM_WORKER = 4
DEVICE = "cpu"


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
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )
    return DataLoader(
        Market1501(DATA_DIR, mode=mode, transform=transform),
        BATCH_SIZE,
        num_workers=NUM_WORKER,
    )


@torch.no_grad()
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


def main():
    encoder: RobustReID = (
        RobustReID.load_from_checkpoint(MODEL_PATH).eval().to(DEVICE)
    )

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
            results.append(
                retrieval_average_precision(-pred, target == gallery_targets)
            )

    print(f"Total mAP: {torch.stack(results).mean()}")

    pass


if __name__ == "__main__":
    main()
