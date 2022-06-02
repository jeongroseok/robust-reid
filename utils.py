from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as transform_lib
from torchvision.utils import make_grid
from tqdm import tqdm

from .datasets.market1501 import Market1501


def set_persistent_workers(datamodule: VisionDataModule):
    def _data_loader(self: VisionDataModule,
                     dataset: Dataset,
                     shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )

    datamodule._data_loader = _data_loader


class Evaluator():
    def __init__(
        self,
        root: str,
        encoder: any,
        output_prefix: str,
        batch_size: int = 512,
        num_workers: int = 2
    ) -> None:
        self._encoder = encoder.cuda().eval()
        self._output_prefix = output_prefix
        self._batch_size = batch_size
        self._num_workers = num_workers

        transforms = transform_lib.Compose([
            transform_lib.ToTensor(),
            transform_lib.Normalize((0.5, ), (0.5, )),
        ])

        self._query_set = Market1501(root, mode='query', transform=transforms)
        query_features = self._extract_features(self._query_set)

        self._gallery_set = Market1501(
            root, mode='gallery', transform=transforms)
        gallery_features = self._extract_features(self._gallery_set)

        self._matrix = torch.cdist(query_features, gallery_features)

    def evaluate_top_k_accuracy(self, k=10):
        print(f'top-{k} evaluating...')
        topk = self.matrix.topk(k, largest=False)[1]

        count = 0
        for query_idx in tqdm(range(len(topk))):
            query_label = self._query_set[query_idx][1]
            for gallery_idx in topk[query_idx]:
                gallery_label = self._gallery_set[gallery_idx][1]
                if query_label == gallery_label:
                    count += 1
                    break
        return count / len(topk)

    def visualize_rand(self, k=10):
        topk = self.matrix.topk(k, largest=False)[1]

        query_idx = np.random.choice(len(topk))
        query_img = self._query_set[query_idx][0]

        images = [query_img]
        for gallery_idx in topk[query_idx]:
            gallery_img = self._gallery_set[gallery_idx][0]
            images.append(gallery_img)

        grid = make_grid(torch.stack(images), k + 1)
        grid -= grid.min()
        grid /= grid.max()

        plt.imsave(
            f'{self._output_prefix}_{str(time())}.png', grid.permute(1, 2, 0).numpy())

    def _data_loader(self, dataset: Market1501) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
            persistent_workers=False,
        )

    @torch.no_grad()  # 이거 안넣으면 램 해제 안됨
    def _extract_features(self, dataset: Market1501):
        print(f'extracting features of {dataset.mode}...')
        dataloader = self._data_loader(dataset)
        features = torch.Tensor().cuda()
        for (images, _) in tqdm(dataloader):
            images = images.cuda()
            features = torch.cat([features, self._encoder.encode(images)])
        return features

    @property
    def matrix(self):
        return self._matrix

    def summary(self, k_list: list[int]):
        results = []
        for k in k_list:
            acc = self.evaluate_top_k_accuracy(k)
            results.append((k, acc))
        print('# Accuracies:')
        for item in results:
            print(f'- top-{item[0]}: {item[1]}')
