import os
from typing import List, Optional, Union

import torch
from datasets.market1501 import Market1501, PairedMarket1501
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms as transform_lib


class Market1501DataModule(LightningDataModule):
    dataset_cls = Market1501
    name = "market1501"
    dims = (3, 128, 64)

    def __init__(
        self,
        data_dir: Optional[str],
        val_split: Union[int, float] = 0.2,
        num_workers: int = 4,
        normalize: bool = True,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        persistent_workers: bool = True,
        *args: any,
        **kwargs: any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.val_split = val_split
        self.num_workers = num_workers
        self.normalize = normalize
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.persistent_workers = persistent_workers

    def prepare_data(self, *args: any, **kwargs: any) -> None:
        self.num_classes = len(self.dataset_cls(self.data_dir).classes)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            train_transforms = (
                self.default_transforms()
                if self.train_transforms is None
                else self.train_transforms
            )
            val_transforms = (
                self.default_transforms()
                if self.val_transforms is None
                else self.val_transforms
            )

            dataset_train = self.dataset_cls(
                self.data_dir, mode="train", transform=train_transforms
            )
            dataset_val = self.dataset_cls(
                self.data_dir, mode="train", transform=val_transforms
            )

            # Split
            self.dataset_train = self._split_dataset(dataset_train)
            self.dataset_val = self._split_dataset(dataset_val, train=False)

        if stage == "test" or stage is None:
            test_transforms = (
                self.default_transforms()
                if self.test_transforms is None
                else self.test_transforms
            )
            self.dataset_test = self.dataset_cls(
                # 나중에 query, gallery에 맞게 바꿔야함!
                self.data_dir,
                mode="train",
                transform=test_transforms,
            )

    def _split_dataset(self, dataset: Dataset, train: bool = True) -> Dataset:
        len_dataset = len(dataset)  # type: ignore[arg-type]
        splits = self._get_splits(len_dataset)
        dataset_train, dataset_val = random_split(
            dataset, splits, generator=torch.Generator().manual_seed(self.seed)
        )

        if train:
            return dataset_train
        return dataset_val

    def _get_splits(self, len_dataset: int) -> List[int]:
        if isinstance(self.val_split, int):
            train_len = len_dataset - self.val_split
            splits = [train_len, self.val_split]
        elif isinstance(self.val_split, float):
            val_len = int(self.val_split * len_dataset)
            train_len = len_dataset - val_len
            splits = [train_len, val_len]
        else:
            raise ValueError(f"Unsupported type {type(self.val_split)}")

        return splits

    def default_transforms(self) -> callable:
        if self.normalize:
            mnist_transforms = transform_lib.Compose(
                [
                    transform_lib.Resize((256, 128), interpolation=3),
                    transform_lib.ToTensor(),
                    transform_lib.Normalize(mean=(0.5,), std=(0.5,)),
                ]
            )
        else:
            mnist_transforms = transform_lib.Compose(
                [
                    transform_lib.Resize((256, 128), interpolation=3),
                    transform_lib.ToTensor(),
                ]
            )

        return mnist_transforms

    def train_dataloader(self, *args: any, **kwargs: any) -> DataLoader:
        return self._data_loader(self.dataset_train, shuffle=self.shuffle)

    def val_dataloader(
        self, *args: any, **kwargs: any
    ) -> Union[DataLoader, List[DataLoader]]:
        return self._data_loader(self.dataset_val)

    def test_dataloader(
        self, *args: any, **kwargs: any
    ) -> Union[DataLoader, List[DataLoader]]:
        return self._data_loader(self.dataset_test)

    def _data_loader(
        self, dataset: Dataset, shuffle: bool = False
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
            if self.num_workers > 0
            else False,
        )


class PairedMarket1501DataModule(Market1501DataModule):
    dataset_cls = PairedMarket1501

    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)
