from typing import Optional, Union

from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pl_examples import _DATASETS_PATH
from torchvision import transforms as transform_lib

from datasets.paired_mnist import PairedMNIST


class PairedMNISTDataModule(VisionDataModule):
    name = "mnist"
    dataset_cls = PairedMNIST
    dims = (1, 28, 28)

    def __init__(
        self,
        data_dir: Optional[str] = _DATASETS_PATH,
        val_split: Union[int, float] = 0.2,
        num_workers: int = 4,
        normalize: bool = True,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        *args: any,
        **kwargs: any,
    ) -> None:
        super().__init__(  # type: ignore[misc]
            data_dir=data_dir,
            val_split=val_split,
            num_workers=num_workers,
            normalize=normalize,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            *args,
            **kwargs,
        )

    @property
    def num_classes(self) -> int:
        return 10

    def default_transforms(self) -> callable:
        if self.normalize:
            mnist_transforms = transform_lib.Compose([
                transform_lib.ToTensor(), transform_lib.Normalize(mean=(0.5, ), std=(0.5, ))
            ])
        else:
            mnist_transforms = transform_lib.Compose(
                [transform_lib.ToTensor()])

        return mnist_transforms
