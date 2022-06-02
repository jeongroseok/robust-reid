from typing import Optional

import numpy as np
from PIL import Image
from torchvision.datasets import MNIST


class PairedMNIST(MNIST):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[callable] = None,
            target_transform: Optional[callable] = None,
            download: bool = False,
    ):
        PairedMNIST.__name__ = MNIST.__name__  # optim trick
        super().__init__(root, train, transform, target_transform, download)

        self.targets_set = set(self.targets.numpy())
        self.target_to_indices = {target: np.where(self.targets.numpy() == target)[
            0] for target in self.targets_set}

    def __getitem__(self, index_a):
        img_a, target_a = self.data[index_a], int(self.targets[index_a])

        index_p = index_a
        while index_p == index_a:
            index_p = np.random.choice(self.target_to_indices[target_a])
        img_p, target_p = self.data[index_p], int(self.targets[index_p])

        target_n = np.random.choice(list(self.targets_set - set([target_a])))
        index_n = np.random.choice(self.target_to_indices[target_n])
        img_n, target_n = self.data[index_n], int(self.targets[index_n])

        img_a = Image.fromarray(img_a.numpy(), mode='L')
        img_p = Image.fromarray(img_p.numpy(), mode='L')
        img_n = Image.fromarray(img_n.numpy(), mode='L')

        if self.transform is not None:
            img_a = self.transform(img_a)
            img_p = self.transform(img_p)
            img_n = self.transform(img_n)

        if self.target_transform is not None:
            target_a = self.target_transform(target_a)
            target_p = self.target_transform(target_p)
            target_n = self.target_transform(target_n)

        return (img_a, img_p, img_n), (target_a, target_p, target_n)
