import glob
import os
import os.path
import re
from os.path import join
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import VisionDataset


class Market1501(VisionDataset):
    name = "Market1501"

    def __init__(
        self,
        root: str,
        mode: str = "train",
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        market1501_500k=False,
    ):
        super().__init__(
            root, transform=transform, target_transform=target_transform
        )
        self.mode = mode
        self.market1501_500k = market1501_500k

        self.image_paths, self.targets = self._load_data()
        self.classes = set(self.targets)

    def _load_data(self):
        data = []
        if self.mode == "train":
            data = _process_dir(
                join(self.root, self.name, "bounding_box_train"), relabel=True
            )
        elif self.mode == "query":
            data = _process_dir(
                join(self.root, self.name, "query"), relabel=False
            )
        elif self.mode == "gallery":
            data = _process_dir(
                join(self.root, self.name, "bounding_box_test"), relabel=False
            )
            if self.market1501_500k:
                data += _process_dir(
                    join(self.root, self.name, "images"),
                    relabel=False,
                )
        targets = [i[1] for i in data]  # (pid, cid)
        image_paths = [i[0] for i in data]

        return image_paths, torch.Tensor(targets).long()

    def __getitem__(self, index):
        img_path, target = self.image_paths[index], self.targets[index]
        img = _read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.targets)

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}


def _read_image(path):
    got_img = False
    if not os.path.exists(path):
        raise IOError('"{}" does not exist'.format(path))
    while not got_img:
        try:
            img = Image.open(path).convert("RGB")
            got_img = True
        except IOError:
            print(
                'IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'.format(
                    path
                )
            )
    return img


def _process_dir(dir_path, relabel=False) -> List[Tuple[str, int, int]]:
    img_paths = glob.glob(os.path.join(dir_path, "*.jpg"))
    pattern = re.compile(r"([-\d]+)_c(\d)")
    pid_container = set()
    for img_path in img_paths:
        pid, _ = map(int, pattern.search(img_path).groups())
        if pid == -1:
            continue  # junk images are just ignored
        pid_container.add(pid)
    pid2label = {pid: label for label, pid in enumerate(pid_container)}
    data = []
    for img_path in img_paths:
        pid, camid = map(int, pattern.search(img_path).groups())
        if pid == -1:
            continue  # junk images are just ignored
        assert 0 <= pid <= 1501  # pid == 0 means background
        assert 1 <= camid <= 6
        camid -= 1  # index starts from 0
        if relabel:
            pid = pid2label[pid]
        data.append((img_path, pid, camid))  # (IMG_SRC, Person ID, Cam ID)
    return data


class PairedMarket1501(Market1501):
    def __init__(
        self,
        *args: any,
        **kwargs: any,
    ):
        super().__init__(*args, **kwargs)

        self.targets_set = set(self.targets.numpy())
        self.target_to_indices = {
            target: np.where(self.targets.numpy() == target)[0]
            for target in self.targets_set
        }

    def __getitem__(self, index_a):
        img_a, target_a = super().__getitem__(index_a)
        target_a = int(target_a)

        index_p = index_a
        while index_p == index_a:
            index_p = np.random.choice(self.target_to_indices[target_a])
        img_p, target_p = super().__getitem__(index_p)

        target_n = np.random.choice(list(self.targets_set - set([target_a])))
        index_n = np.random.choice(self.target_to_indices[target_n])
        img_n, target_n = super().__getitem__(index_n)

        return (img_a, img_p, img_n), (target_a, target_p, target_n)
