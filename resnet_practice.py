from PIL import Image
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
from torchvision import models
from torchvision.transforms import Compose, ToTensor

from models.components import ResnetBackbone


def main():
    transforms = Compose(
        [
            ToTensor(),
            imagenet_normalization(),
        ]
    )

    img = transforms(Image.open("./data/samples/tench.jpg"))

    model = models.resnet152(True)

    model = ResnetBackbone("resnet50", True, True)
    ...


if __name__ == "__main__":
    main()
