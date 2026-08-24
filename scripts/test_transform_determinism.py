import torch
from torchvision.datasets import CIFAR10

from src.data.cifar10 import (
    get_cifar10_train_transform,
    get_cifar10_val_transform,
)


def main():

    train_dataset = CIFAR10(
        root="data/cifar10",
        train=True,
        download=False,
        transform=get_cifar10_train_transform(),
    )

    val_dataset = CIFAR10(
        root="data/cifar10",
        train=True,
        download=False,
        transform=get_cifar10_val_transform(),
    )

    index = 0

    train_img_1, _ = train_dataset[index]
    train_img_2, _ = train_dataset[index]

    val_img_1, _ = val_dataset[index]
    val_img_2, _ = val_dataset[index]

    print(
        "Train identical:",
        torch.equal(train_img_1, train_img_2),
    )

    print(
        "Validation identical:",
        torch.equal(val_img_1, val_img_2),
    )


if __name__ == "__main__":
    main()