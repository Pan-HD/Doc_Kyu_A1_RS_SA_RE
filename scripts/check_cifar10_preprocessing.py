import numpy as np
from torchvision.datasets import CIFAR10

from src.data.cifar10 import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    get_cifar10_train_transform,
    get_cifar10_val_transform,
)


def main():
    print("=" * 60)
    print("CIFAR-10 preprocessing check")
    print("=" * 60)

    print("\nMean:")
    print(CIFAR10_MEAN)

    print("\nStd:")
    print(CIFAR10_STD)

    train_transform = get_cifar10_train_transform()
    val_transform = get_cifar10_val_transform()

    print("\nTrain transform:")
    print(train_transform)

    print("\nValidation transform:")
    print(val_transform)

    train_indices = np.load(
        "data/cifar10/splits/search_train_indices.npy"
    )

    val_indices = np.load(
        "data/cifar10/splits/search_val_indices.npy"
    )

    train_dataset = CIFAR10(
        root="data/cifar10",
        train=True,
        download=True,
        transform=train_transform,
    )

    val_dataset = CIFAR10(
        root="data/cifar10",
        train=True,
        download=True,
        transform=val_transform,
    )

    train_index = int(train_indices[0])
    val_index = int(val_indices[0])

    train_img, train_label = train_dataset[train_index]
    val_img, val_label = val_dataset[val_index]

    print("\nTrain sample:")
    print("shape:", train_img.shape)
    print("dtype:", train_img.dtype)
    print("label:", train_label)
    print("min:", train_img.min().item())
    print("max:", train_img.max().item())

    print("\nValidation sample:")
    print("shape:", val_img.shape)
    print("dtype:", val_img.dtype)
    print("label:", val_label)
    print("min:", val_img.min().item())
    print("max:", val_img.max().item())

    print("\nCheck completed.")


if __name__ == "__main__":
    main()