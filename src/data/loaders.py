from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from src.data.cifar10 import (
    get_cifar10_train_transform,
    get_cifar10_val_transform,
)


def create_search_datasets(
    data_root="data/cifar10",
):
    """
    Create CIFAR-10 search-train and search-validation datasets.

    Important:
    train and validation use separate CIFAR10 objects because
    their transforms are different.
    """

    data_root = Path(data_root)
    split_root = data_root / "splits"

    train_indices = np.load(
        split_root / "search_train_indices.npy"
    )

    val_indices = np.load(
        split_root / "search_val_indices.npy"
    )

    # Training dataset:
    # random augmentation is enabled.
    train_base = CIFAR10(
        root=str(data_root),
        train=True,
        download=False,
        transform=get_cifar10_train_transform(),
    )

    # Validation dataset:
    # NO random augmentation.
    val_base = CIFAR10(
        root=str(data_root),
        train=True,
        download=False,
        transform=get_cifar10_val_transform(),
    )

    search_train = Subset(
        train_base,
        train_indices.tolist(),
    )

    search_val = Subset(
        val_base,
        val_indices.tolist(),
    )

    return search_train, search_val


def create_search_loaders(
    data_root="data/cifar10",
    batch_size=128,
    num_workers=0,
):
    """
    Create DataLoaders for architecture/search training and validation.
    """

    search_train, search_val = create_search_datasets(
        data_root=data_root,
    )

    train_loader = DataLoader(
        search_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        search_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader