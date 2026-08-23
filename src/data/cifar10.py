from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def ensure_fixed_split(
    split_dir,
    train_size=20000,
    val_size=5000,
    split_seed=20260823,
):
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    train_path = split_dir / "train_indices.npy"
    val_path = split_dir / "val_indices.npy"

    if train_path.exists() and val_path.exists():
        return np.load(train_path), np.load(val_path)

    rng = np.random.default_rng(split_seed)
    indices = np.arange(50000)
    rng.shuffle(indices)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]

    np.save(train_path, train_idx)
    np.save(val_path, val_idx)
    return train_idx, val_idx


def build_cifar10_loaders(
    data_root,
    split_dir,
    train_size=20000,
    val_size=5000,
    split_seed=20260823,
    batch_size=128,
    num_workers=2,
    pin_memory=True,
):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_full_aug = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform
    )
    train_full_eval = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=eval_transform
    )

    train_idx, val_idx = ensure_fixed_split(
        split_dir,
        train_size=train_size,
        val_size=val_size,
        split_seed=split_seed,
    )

    train_ds = Subset(train_full_aug, train_idx.tolist())
    val_ds = Subset(train_full_eval, val_idx.tolist())

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_ds = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=eval_transform
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
