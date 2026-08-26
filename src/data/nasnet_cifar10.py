"""CIFAR-10 data and fixed NASNet search-split utilities.

The 50,000 official CIFAR-10 training examples are divided into a fixed
45,000-example search-training set and a 5,000-example search-validation set.
The official 10,000-example test set is returned separately and must never be
used as architecture-search fitness.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


CIFAR10_TRAIN_SIZE = 50_000
CIFAR10_TEST_SIZE = 10_000
NASNET_SEARCH_TRAIN_SIZE = 45_000
NASNET_SEARCH_VAL_SIZE = 5_000
NASNET_SPLIT_SEED = 20_260_823
DEFAULT_SPLIT_DIR = Path("data/splits/nasnet_v041")

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass(frozen=True)
class CIFAR10SearchLoaders:
    """Loaders for search training, search validation, and final testing."""

    train_loader: Any
    val_loader: Any
    official_test_loader: Any
    train_indices: np.ndarray
    val_indices: np.ndarray


def create_nasnet_split_indices(
    seed: int = NASNET_SPLIT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic 45k/5k indices over CIFAR-10's training set."""

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("split seed must be an integer")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(CIFAR10_TRAIN_SIZE).astype(
        np.int64,
        copy=False,
    )
    train_indices = indices[:NASNET_SEARCH_TRAIN_SIZE].copy()
    val_indices = indices[NASNET_SEARCH_TRAIN_SIZE:].copy()

    validate_nasnet_split_indices(train_indices, val_indices)
    return train_indices, val_indices


def validate_nasnet_split_indices(
    train_indices: np.ndarray,
    val_indices: np.ndarray,
) -> None:
    """Raise ``ValueError`` unless indices form an exact 45k/5k partition."""

    train_indices = np.asarray(train_indices)
    val_indices = np.asarray(val_indices)

    for name, indices, expected_size in (
        ("train", train_indices, NASNET_SEARCH_TRAIN_SIZE),
        ("validation", val_indices, NASNET_SEARCH_VAL_SIZE),
    ):
        if indices.ndim != 1:
            raise ValueError(f"{name} indices must be one-dimensional")
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"{name} indices must have an integer dtype")
        if len(indices) != expected_size:
            raise ValueError(
                f"{name} split has {len(indices)} examples; "
                f"expected {expected_size}"
            )
        if len(np.unique(indices)) != expected_size:
            raise ValueError(f"{name} split contains duplicate indices")
        if indices.min() < 0 or indices.max() >= CIFAR10_TRAIN_SIZE:
            raise ValueError(
                f"{name} indices must be in 0..{CIFAR10_TRAIN_SIZE - 1}"
            )

    overlap = np.intersect1d(
        train_indices,
        val_indices,
        assume_unique=False,
    )
    if overlap.size:
        raise ValueError(
            "train and validation splits overlap; "
            f"first overlapping index is {int(overlap[0])}"
        )

    covered = np.union1d(train_indices, val_indices)
    if len(covered) != CIFAR10_TRAIN_SIZE:
        raise ValueError(
            "train and validation splits do not cover all 50,000 "
            "CIFAR-10 training examples"
        )


def load_nasnet_split(
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate the stored NASNet v0.4.1 search split."""

    split_dir = Path(split_dir)
    train_path = split_dir / "train_indices.npy"
    val_path = split_dir / "val_indices.npy"

    missing = [
        str(path)
        for path in (train_path, val_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "NASNet split files are missing: "
            + ", ".join(missing)
            + ". Run scripts/create_nasnet_split.py first."
        )

    train_indices = np.load(train_path, allow_pickle=False)
    val_indices = np.load(val_path, allow_pickle=False)
    validate_nasnet_split_indices(train_indices, val_indices)

    return (
        train_indices.astype(np.int64, copy=False),
        val_indices.astype(np.int64, copy=False),
    )


def build_cifar10_transforms(
    augment_train: bool = True,
) -> tuple[Any, Any]:
    """Build provisional basic CIFAR transforms without Cutout or AutoAugment."""

    from torchvision import transforms

    train_steps = []
    if augment_train:
        train_steps.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    train_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return transforms.Compose(train_steps), eval_transform


def _seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy from PyTorch's deterministic worker seed."""

    del worker_id
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_cifar10_search_loaders(
    data_root: str | Path,
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
    batch_size: int = 128,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    download: bool = False,
    augment_train: bool = True,
    loader_seed: int = NASNET_SPLIT_SEED,
    train_transform: Any | None = None,
    eval_transform: Any | None = None,
) -> CIFAR10SearchLoaders:
    """Construct fixed search loaders and a separately named test loader.

    ``official_test_loader`` is for post-search reporting only. Architecture
    search and fitness evaluation must use ``val_loader`` exclusively.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision.datasets import CIFAR10

    train_indices, val_indices = load_nasnet_split(split_dir)

    if train_transform is None or eval_transform is None:
        default_train, default_eval = build_cifar10_transforms(
            augment_train=augment_train,
        )
        if train_transform is None:
            train_transform = default_train
        if eval_transform is None:
            eval_transform = default_eval

    data_root = str(Path(data_root))
    training_base = CIFAR10(
        root=data_root,
        train=True,
        transform=train_transform,
        download=download,
    )
    validation_base = CIFAR10(
        root=data_root,
        train=True,
        transform=eval_transform,
        download=download,
    )
    official_test = CIFAR10(
        root=data_root,
        train=False,
        transform=eval_transform,
        download=download,
    )

    if len(training_base) != CIFAR10_TRAIN_SIZE:
        raise RuntimeError(
            f"official CIFAR-10 train dataset has {len(training_base)} "
            f"examples; expected {CIFAR10_TRAIN_SIZE}"
        )
    if len(official_test) != CIFAR10_TEST_SIZE:
        raise RuntimeError(
            f"official CIFAR-10 test dataset has {len(official_test)} "
            f"examples; expected {CIFAR10_TEST_SIZE}"
        )

    training_subset = Subset(training_base, train_indices.tolist())
    validation_subset = Subset(validation_base, val_indices.tolist())

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    generator = torch.Generator()
    generator.manual_seed(loader_seed)
    common_loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
        "worker_init_fn": _seed_worker,
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(
        training_subset,
        shuffle=True,
        generator=generator,
        **common_loader_options,
    )
    val_loader = DataLoader(
        validation_subset,
        shuffle=False,
        **common_loader_options,
    )
    official_test_loader = DataLoader(
        official_test,
        shuffle=False,
        **common_loader_options,
    )

    return CIFAR10SearchLoaders(
        train_loader=train_loader,
        val_loader=val_loader,
        official_test_loader=official_test_loader,
        train_indices=train_indices,
        val_indices=val_indices,
    )


__all__ = [
    "CIFAR10_MEAN",
    "CIFAR10_STD",
    "CIFAR10_TEST_SIZE",
    "CIFAR10_TRAIN_SIZE",
    "CIFAR10SearchLoaders",
    "DEFAULT_SPLIT_DIR",
    "NASNET_SEARCH_TRAIN_SIZE",
    "NASNET_SEARCH_VAL_SIZE",
    "NASNET_SPLIT_SEED",
    "build_cifar10_search_loaders",
    "build_cifar10_transforms",
    "create_nasnet_split_indices",
    "load_nasnet_split",
    "validate_nasnet_split_indices",
]
