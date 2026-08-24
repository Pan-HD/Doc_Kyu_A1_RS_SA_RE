import numpy as np
import torch

from torch.utils.data import RandomSampler, SequentialSampler

from src.data.loaders import (
    create_search_datasets,
    create_search_loaders,
)


def main():

    print("=" * 60)
    print("CIFAR-10 DataLoader / split check")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load split indices
    # --------------------------------------------------

    train_indices = np.load(
        "data/cifar10/splits/search_train_indices.npy"
    )

    val_indices = np.load(
        "data/cifar10/splits/search_val_indices.npy"
    )

    print("\n[1] Split sizes")
    print("search_train:", len(train_indices))
    print("search_val  :", len(val_indices))

    # --------------------------------------------------
    # 2. Check overlap
    # --------------------------------------------------

    overlap = np.intersect1d(
        train_indices,
        val_indices,
    )

    print("\n[2] Train / validation overlap")
    print("overlap count:", len(overlap))

    if len(overlap) != 0:
        raise RuntimeError(
            "ERROR: train and validation indices overlap."
        )

    # --------------------------------------------------
    # 3. Create datasets
    # --------------------------------------------------

    search_train, search_val = create_search_datasets()

    print("\n[3] Dataset sizes")
    print("search_train dataset:", len(search_train))
    print("search_val dataset  :", len(search_val))

    assert len(search_train) == len(train_indices)
    assert len(search_val) == len(val_indices)

    # --------------------------------------------------
    # 4. Check transforms
    # --------------------------------------------------

    print("\n[4] Train transform")
    print(search_train.dataset.transform)

    print("\n[5] Validation transform")
    print(search_val.dataset.transform)

    # --------------------------------------------------
    # 5. Create loaders
    # --------------------------------------------------

    train_loader, val_loader = create_search_loaders(
        batch_size=128,
        num_workers=0,
    )

    print("\n[6] Sampler check")

    print(
        "Train sampler:",
        type(train_loader.sampler).__name__,
    )

    print(
        "Validation sampler:",
        type(val_loader.sampler).__name__,
    )

    if not isinstance(
        train_loader.sampler,
        RandomSampler,
    ):
        raise RuntimeError(
            "Train loader is not shuffled."
        )

    if not isinstance(
        val_loader.sampler,
        SequentialSampler,
    ):
        raise RuntimeError(
            "Validation loader should use shuffle=False."
        )

    # --------------------------------------------------
    # 6. Read first batch
    # --------------------------------------------------

    x_train, y_train = next(iter(train_loader))
    x_val, y_val = next(iter(val_loader))

    print("\n[7] Train batch")
    print("images shape:", x_train.shape)
    print("labels shape:", y_train.shape)
    print("dtype:", x_train.dtype)
    print("labels dtype:", y_train.dtype)

    print("\n[8] Validation batch")
    print("images shape:", x_val.shape)
    print("labels shape:", y_val.shape)
    print("dtype:", x_val.dtype)
    print("labels dtype:", y_val.dtype)

    # --------------------------------------------------
    # 7. Shape checks
    # --------------------------------------------------

    assert x_train.ndim == 4
    assert x_val.ndim == 4

    assert x_train.shape[1:] == (3, 32, 32)
    assert x_val.shape[1:] == (3, 32, 32)

    assert y_train.ndim == 1
    assert y_val.ndim == 1

    assert x_train.dtype == torch.float32
    assert x_val.dtype == torch.float32

    # --------------------------------------------------
    # Final result
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("All DataLoader checks PASSED.")
    print("=" * 60)


if __name__ == "__main__":
    main()