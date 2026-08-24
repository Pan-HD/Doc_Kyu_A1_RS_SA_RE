import json
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR10


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data" / "cifar10" / "raw"
SPLIT_ROOT = PROJECT_ROOT / "data" / "cifar10" / "splits"

SEED = 42

NUM_CLASSES = 10

# Per class:
# 2000 x 10 = 20,000 search training
#  500 x 10 =  5,000 validation
TRAIN_PER_CLASS = 2000
VAL_PER_CLASS = 500


def main():

    print("=" * 70)
    print("CIFAR-10 preparation")
    print("=" * 70)

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    SPLIT_ROOT.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Download/load official CIFAR-10 training set
    # --------------------------------------------------------
    print("\n[1/5] Loading CIFAR-10 training set...")

    train_dataset = CIFAR10(
        root=DATA_ROOT,
        train=True,
        download=True
    )

    targets = np.array(train_dataset.targets)

    print(f"Official training set: {len(train_dataset)} images")

    # --------------------------------------------------------
    # 2. Load official test set
    # --------------------------------------------------------
    print("\n[2/5] Loading CIFAR-10 official test set...")

    test_dataset = CIFAR10(
        root=DATA_ROOT,
        train=False,
        download=True
    )

    print(f"Official test set: {len(test_dataset)} images")

    # --------------------------------------------------------
    # 3. Stratified split
    # --------------------------------------------------------
    print("\n[3/5] Creating fixed stratified split...")

    rng = np.random.default_rng(SEED)

    search_train_indices = []
    search_val_indices = []
    unused_indices = []

    class_statistics = {}

    for class_id in range(NUM_CLASSES):

        class_indices = np.where(targets == class_id)[0]

        # Randomize within the class
        class_indices = rng.permutation(class_indices)

        train_idx = class_indices[:TRAIN_PER_CLASS]

        val_idx = class_indices[
            TRAIN_PER_CLASS:
            TRAIN_PER_CLASS + VAL_PER_CLASS
        ]

        unused_idx = class_indices[
            TRAIN_PER_CLASS + VAL_PER_CLASS:
        ]

        search_train_indices.extend(train_idx)
        search_val_indices.extend(val_idx)
        unused_indices.extend(unused_idx)

        class_statistics[str(class_id)] = {
            "class_name": train_dataset.classes[class_id],
            "search_train": len(train_idx),
            "search_validation": len(val_idx),
            "unused": len(unused_idx),
        }

    # Shuffle final index arrays as well
    search_train_indices = rng.permutation(
        np.array(search_train_indices, dtype=np.int64)
    )

    search_val_indices = rng.permutation(
        np.array(search_val_indices, dtype=np.int64)
    )

    unused_indices = rng.permutation(
        np.array(unused_indices, dtype=np.int64)
    )

    # --------------------------------------------------------
    # 4. Validation
    # --------------------------------------------------------
    print("\n[4/5] Validating split...")

    assert len(search_train_indices) == 20000
    assert len(search_val_indices) == 5000
    assert len(unused_indices) == 25000

    train_set = set(search_train_indices.tolist())
    val_set = set(search_val_indices.tolist())
    unused_set = set(unused_indices.tolist())

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(unused_set)
    assert val_set.isdisjoint(unused_set)

    all_indices = train_set | val_set | unused_set

    assert len(all_indices) == 50000

    print("No overlap between subsets: OK")
    print("All 50,000 training images accounted for: OK")

    # --------------------------------------------------------
    # 5. Save
    # --------------------------------------------------------
    print("\n[5/5] Saving split files...")

    np.save(
        SPLIT_ROOT / "search_train_indices.npy",
        search_train_indices
    )

    np.save(
        SPLIT_ROOT / "search_val_indices.npy",
        search_val_indices
    )

    np.save(
        SPLIT_ROOT / "unused_indices.npy",
        unused_indices
    )

    split_info = {
        "dataset": "CIFAR-10",
        "version": "v0.3",
        "seed": SEED,
        "official_training_size": 50000,
        "search_training_size": 20000,
        "search_validation_size": 5000,
        "unused_training_size": 25000,
        "official_test_size": 10000,
        "split_method": "stratified random split",
        "class_statistics": class_statistics,
        "important_note": (
            "The remaining 25,000 CIFAR-10 training images "
            "must not be used during the current experiments."
        )
    }

    with open(
        SPLIT_ROOT / "split_info.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            split_info,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 70)
    print("CIFAR-10 split completed")
    print("=" * 70)

    print(f"Search training : {len(search_train_indices):,}")
    print(f"Validation      : {len(search_val_indices):,}")
    print(f"Unused          : {len(unused_indices):,}")
    print(f"Official test   : {len(test_dataset):,}")

    print("\nClass distribution:")

    for class_id in range(NUM_CLASSES):

        name = train_dataset.classes[class_id]

        train_count = np.sum(
            targets[search_train_indices] == class_id
        )

        val_count = np.sum(
            targets[search_val_indices] == class_id
        )

        unused_count = np.sum(
            targets[unused_indices] == class_id
        )

        print(
            f"{class_id:2d} {name:12s}: "
            f"train={train_count:4d}, "
            f"val={val_count:3d}, "
            f"unused={unused_count:4d}"
        )

    print("\nSaved to:")
    print(SPLIT_ROOT)


if __name__ == "__main__":
    main()