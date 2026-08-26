"""Create the deterministic NASNet v0.4.1 CIFAR-10 45k/5k split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.nasnet_cifar10 import (
    DEFAULT_SPLIT_DIR,
    NASNET_SPLIT_SEED,
    create_nasnet_split_indices,
    load_nasnet_split,
)


def create_and_save_split(
    output_dir: str | Path = DEFAULT_SPLIT_DIR,
    seed: int = NASNET_SPLIT_SEED,
    force: bool = False,
) -> tuple[Path, Path]:
    """Create split files without silently replacing different existing data."""

    output_dir = Path(output_dir)
    train_path = output_dir / "train_indices.npy"
    val_path = output_dir / "val_indices.npy"
    train_indices, val_indices = create_nasnet_split_indices(seed)

    existing = (train_path.exists(), val_path.exists())
    if any(existing) and not force:
        if all(existing):
            stored_train, stored_val = load_nasnet_split(output_dir)
            if np.array_equal(stored_train, train_indices) and np.array_equal(
                stored_val,
                val_indices,
            ):
                return train_path, val_path

        raise FileExistsError(
            f"split files already exist in {output_dir} and do not match "
            "the requested seed; use --force only after checking the target"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(train_path, train_indices, allow_pickle=False)
    np.save(val_path, val_indices, allow_pickle=False)

    # Read back from disk so a truncated or incorrectly targeted write fails
    # before the script reports success.
    load_nasnet_split(output_dir)
    return train_path, val_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the fixed 45,000/5,000 CIFAR-10 NASNet search split."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_SPLIT_DIR,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=NASNET_SPLIT_SEED,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace split files in the explicitly selected output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path, val_path = create_and_save_split(
        output_dir=args.output_dir,
        seed=args.seed,
        force=args.force,
    )

    train_indices, val_indices = load_nasnet_split(args.output_dir)
    print(f"train: {train_path} ({len(train_indices)} indices)")
    print(f"validation: {val_path} ({len(val_indices)} indices)")
    print(f"split seed: {args.seed}")


if __name__ == "__main__":
    main()
