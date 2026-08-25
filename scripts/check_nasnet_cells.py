"""Stress-check random PyTorch NASNet Normal and Reduction Cells.

Run from the project root:

    python scripts/check_nasnet_cells.py

The default run creates 100 deterministic random architectures and performs
one Normal Cell plus one Reduction Cell forward for each architecture.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch


# Make ``src`` importable when this file is executed directly as
# ``python scripts/check_nasnet_cells.py`` without installing the project.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nasnet.cell import NASNetCell  # noqa: E402
from src.nasnet.genotype import (  # noqa: E402
    get_unused_states,
    random_architecture,
)


DEFAULT_SEED = 20260825
DEFAULT_NUM_ARCHITECTURES = 100
DEFAULT_BATCH_SIZE = 2
INPUT_SIZE = 32
BASE_CHANNELS = 24
REDUCTION_CHANNELS = 48


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False"
        )
    return device


def _expected_channels(gene, cell_channels: int) -> int:
    unused_indices = get_unused_states(gene)

    if not unused_indices:
        raise AssertionError("cell has no unused generated state")
    if not all(2 <= index <= 6 for index in unused_indices):
        raise AssertionError(
            "get_unused_states() must return only generated states 2..6; "
            f"got {unused_indices}"
        )

    return len(unused_indices) * cell_channels


def check_random_cells(
    *,
    num_architectures: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> None:
    if num_architectures <= 0:
        raise ValueError("num_architectures must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    rng = random.Random(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    s0 = torch.randn(
        batch_size,
        BASE_CHANNELS,
        INPUT_SIZE,
        INPUT_SIZE,
        device=device,
    )
    s1 = torch.randn_like(s0)

    normal_passed = 0
    reduction_passed = 0

    for case_index in range(num_architectures):
        arch = random_architecture(rng)

        normal = NASNetCell(
            gene=arch.normal,
            prev_channels=BASE_CHANNELS,
            curr_channels=BASE_CHANNELS,
            cell_channels=BASE_CHANNELS,
            reduction=False,
        ).to(device)
        normal.eval()

        reduction = NASNetCell(
            gene=arch.reduction,
            prev_channels=BASE_CHANNELS,
            curr_channels=BASE_CHANNELS,
            cell_channels=REDUCTION_CHANNELS,
            reduction=True,
        ).to(device)
        reduction.eval()

        with torch.no_grad():
            normal_output = normal(s0, s1)
            reduction_output = reduction(s0, s1)

        expected_normal_channels = _expected_channels(
            arch.normal,
            BASE_CHANNELS,
        )
        expected_reduction_channels = _expected_channels(
            arch.reduction,
            REDUCTION_CHANNELS,
        )

        expected_normal_shape = (
            batch_size,
            expected_normal_channels,
            INPUT_SIZE,
            INPUT_SIZE,
        )
        expected_reduction_shape = (
            batch_size,
            expected_reduction_channels,
            INPUT_SIZE // 2,
            INPUT_SIZE // 2,
        )

        if tuple(normal_output.shape) != expected_normal_shape:
            raise AssertionError(
                f"random case {case_index}: Normal Cell shape mismatch; "
                f"expected {expected_normal_shape}, "
                f"got {tuple(normal_output.shape)}"
            )
        if tuple(reduction_output.shape) != expected_reduction_shape:
            raise AssertionError(
                f"random case {case_index}: Reduction Cell shape mismatch; "
                f"expected {expected_reduction_shape}, "
                f"got {tuple(reduction_output.shape)}"
            )

        if normal.output_channels != expected_normal_channels:
            raise AssertionError(
                f"random case {case_index}: Normal Cell output_channels "
                f"metadata is {normal.output_channels}, expected "
                f"{expected_normal_channels}"
            )
        if reduction.output_channels != expected_reduction_channels:
            raise AssertionError(
                f"random case {case_index}: Reduction Cell output_channels "
                f"metadata is {reduction.output_channels}, expected "
                f"{expected_reduction_channels}"
            )

        if not torch.isfinite(normal_output).all():
            raise AssertionError(
                f"random case {case_index}: Normal Cell produced NaN or Inf"
            )
        if not torch.isfinite(reduction_output).all():
            raise AssertionError(
                f"random case {case_index}: Reduction Cell produced NaN or Inf"
            )

        normal_passed += 1
        reduction_passed += 1

    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Normal Cells: {normal_passed}/{num_architectures} PASSED")
    print(f"Reduction Cells: {reduction_passed}/{num_architectures} PASSED")
    print(f"{num_architectures} random NASNet cell pairs: PASSED")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic structural forwards for random NASNet "
            "Normal and Reduction Cells."
        )
    )
    parser.add_argument(
        "--num-architectures",
        type=int,
        default=DEFAULT_NUM_ARCHITECTURES,
        help="number of random architecture pairs (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="dummy input batch size (default: 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="random seed (default: 20260825)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device, for example cpu, cuda, or auto (default: auto)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)

    check_random_cells(
        num_architectures=args.num_architectures,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
    )


if __name__ == "__main__":
    main()
