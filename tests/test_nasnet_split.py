"""Tests for the deterministic NASNet CIFAR-10 search split."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.nasnet_cifar10 import (
    CIFAR10_TRAIN_SIZE,
    NASNET_SEARCH_TRAIN_SIZE,
    NASNET_SEARCH_VAL_SIZE,
    NASNET_SPLIT_SEED,
    create_nasnet_split_indices,
    load_nasnet_split,
    validate_nasnet_split_indices,
)


def test_split_sizes_disjointness_and_complete_coverage() -> None:
    train_indices, val_indices = create_nasnet_split_indices(
        NASNET_SPLIT_SEED
    )

    assert len(train_indices) == 45_000
    assert len(val_indices) == 5_000
    assert len(train_indices) == NASNET_SEARCH_TRAIN_SIZE
    assert len(val_indices) == NASNET_SEARCH_VAL_SIZE

    assert len(np.unique(train_indices)) == 45_000
    assert len(np.unique(val_indices)) == 5_000
    assert not set(train_indices) & set(val_indices)
    assert len(set(train_indices) | set(val_indices)) == 50_000

    assert train_indices.min() >= 0
    assert val_indices.min() >= 0
    assert train_indices.max() < CIFAR10_TRAIN_SIZE
    assert val_indices.max() < CIFAR10_TRAIN_SIZE


def test_same_seed_produces_identical_split() -> None:
    train_1, val_1 = create_nasnet_split_indices(NASNET_SPLIT_SEED)
    train_2, val_2 = create_nasnet_split_indices(NASNET_SPLIT_SEED)

    np.testing.assert_array_equal(train_1, train_2)
    np.testing.assert_array_equal(val_1, val_2)


def test_different_seed_changes_split() -> None:
    train_1, val_1 = create_nasnet_split_indices(NASNET_SPLIT_SEED)
    train_2, val_2 = create_nasnet_split_indices(NASNET_SPLIT_SEED + 1)

    assert not np.array_equal(train_1, train_2)
    assert not np.array_equal(val_1, val_2)


def test_split_round_trip_from_npy_files(tmp_path) -> None:
    train_indices, val_indices = create_nasnet_split_indices()
    np.save(tmp_path / "train_indices.npy", train_indices, allow_pickle=False)
    np.save(tmp_path / "val_indices.npy", val_indices, allow_pickle=False)

    loaded_train, loaded_val = load_nasnet_split(tmp_path)

    np.testing.assert_array_equal(loaded_train, train_indices)
    np.testing.assert_array_equal(loaded_val, val_indices)


def test_validation_rejects_overlap() -> None:
    train_indices, val_indices = create_nasnet_split_indices()
    invalid_val = val_indices.copy()
    invalid_val[0] = train_indices[0]

    with pytest.raises(ValueError, match="overlap"):
        validate_nasnet_split_indices(train_indices, invalid_val)


def test_validation_rejects_wrong_sizes() -> None:
    train_indices, val_indices = create_nasnet_split_indices()

    with pytest.raises(ValueError, match="45000"):
        validate_nasnet_split_indices(train_indices[:-1], val_indices)
