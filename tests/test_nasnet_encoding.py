"""Tests for the fixed-order 280-D NASNet surrogate encoding."""

from __future__ import annotations

import random
from dataclasses import replace

import numpy as np

from src.nasnet.encoding import (
    ARCHITECTURE_ENCODING_DIM,
    BRANCH_ENCODING_DIM,
    NUM_BRANCHES,
    NUM_INPUT_STATES,
    NUM_OPERATIONS,
    OP_NAMES,
    encode_architecture,
)
from src.nasnet.genotype import random_architecture


SEED = 20260826


def _replace_branch(
    architecture,
    cell_name: str,
    pair_index: int,
    branch_name: str,
    **branch_changes,
):
    cell = getattr(architecture, cell_name)
    pair = cell.pairs[pair_index]
    branch = getattr(pair, branch_name)

    new_branch = replace(branch, **branch_changes)
    new_pair = replace(pair, **{branch_name: new_branch})

    new_pairs = list(cell.pairs)
    new_pairs[pair_index] = new_pair
    new_cell = replace(cell, pairs=tuple(new_pairs))

    return replace(architecture, **{cell_name: new_cell})


def _change_first_operation(architecture, cell_name: str):
    branch = getattr(architecture, cell_name).pairs[0].branch_1
    new_operation = next(
        operation
        for operation in OP_NAMES
        if operation != branch.op
    )

    return _replace_branch(
        architecture,
        cell_name,
        pair_index=0,
        branch_name="branch_1",
        op=new_operation,
    )


def test_encoding_shape_dtype_and_sum() -> None:
    architecture = random_architecture(random.Random(SEED))
    encoding = encode_architecture(architecture)

    assert ARCHITECTURE_ENCODING_DIM == 280
    assert encoding.shape == (280,)
    assert encoding.dtype == np.float32
    assert encoding.sum() == 40.0
    assert set(np.unique(encoding)).issubset({0.0, 1.0})


def test_each_branch_contains_two_one_hot_vectors() -> None:
    architecture = random_architecture(random.Random(SEED))
    encoding = encode_architecture(architecture)
    branches = encoding.reshape(NUM_BRANCHES, BRANCH_ENCODING_DIM)

    assert NUM_BRANCHES == 20
    assert BRANCH_ENCODING_DIM == 14
    assert NUM_INPUT_STATES == 6
    assert NUM_OPERATIONS == 8

    np.testing.assert_array_equal(
        branches[:, :NUM_INPUT_STATES].sum(axis=1),
        np.ones(NUM_BRANCHES, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        branches[:, NUM_INPUT_STATES:].sum(axis=1),
        np.ones(NUM_BRANCHES, dtype=np.float32),
    )


def test_same_architecture_has_same_encoding() -> None:
    architecture = random_architecture(random.Random(SEED))

    encoding_1 = encode_architecture(architecture)
    encoding_2 = encode_architecture(architecture)

    np.testing.assert_array_equal(encoding_1, encoding_2)


def test_different_operation_has_different_encoding() -> None:
    architecture = random_architecture(random.Random(SEED))
    changed = _change_first_operation(architecture, "normal")

    assert changed != architecture
    assert not np.array_equal(
        encode_architecture(architecture),
        encode_architecture(changed),
    )


def test_different_input_state_has_different_encoding() -> None:
    architecture = random_architecture(random.Random(SEED))
    branch = architecture.normal.pairs[0].branch_1

    # Pair 0 can reference exactly states 0 and 1.
    new_state = 1 if branch.input_state == 0 else 0
    changed = _replace_branch(
        architecture,
        "normal",
        pair_index=0,
        branch_name="branch_1",
        input_state=new_state,
    )

    assert changed != architecture
    assert not np.array_equal(
        encode_architecture(architecture),
        encode_architecture(changed),
    )


def test_normal_cell_is_encoded_before_reduction_cell() -> None:
    architecture = random_architecture(random.Random(SEED))
    base = encode_architecture(architecture)

    normal_changed = encode_architecture(
        _change_first_operation(architecture, "normal")
    )
    reduction_changed = encode_architecture(
        _change_first_operation(architecture, "reduction")
    )

    normal_difference = np.flatnonzero(base != normal_changed)
    reduction_difference = np.flatnonzero(base != reduction_changed)
    cell_encoding_dim = ARCHITECTURE_ENCODING_DIM // 2

    assert normal_difference.size == 2
    assert reduction_difference.size == 2
    assert np.all(normal_difference < cell_encoding_dim)
    assert np.all(reduction_difference >= cell_encoding_dim)
