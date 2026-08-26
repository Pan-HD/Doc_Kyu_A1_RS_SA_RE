import json
import random
from dataclasses import replace

from src.nasnet.genotype import (
    NASNetArchitecture,
    get_unused_states,
    random_architecture,
    validate_architecture,
)
from src.nasnet.operations import OPS


def test_operation_registry():
    assert len(OPS) == 8
    assert len(set(OPS)) == 8

    assert "identity" in OPS
    assert "sep_conv_3x3" in OPS
    assert "sep_conv_5x5" in OPS
    assert "sep_conv_7x7" in OPS
    assert "avg_pool_3x3" in OPS
    assert "max_pool_3x3" in OPS
    assert "dil_sep_conv_3x3" in OPS
    assert "conv_1x7_7x1" in OPS


def test_random_architecture_is_valid():
    rng = random.Random(20260824)

    for _ in range(1000):
        arch = random_architecture(rng)
        assert validate_architecture(arch)


def test_each_cell_has_five_pairs():
    rng = random.Random(1)

    for _ in range(100):
        arch = random_architecture(rng)

        assert len(arch.normal.pairs) == 5
        assert len(arch.reduction.pairs) == 5


def test_input_states_are_acyclic():
    rng = random.Random(2)

    for _ in range(1000):
        arch = random_architecture(rng)

        for cell in (arch.normal, arch.reduction):
            for pair_index, pair in enumerate(cell.pairs):
                max_state = pair_index + 1

                assert 0 <= pair.branch_1.input_state <= max_state
                assert 0 <= pair.branch_2.input_state <= max_state


def test_unused_states_are_all_unconsumed_hidden_states():
    """Unused outputs must be selected from all states 0..6."""

    rng = random.Random(20260826)

    for _ in range(1000):
        arch = random_architecture(rng)

        for cell in (arch.normal, arch.reduction):
            used_states = {
                branch.input_state
                for pair in cell.pairs
                for branch in (pair.branch_1, pair.branch_2)
            }
            expected = tuple(sorted(set(range(7)) - used_states))
            unused = get_unused_states(cell)

            assert unused == expected
            assert all(state in range(7) for state in unused)

            # State 6 is created by the final pair, so no later pair can use it.
            assert 6 in unused


def _replace_input_state(cell, old_state, new_state):
    """Return a frozen CellGene with one input-state value replaced."""

    pairs = []

    for pair in cell.pairs:
        branch_1 = pair.branch_1
        branch_2 = pair.branch_2

        if branch_1.input_state == old_state:
            branch_1 = replace(branch_1, input_state=new_state)
        if branch_2.input_state == old_state:
            branch_2 = replace(branch_2, input_state=new_state)

        pairs.append(
            replace(
                pair,
                branch_1=branch_1,
                branch_2=branch_2,
            )
        )

    return replace(cell, pairs=tuple(pairs))


def test_original_input_states_can_be_cell_outputs():
    """States 0 and 1 are outputs whenever no pair consumes them."""

    rng = random.Random(20260826)
    cell = random_architecture(rng).normal

    # Replacing every reference to state 0 with state 1 keeps all pair inputs
    # acyclic while making state 0 unused.
    without_state_0 = _replace_input_state(
        cell,
        old_state=0,
        new_state=1,
    )
    assert 0 in get_unused_states(without_state_0)

    # Likewise, replacing state 1 with state 0 makes state 1 unused.
    without_state_1 = _replace_input_state(
        cell,
        old_state=1,
        new_state=0,
    )
    assert 1 in get_unused_states(without_state_1)


def test_architecture_is_hashable():
    rng = random.Random(3)
    arch = random_architecture(rng)

    visited = {arch}
    assert arch in visited

    cache = {arch: 0.75}
    assert cache[arch] == 0.75


def test_same_seed_same_architecture_sequence():
    rng1 = random.Random(12345)
    rng2 = random.Random(12345)

    seq1 = [random_architecture(rng1) for _ in range(100)]
    seq2 = [random_architecture(rng2) for _ in range(100)]

    assert seq1 == seq2


def test_different_seeds_change_sequence():
    rng1 = random.Random(1001)
    rng2 = random.Random(1002)

    seq1 = [random_architecture(rng1) for _ in range(20)]
    seq2 = [random_architecture(rng2) for _ in range(20)]

    assert seq1 != seq2


def test_json_round_trip():
    rng = random.Random(20260824)
    arch1 = random_architecture(rng)

    serialized = json.dumps(arch1.to_dict())
    arch2 = NASNetArchitecture.from_dict(json.loads(serialized))

    assert arch1 == arch2
    assert hash(arch1) == hash(arch2)


def test_normal_and_reduction_are_independently_generated():
    rng = random.Random(20260824)
    differences = 0

    for _ in range(100):
        arch = random_architecture(rng)

        if arch.normal != arch.reduction:
            differences += 1

    assert differences > 0
