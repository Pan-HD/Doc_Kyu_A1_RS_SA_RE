"""Tests for NASNet hidden-state, operation, and identity mutations."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from src.nasnet.genotype import random_architecture, validate_architecture
from src.nasnet.operations import OPS
from src.nasnet.mutation import (
    HIDDEN_STATE_MUTATION,
    HIDDEN_STATE_MUTATION_PROBABILITY,
    IDENTITY_MUTATION,
    IDENTITY_MUTATION_PROBABILITY,
    MUTATION_TYPES,
    OPERATION_MUTATION,
    OPERATION_MUTATION_PROBABILITY,
    mutate_architecture,
)


SEED = 20260826
MUTATION_CASES = 1000


class ControlledMutationRNG:
    """Fix the mutation-type draw while randomizing its target normally."""

    def __init__(self, mutation_draw: float, seed: int = SEED) -> None:
        self.mutation_draw = mutation_draw
        self._rng = random.Random(seed)

    def random(self) -> float:
        return self.mutation_draw

    def choice(self, sequence):
        return self._rng.choice(sequence)

    def randrange(self, *args):
        return self._rng.randrange(*args)


def _branch_records(architecture):
    records = {}

    for cell_name in ("normal", "reduction"):
        cell = getattr(architecture, cell_name)

        for pair_index, pair in enumerate(cell.pairs):
            for branch_name in ("branch_1", "branch_2"):
                branch = getattr(pair, branch_name)
                records[(cell_name, pair_index, branch_name)] = (
                    branch.input_state,
                    branch.op,
                )

    return records


def _changed_locations(parent, child):
    parent_records = _branch_records(parent)
    child_records = _branch_records(child)

    return [
        location
        for location in parent_records
        if parent_records[location] != child_records[location]
    ]


def test_mutation_probabilities_sum_to_one() -> None:
    assert IDENTITY_MUTATION_PROBABILITY == pytest.approx(0.05)
    assert HIDDEN_STATE_MUTATION_PROBABILITY == pytest.approx(0.475)
    assert OPERATION_MUTATION_PROBABILITY == pytest.approx(0.475)
    assert (
        IDENTITY_MUTATION_PROBABILITY
        + HIDDEN_STATE_MUTATION_PROBABILITY
        + OPERATION_MUTATION_PROBABILITY
    ) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("mutation_draw", "expected_type"),
    (
        (0.0, IDENTITY_MUTATION),
        (0.049999, IDENTITY_MUTATION),
        (0.05, HIDDEN_STATE_MUTATION),
        (0.524999, HIDDEN_STATE_MUTATION),
        (0.525, OPERATION_MUTATION),
        (0.999999, OPERATION_MUTATION),
    ),
)
def test_mutation_probability_boundaries(
    mutation_draw: float,
    expected_type: str,
) -> None:
    parent = random_architecture(random.Random(SEED))
    result = mutate_architecture(
        parent,
        ControlledMutationRNG(mutation_draw),
    )

    assert result.mutation_type == expected_type
    assert validate_architecture(result.architecture)


def test_identity_mutation_returns_unchanged_parent() -> None:
    parent = random_architecture(random.Random(SEED))
    result = mutate_architecture(
        parent,
        ControlledMutationRNG(0.0),
    )

    assert result.mutation_type == IDENTITY_MUTATION
    assert result.architecture == parent
    assert result.architecture is parent


def test_one_thousand_random_mutations() -> None:
    architecture_rng = random.Random(SEED)
    mutation_rng = random.Random(SEED + 1)
    mutation_counts = Counter()

    for _ in range(MUTATION_CASES):
        parent = random_architecture(architecture_rng)
        result = mutate_architecture(parent, mutation_rng)
        child = result.architecture
        mutation_counts[result.mutation_type] += 1

        assert result.mutation_type in MUTATION_TYPES
        assert validate_architecture(child)

        changed_locations = _changed_locations(parent, child)

        if result.mutation_type == IDENTITY_MUTATION:
            assert child == parent
            assert changed_locations == []
            continue

        assert child != parent
        assert len(changed_locations) == 1

        location = changed_locations[0]
        _, pair_index, _ = location
        parent_state, parent_op = _branch_records(parent)[location]
        child_state, child_op = _branch_records(child)[location]

        if result.mutation_type == HIDDEN_STATE_MUTATION:
            assert child_state != parent_state
            assert child_op == parent_op
            assert 0 <= child_state <= pair_index + 1
        elif result.mutation_type == OPERATION_MUTATION:
            assert child_state == parent_state
            assert child_op != parent_op
            assert child_op in OPS
        else:  # pragma: no cover - guarded by MUTATION_TYPES assertion above.
            raise AssertionError(
                f"unexpected mutation type {result.mutation_type!r}"
            )

    # The deterministic run should exercise all three explicit branches.
    assert all(mutation_counts[name] > 0 for name in MUTATION_TYPES)


def test_same_seed_produces_same_mutation_sequence() -> None:
    parent = random_architecture(random.Random(SEED))
    rng_1 = random.Random(12345)
    rng_2 = random.Random(12345)

    sequence_1 = [
        mutate_architecture(parent, rng_1)
        for _ in range(100)
    ]
    sequence_2 = [
        mutate_architecture(parent, rng_2)
        for _ in range(100)
    ]

    assert sequence_1 == sequence_2
