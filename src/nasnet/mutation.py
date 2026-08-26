"""NASNet architecture mutations used by regularized evolution.

Each call applies exactly one mutation event:

* identity mutation with probability 0.05;
* hidden-state mutation with probability 0.475; or
* operation mutation with probability 0.475.

An identity mutation is deliberately different from the searchable
``identity`` branch operation. It returns an unchanged architecture, but the
caller must still train/evaluate the child and consume one search evaluation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from .genotype import NASNetArchitecture, validate_architecture
from .operations import OPS


IDENTITY_MUTATION_PROBABILITY = 0.05
HIDDEN_STATE_MUTATION_PROBABILITY = 0.475
OPERATION_MUTATION_PROBABILITY = 0.475

IDENTITY_MUTATION = "identity"
HIDDEN_STATE_MUTATION = "hidden_state"
OPERATION_MUTATION = "operation"
MUTATION_TYPES = (
    IDENTITY_MUTATION,
    HIDDEN_STATE_MUTATION,
    OPERATION_MUTATION,
)

_CELL_NAMES = ("normal", "reduction")
_BRANCH_NAMES = ("branch_1", "branch_2")


@dataclass(frozen=True)
class MutationResult:
    """An architecture produced by one evolution mutation event."""

    architecture: NASNetArchitecture
    mutation_type: str


def _select_branch(parent: NASNetArchitecture, rng: random.Random):
    """Select cell, pair, and branch in the paper-specified order."""

    cell_name = rng.choice(_CELL_NAMES)
    cell = getattr(parent, cell_name)

    if len(cell.pairs) != 5:
        raise ValueError(
            "NASNet mutation requires exactly five pairs per cell; "
            f"{cell_name} has {len(cell.pairs)}"
        )

    pair_index = rng.randrange(len(cell.pairs))
    branch_name = rng.choice(_BRANCH_NAMES)
    branch = getattr(cell.pairs[pair_index], branch_name)

    return cell_name, pair_index, branch_name, branch


def _replace_branch(
    parent: NASNetArchitecture,
    cell_name: str,
    pair_index: int,
    branch_name: str,
    **branch_changes,
) -> NASNetArchitecture:
    """Replace one branch while preserving all unrelated frozen fields."""

    cell = getattr(parent, cell_name)
    pair = cell.pairs[pair_index]
    branch = getattr(pair, branch_name)

    new_branch = replace(branch, **branch_changes)
    new_pair = replace(pair, **{branch_name: new_branch})

    new_pairs = list(cell.pairs)
    new_pairs[pair_index] = new_pair
    new_cell = replace(cell, pairs=tuple(new_pairs))

    return replace(parent, **{cell_name: new_cell})


def _mutate_hidden_state(
    parent: NASNetArchitecture,
    rng: random.Random,
) -> NASNetArchitecture:
    cell_name, pair_index, branch_name, branch = _select_branch(parent, rng)

    # Pair i can read states 0..i+1. Excluding the current state guarantees
    # that hidden-state mutation changes exactly one stored genotype field.
    candidate_states = [
        state
        for state in range(pair_index + 2)
        if state != branch.input_state
    ]
    if not candidate_states:
        raise RuntimeError(
            f"no hidden-state mutation candidate for pair {pair_index}"
        )

    new_state = rng.choice(candidate_states)
    return _replace_branch(
        parent,
        cell_name,
        pair_index,
        branch_name,
        input_state=new_state,
    )


def _mutate_operation(
    parent: NASNetArchitecture,
    rng: random.Random,
) -> NASNetArchitecture:
    cell_name, pair_index, branch_name, branch = _select_branch(parent, rng)

    # Excluding the current operation keeps explicit identity mutation as the
    # only no-change mutation type.
    candidate_ops = [
        operation
        for operation in OPS
        if operation != branch.op
    ]
    if not candidate_ops:
        raise RuntimeError("operation registry has no alternative operation")

    new_operation = rng.choice(candidate_ops)
    return _replace_branch(
        parent,
        cell_name,
        pair_index,
        branch_name,
        op=new_operation,
    )


def mutate_architecture(
    parent: NASNetArchitecture,
    rng: random.Random,
) -> MutationResult:
    """Apply exactly one NASNet mutation and return its metadata.

    Identity mutation returns ``parent`` unchanged. The regularized-evolution
    loop must not use architecture equality to skip its training/evaluation.
    """

    if not validate_architecture(parent):
        raise ValueError("cannot mutate an invalid NASNet architecture")

    mutation_draw = rng.random()

    if mutation_draw < IDENTITY_MUTATION_PROBABILITY:
        return MutationResult(
            architecture=parent,
            mutation_type=IDENTITY_MUTATION,
        )

    hidden_state_upper_bound = (
        IDENTITY_MUTATION_PROBABILITY
        + HIDDEN_STATE_MUTATION_PROBABILITY
    )

    if mutation_draw < hidden_state_upper_bound:
        child = _mutate_hidden_state(parent, rng)
        mutation_type = HIDDEN_STATE_MUTATION
    else:
        child = _mutate_operation(parent, rng)
        mutation_type = OPERATION_MUTATION

    if child == parent:
        raise RuntimeError(
            f"{mutation_type} mutation unexpectedly left architecture unchanged"
        )
    if not validate_architecture(child):
        raise RuntimeError(f"{mutation_type} mutation produced an invalid child")

    return MutationResult(
        architecture=child,
        mutation_type=mutation_type,
    )


__all__ = [
    "HIDDEN_STATE_MUTATION",
    "HIDDEN_STATE_MUTATION_PROBABILITY",
    "IDENTITY_MUTATION",
    "IDENTITY_MUTATION_PROBABILITY",
    "MUTATION_TYPES",
    "MutationResult",
    "OPERATION_MUTATION",
    "OPERATION_MUTATION_PROBABILITY",
    "mutate_architecture",
]
