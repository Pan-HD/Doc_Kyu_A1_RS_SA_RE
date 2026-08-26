"""Deterministic 280-dimensional encoding for NASNet architectures."""

from __future__ import annotations

import numpy as np

from .genotype import NASNetArchitecture
from .operations import OPS


CELL_ORDER = ("normal", "reduction")
BRANCH_ORDER = ("branch_1", "branch_2")

NUM_CELLS = 2
NUM_PAIRS_PER_CELL = 5
NUM_BRANCHES_PER_PAIR = 2
NUM_BRANCHES = NUM_CELLS * NUM_PAIRS_PER_CELL * NUM_BRANCHES_PER_PAIR

# Branches can reference states 0..5. State 6 is produced by the final pair
# and can be a cell output, but no later branch exists to reference it.
NUM_INPUT_STATES = 6
OP_NAMES = tuple(OPS)
NUM_OPERATIONS = len(OP_NAMES)

if NUM_OPERATIONS != 8:
    raise RuntimeError(
        "NASNet encoding requires exactly eight operations; "
        f"got {NUM_OPERATIONS}"
    )
if len(set(OP_NAMES)) != NUM_OPERATIONS:
    raise RuntimeError("NASNet operation registry contains duplicate names")

OP_TO_INDEX = {
    operation: index
    for index, operation in enumerate(OP_NAMES)
}

BRANCH_ENCODING_DIM = NUM_INPUT_STATES + NUM_OPERATIONS
ARCHITECTURE_ENCODING_DIM = NUM_BRANCHES * BRANCH_ENCODING_DIM


def encode_architecture(
    architecture: NASNetArchitecture,
) -> np.ndarray:
    """Encode Normal then Reduction Cell branches as fixed-order one-hots.

    Each of the 20 branches contributes a 6-D input-state one-hot followed by
    an 8-D operation one-hot. The result has shape ``(280,)`` and exactly 40
    entries equal to one.
    """

    encoding = np.zeros(
        ARCHITECTURE_ENCODING_DIM,
        dtype=np.float32,
    )
    offset = 0

    for cell_name in CELL_ORDER:
        cell = getattr(architecture, cell_name)

        if len(cell.pairs) != NUM_PAIRS_PER_CELL:
            raise ValueError(
                "NASNet encoding requires exactly five pairs per cell; "
                f"{cell_name} has {len(cell.pairs)}"
            )

        for pair_index, pair in enumerate(cell.pairs):
            for branch_name in BRANCH_ORDER:
                branch = getattr(pair, branch_name)
                input_state = branch.input_state

                if not isinstance(input_state, int) or isinstance(
                    input_state,
                    bool,
                ):
                    raise ValueError(
                        f"{cell_name} pair {pair_index} {branch_name} has "
                        f"non-integer input state {input_state!r}"
                    )
                if not 0 <= input_state < NUM_INPUT_STATES:
                    raise ValueError(
                        f"{cell_name} pair {pair_index} {branch_name} input "
                        f"state must be in 0..5; got {input_state}"
                    )
                if input_state > pair_index + 1:
                    raise ValueError(
                        f"{cell_name} pair {pair_index} {branch_name} "
                        f"references unavailable state {input_state}"
                    )

                try:
                    operation_index = OP_TO_INDEX[branch.op]
                except KeyError as error:
                    raise ValueError(
                        f"unknown NASNet operation {branch.op!r}"
                    ) from error

                encoding[offset + input_state] = 1.0
                encoding[
                    offset
                    + NUM_INPUT_STATES
                    + operation_index
                ] = 1.0
                offset += BRANCH_ENCODING_DIM

    if offset != ARCHITECTURE_ENCODING_DIM:
        raise RuntimeError(
            f"encoded {offset} values, expected "
            f"{ARCHITECTURE_ENCODING_DIM}"
        )

    return encoding


__all__ = [
    "ARCHITECTURE_ENCODING_DIM",
    "BRANCH_ENCODING_DIM",
    "BRANCH_ORDER",
    "CELL_ORDER",
    "NUM_BRANCHES",
    "NUM_INPUT_STATES",
    "NUM_OPERATIONS",
    "OP_NAMES",
    "OP_TO_INDEX",
    "encode_architecture",
]
