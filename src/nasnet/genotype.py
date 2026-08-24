from dataclasses import dataclass
from typing import Tuple
import random

from .operations import OPS


@dataclass(frozen=True)
class BranchGene:
    input_state: int
    op: str

    def to_dict(self):
        return {
            "input_state": self.input_state,
            "op": self.op,
        }

    @staticmethod
    def from_dict(data):
        return BranchGene(
            input_state=data["input_state"],
            op=data["op"],
        )


@dataclass(frozen=True)
class PairGene:
    branch_1: BranchGene
    branch_2: BranchGene

    def to_dict(self):
        return {
            "branch_1":
                self.branch_1.to_dict(),
            "branch_2":
                self.branch_2.to_dict(),
        }

    @staticmethod
    def from_dict(data):
        return PairGene(
            branch_1=BranchGene.from_dict(
                data["branch_1"]
            ),
            branch_2=BranchGene.from_dict(
                data["branch_2"]
            ),
        )


@dataclass(frozen=True)
class CellGene:
    pairs: Tuple[PairGene, ...]

    def to_dict(self):
        return {
            "pairs": [
                pair.to_dict()
                for pair in self.pairs
            ]
        }

    @staticmethod
    def from_dict(data):
        return CellGene(
            pairs=tuple(
                PairGene.from_dict(pair)
                for pair in data["pairs"]
            )
        )


@dataclass(frozen=True)
class NASNetArchitecture:
    normal: CellGene
    reduction: CellGene

    def to_dict(self):
        return {
            "normal":
                self.normal.to_dict(),
            "reduction":
                self.reduction.to_dict(),
        }

    @staticmethod
    def from_dict(data):
        return NASNetArchitecture(
            normal=CellGene.from_dict(
                data["normal"]
            ),
            reduction=CellGene.from_dict(
                data["reduction"]
            ),
        )


def validate_cell(
    cell: CellGene
) -> bool:

    if len(cell.pairs) != 5:
        return False

    for pair_index, pair in enumerate(
        cell.pairs
    ):

        max_state = pair_index + 1

        for branch in (
            pair.branch_1,
            pair.branch_2,
        ):

            if not (
                0
                <= branch.input_state
                <= max_state
            ):
                return False

            if branch.op not in OPS:
                return False

    return True


def validate_architecture(
    arch: NASNetArchitecture
) -> bool:

    return (
        validate_cell(arch.normal)
        and
        validate_cell(arch.reduction)
    )


def random_branch(
    valid_states,
    rng: random.Random
):

    return BranchGene(
        input_state=rng.choice(
            valid_states
        ),
        op=rng.choice(OPS),
    )


def random_cell(
    rng: random.Random
):

    pairs = []

    for pair_index in range(5):

        valid_states = list(
            range(pair_index + 2)
        )

        pairs.append(
            PairGene(
                branch_1=random_branch(
                    valid_states,
                    rng
                ),
                branch_2=random_branch(
                    valid_states,
                    rng
                ),
            )
        )

    cell = CellGene(
        pairs=tuple(pairs)
    )

    assert validate_cell(cell)

    return cell


def random_architecture(
    rng: random.Random
):

    arch = NASNetArchitecture(
        normal=random_cell(rng),
        reduction=random_cell(rng),
    )

    assert validate_architecture(arch)

    return arch

def get_unused_states(
    cell: CellGene
):

    generated_states = set(
        range(2, 7)
    )

    used_states = set()

    for pair in cell.pairs:

        used_states.add(
            pair.branch_1.input_state
        )

        used_states.add(
            pair.branch_2.input_state
        )

    unused = sorted(
        generated_states
        -
        used_states
    )

    return tuple(unused)

def test_cell_has_output_states():

    rng = random.Random(
        20260824
    )

    for _ in range(1000):

        arch = random_architecture(
            rng
        )

        for cell in (
            arch.normal,
            arch.reduction,
        ):

            unused = (
                get_unused_states(
                    cell
                )
            )

            assert len(unused) >= 1

            assert all(
                state in range(2, 7)
                for state in unused
            )