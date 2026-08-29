"""Paired labels used by the RS-SA-RE multi-task surrogate.

Search-stage instability is intentionally defined as the
``two-seed retraining-instability proxy``::

    d(a) = abs(y_1 - y_2)

It is not a variance, standard deviation, or claim of true stability.
Records are identified by ``base_evaluation_index`` rather than architecture:
regularized evolution may naturally evaluate the same architecture more than
once, and those evaluations must remain independent records.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Generic, Iterator, TypeVar


ArchitectureT = TypeVar("ArchitectureT")

INSTABILITY_PROXY_NAME = "two-seed retraining-instability proxy"


def _validated_index(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("base_evaluation_index must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError("base_evaluation_index must be non-negative")
    return value


def _validated_seed(value: int, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be an integer")
    return int(value)


def _validated_accuracy(value: float, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number")
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]")
    return value


@dataclass(slots=True)
class PairedEvaluationRecord(Generic[ArchitectureT]):
    """One first evaluation and, optionally, its scheduled second-seed repeat."""

    base_evaluation_index: int
    architecture: ArchitectureT
    seed_1: int
    accuracy_1: float
    seed_2: int | None = None
    accuracy_2: float | None = None

    def __post_init__(self) -> None:
        self.base_evaluation_index = _validated_index(
            self.base_evaluation_index
        )
        self.seed_1 = _validated_seed(self.seed_1, field_name="seed_1")
        self.accuracy_1 = _validated_accuracy(
            self.accuracy_1,
            field_name="accuracy_1",
        )

        seed_2_is_set = self.seed_2 is not None
        accuracy_2_is_set = self.accuracy_2 is not None
        if seed_2_is_set != accuracy_2_is_set:
            raise ValueError(
                "seed_2 and accuracy_2 must either both be set or both be None"
            )

        if seed_2_is_set:
            assert self.seed_2 is not None
            assert self.accuracy_2 is not None
            self.seed_2 = _validated_seed(self.seed_2, field_name="seed_2")
            self.accuracy_2 = _validated_accuracy(
                self.accuracy_2,
                field_name="accuracy_2",
            )
            if self.seed_2 == self.seed_1:
                raise ValueError("seed_2 must differ from seed_1")

    @property
    def has_pair(self) -> bool:
        return self.seed_2 is not None and self.accuracy_2 is not None

    @property
    def mean_target(self) -> float:
        if not self.has_pair:
            return self.accuracy_1
        assert self.accuracy_2 is not None
        return (self.accuracy_1 + self.accuracy_2) / 2.0

    @property
    def instability_target(self) -> float | None:
        if not self.has_pair:
            return None
        assert self.accuracy_2 is not None
        return abs(self.accuracy_1 - self.accuracy_2)

    def add_repeat(self, *, seed_2: int, accuracy_2: float) -> None:
        """Attach exactly one scheduled repeat to this evaluation record."""

        if self.has_pair:
            raise ValueError(
                "this evaluation record already has a scheduled repeat"
            )

        checked_seed = _validated_seed(seed_2, field_name="seed_2")
        if checked_seed == self.seed_1:
            raise ValueError("seed_2 must differ from seed_1")
        checked_accuracy = _validated_accuracy(
            accuracy_2,
            field_name="accuracy_2",
        )

        self.seed_2 = checked_seed
        self.accuracy_2 = checked_accuracy


class PairedEvaluationStore(Generic[ArchitectureT]):
    """Registry keyed only by base evaluation identity, never architecture."""

    def __init__(self) -> None:
        self._records: dict[int, PairedEvaluationRecord[ArchitectureT]] = {}

    def add(self, record: PairedEvaluationRecord[ArchitectureT]) -> None:
        index = record.base_evaluation_index
        if index in self._records:
            raise ValueError(f"duplicate base_evaluation_index: {index}")
        self._records[index] = record

    def get(self, base_evaluation_index: int) -> PairedEvaluationRecord[ArchitectureT]:
        return self._records[_validated_index(base_evaluation_index)]

    def add_repeat(
        self,
        *,
        base_evaluation_index: int,
        seed_2: int,
        accuracy_2: float,
    ) -> PairedEvaluationRecord[ArchitectureT]:
        record = self.get(base_evaluation_index)
        record.add_repeat(seed_2=seed_2, accuracy_2=accuracy_2)
        return record

    def unpaired_records(self) -> tuple[PairedEvaluationRecord[ArchitectureT], ...]:
        return tuple(record for record in self if not record.has_pair)

    def __contains__(self, base_evaluation_index: object) -> bool:
        return base_evaluation_index in self._records

    def __iter__(self) -> Iterator[PairedEvaluationRecord[ArchitectureT]]:
        return iter(self._records.values())

    def __len__(self) -> int:
        return len(self._records)

