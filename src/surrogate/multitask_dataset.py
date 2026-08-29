"""Paired accuracy labels for the RS-SA-RE multi-task surrogate.

This module preserves the August 28 scaffold API while adding evaluation-level
paired records for deterministic second-seed retraining. Search-stage
instability is the ``two-seed retraining-instability proxy``::

    d(a) = abs(y_1 - y_2)

It is not a variance, standard deviation, or claim of true stability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset


INSTABILITY_PROXY_NAME = "two-seed retraining-instability proxy"


def _validated_accuracy(value: float, field_name: str) -> float:
    try:
        accuracy = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field_name} must be numeric") from error
    if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
        raise ValueError(f"{field_name} must be finite and in [0, 1]")
    return accuracy


def _validated_index(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("base_evaluation_index must be an integer")
    index = int(value)
    if index < 0:
        raise ValueError("base_evaluation_index must be non-negative")
    return index


def _validated_seed(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be an integer")
    return int(value)


@dataclass(frozen=True)
class StabilityRecord:
    """Legacy-compatible labels for one architecture."""

    architecture: Any
    accuracy_seed_1: float
    accuracy_seed_2: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "accuracy_seed_1",
            _validated_accuracy(self.accuracy_seed_1, "accuracy_seed_1"),
        )
        if self.accuracy_seed_2 is not None:
            object.__setattr__(
                self,
                "accuracy_seed_2",
                _validated_accuracy(self.accuracy_seed_2, "accuracy_seed_2"),
            )

    @property
    def has_pair(self) -> bool:
        return self.accuracy_seed_2 is not None

    @property
    def mean_target_available(self) -> bool:
        return True

    @property
    def instability_target_available(self) -> bool:
        return self.has_pair

    @property
    def mean_target(self) -> float:
        if self.accuracy_seed_2 is None:
            return self.accuracy_seed_1
        return (self.accuracy_seed_1 + self.accuracy_seed_2) / 2.0

    @property
    def instability_target(self) -> float | None:
        if self.accuracy_seed_2 is None:
            return None
        return abs(self.accuracy_seed_1 - self.accuracy_seed_2)


@dataclass(slots=True)
class PairedEvaluationRecord:
    """One first evaluation and, optionally, its scheduled repeat.

    Identity is ``base_evaluation_index`` rather than architecture so natural
    duplicate architectures remain distinct evaluation records.
    """

    base_evaluation_index: int
    architecture: Any
    seed_1: int
    accuracy_1: float
    seed_2: int | None = None
    accuracy_2: float | None = None

    def __post_init__(self) -> None:
        self.base_evaluation_index = _validated_index(
            self.base_evaluation_index
        )
        self.seed_1 = _validated_seed(self.seed_1, "seed_1")
        self.accuracy_1 = _validated_accuracy(self.accuracy_1, "accuracy_1")

        seed_2_is_set = self.seed_2 is not None
        accuracy_2_is_set = self.accuracy_2 is not None
        if seed_2_is_set != accuracy_2_is_set:
            raise ValueError(
                "seed_2 and accuracy_2 must either both be set or both be None"
            )
        if seed_2_is_set:
            assert self.seed_2 is not None
            assert self.accuracy_2 is not None
            self.seed_2 = _validated_seed(self.seed_2, "seed_2")
            self.accuracy_2 = _validated_accuracy(
                self.accuracy_2,
                "accuracy_2",
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
        checked_seed = _validated_seed(seed_2, "seed_2")
        if checked_seed == self.seed_1:
            raise ValueError("seed_2 must differ from seed_1")
        checked_accuracy = _validated_accuracy(accuracy_2, "accuracy_2")
        self.seed_2 = checked_seed
        self.accuracy_2 = checked_accuracy

    def to_stability_record(self) -> StabilityRecord:
        """Convert to the August 28 label interface used by the dataset."""

        return StabilityRecord(
            architecture=self.architecture,
            accuracy_seed_1=self.accuracy_1,
            accuracy_seed_2=self.accuracy_2,
        )


class PairedEvaluationStore:
    """Registry keyed only by base evaluation identity, never architecture."""

    def __init__(self) -> None:
        self._records: dict[int, PairedEvaluationRecord] = {}

    def add(self, record: PairedEvaluationRecord) -> None:
        if not isinstance(record, PairedEvaluationRecord):
            raise TypeError("record must be a PairedEvaluationRecord")
        index = record.base_evaluation_index
        if index in self._records:
            raise ValueError(f"duplicate base_evaluation_index: {index}")
        self._records[index] = record

    def get(self, base_evaluation_index: int) -> PairedEvaluationRecord:
        return self._records[_validated_index(base_evaluation_index)]

    def add_repeat(
        self,
        *,
        base_evaluation_index: int,
        seed_2: int,
        accuracy_2: float,
    ) -> PairedEvaluationRecord:
        record = self.get(base_evaluation_index)
        record.add_repeat(seed_2=seed_2, accuracy_2=accuracy_2)
        return record

    def unpaired_records(self) -> tuple[PairedEvaluationRecord, ...]:
        return tuple(record for record in self if not record.has_pair)

    def __contains__(self, base_evaluation_index: object) -> bool:
        return base_evaluation_index in self._records

    def __iter__(self) -> Iterator[PairedEvaluationRecord]:
        return iter(self._records.values())

    def __len__(self) -> int:
        return len(self._records)


@dataclass(frozen=True)
class MultiTaskSurrogateObservation:
    """One encoded architecture and its available multi-task labels."""

    record: StabilityRecord
    encoding: torch.Tensor


class MultiTaskSurrogateDataset(Dataset):
    """Expose mean labels plus masked instability labels as CPU tensors."""

    def __init__(self, input_dim: int = 280) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.input_dim = int(input_dim)
        self._observations: list[MultiTaskSurrogateObservation] = []

    def add(
        self,
        *,
        record: StabilityRecord,
        encoding: Any,
    ) -> MultiTaskSurrogateObservation:
        if not isinstance(record, StabilityRecord):
            raise TypeError("record must be a StabilityRecord")
        encoding_tensor = torch.as_tensor(
            encoding,
            dtype=torch.float32,
            device="cpu",
        ).detach().clone().reshape(-1)
        if encoding_tensor.numel() != self.input_dim:
            raise ValueError(
                f"encoding must contain {self.input_dim} values; "
                f"got {encoding_tensor.numel()}"
            )
        if not bool(torch.isfinite(encoding_tensor).all().item()):
            raise ValueError("encoding contains a non-finite value")
        observation = MultiTaskSurrogateObservation(
            record=record,
            encoding=encoding_tensor,
        )
        self._observations.append(observation)
        return observation

    def add_paired_evaluation(
        self,
        *,
        record: PairedEvaluationRecord,
        encoding: Any,
    ) -> MultiTaskSurrogateObservation:
        """Add an evaluation-level record through the compatible dataset API."""

        if not isinstance(record, PairedEvaluationRecord):
            raise TypeError("record must be a PairedEvaluationRecord")
        return self.add(record=record.to_stability_record(), encoding=encoding)

    @property
    def observations(self) -> tuple[MultiTaskSurrogateObservation, ...]:
        return tuple(self._observations)

    @property
    def records(self) -> tuple[StabilityRecord, ...]:
        return tuple(observation.record for observation in self._observations)

    def tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return features, mean targets, instability targets, and mask.

        A single-seed observation uses 0.0 only as an instability placeholder.
        Its boolean mask is False, so a masked loss must ignore it.
        """

        if not self._observations:
            raise ValueError("multi-task surrogate dataset is empty")
        features = torch.stack(
            [observation.encoding for observation in self._observations]
        )
        mean_targets = torch.tensor(
            [observation.record.mean_target for observation in self._observations],
            dtype=torch.float32,
            device="cpu",
        )
        instability_targets = torch.tensor(
            [
                observation.record.instability_target
                if observation.record.instability_target is not None
                else 0.0
                for observation in self._observations
            ],
            dtype=torch.float32,
            device="cpu",
        )
        instability_mask = torch.tensor(
            [
                observation.record.instability_target_available
                for observation in self._observations
            ],
            dtype=torch.bool,
            device="cpu",
        )
        return features, mean_targets, instability_targets, instability_mask

    def __len__(self) -> int:
        return len(self._observations)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        observation = self._observations[index]
        instability = observation.record.instability_target
        return (
            observation.encoding.clone(),
            torch.tensor(observation.record.mean_target, dtype=torch.float32),
            torch.tensor(
                instability if instability is not None else 0.0,
                dtype=torch.float32,
            ),
            torch.tensor(
                observation.record.instability_target_available,
                dtype=torch.bool,
            ),
        )


__all__ = [
    "INSTABILITY_PROXY_NAME",
    "MultiTaskSurrogateDataset",
    "MultiTaskSurrogateObservation",
    "PairedEvaluationRecord",
    "PairedEvaluationStore",
    "StabilityRecord",
]
