"""Paired accuracy labels for the future RS-SA-RE multi-task surrogate."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset


def _validated_accuracy(value: float, field_name: str) -> float:
    try:
        accuracy = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field_name} must be numeric") from error
    if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
        raise ValueError(f"{field_name} must be finite and in [0, 1]")
    return accuracy


@dataclass(frozen=True)
class StabilityRecord:
    """One architecture with one required and one optional real accuracy."""

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
        Its boolean mask is False, so a future masked loss must ignore it.
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
    "MultiTaskSurrogateDataset",
    "MultiTaskSurrogateObservation",
    "StabilityRecord",
]
