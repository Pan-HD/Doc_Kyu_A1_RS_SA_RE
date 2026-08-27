"""Auditable in-memory observations for accuracy-surrogate training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SurrogateObservation:
    """One architecture whose accuracy came from a real training run."""

    architecture: Any
    encoding: torch.Tensor
    target_accuracy: float
    evaluation_index: int


class SurrogateDataset(Dataset):
    """Store architecture metadata together with full-batch tensors."""

    def __init__(self, input_dim: int = 280) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.input_dim = int(input_dim)
        self._observations: list[SurrogateObservation] = []
        self._evaluation_indices: set[int] = set()

    def add(
        self,
        *,
        architecture: Any,
        encoding: Any,
        target_accuracy: float,
        evaluation_index: int,
    ) -> SurrogateObservation:
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

        target = float(target_accuracy)
        if not math.isfinite(target) or not 0.0 <= target <= 1.0:
            raise ValueError("target_accuracy must be finite and in [0, 1]")
        index = int(evaluation_index)
        if index <= 0:
            raise ValueError("evaluation_index must be positive")
        if index in self._evaluation_indices:
            raise ValueError("evaluation_index is already present")

        observation = SurrogateObservation(
            architecture=architecture,
            encoding=encoding_tensor,
            target_accuracy=target,
            evaluation_index=index,
        )
        self._observations.append(observation)
        self._evaluation_indices.add(index)
        return observation

    @property
    def observations(self) -> tuple[SurrogateObservation, ...]:
        return tuple(self._observations)

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._observations:
            raise ValueError("surrogate dataset is empty")
        features = torch.stack(
            [observation.encoding for observation in self._observations]
        )
        targets = torch.tensor(
            [
                observation.target_accuracy
                for observation in self._observations
            ],
            dtype=torch.float32,
            device="cpu",
        )
        return features, targets

    def __len__(self) -> int:
        return len(self._observations)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        observation = self._observations[index]
        target = torch.tensor(observation.target_accuracy, dtype=torch.float32)
        return observation.encoding.clone(), target


__all__ = ["SurrogateDataset", "SurrogateObservation"]
