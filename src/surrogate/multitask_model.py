"""Model contract for a future RS-SA-RE multi-task surrogate."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MultiTaskPrediction:
    """Predicted mean accuracy and instability for the same batch."""

    predicted_mean: torch.Tensor
    predicted_instability: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.predicted_mean, torch.Tensor):
            raise TypeError("predicted_mean must be a torch.Tensor")
        if not isinstance(self.predicted_instability, torch.Tensor):
            raise TypeError("predicted_instability must be a torch.Tensor")
        if self.predicted_mean.shape != self.predicted_instability.shape:
            raise ValueError("multi-task prediction tensors must have equal shapes")


class MultiTaskSurrogate(nn.Module, ABC):
    """Abstract two-output model; architecture and loss remain undecided."""

    def __init__(self, input_dim: int = 280) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.input_dim = int(input_dim)

    def validate_inputs(self, inputs: torch.Tensor) -> None:
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("multi-task surrogate input must be a torch.Tensor")
        if inputs.ndim != 2:
            raise ValueError("multi-task surrogate input must have shape [batch, input_dim]")
        if inputs.shape[1] != self.input_dim:
            raise ValueError(
                f"multi-task surrogate expected input_dim={self.input_dim}, "
                f"got {inputs.shape[1]}"
            )

    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> MultiTaskPrediction:
        """Return both task predictions without defining a combined loss."""


__all__ = ["MultiTaskPrediction", "MultiTaskSurrogate"]
