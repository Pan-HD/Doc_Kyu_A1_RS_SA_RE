"""Concrete two-head surrogate for RS-SA-RE candidate screening."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn.functional as F
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

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Allow ``mu_hat, d_hat = prediction`` without breaking named access."""

        yield self.predicted_mean
        yield self.predicted_instability


class MultiTaskSurrogate(nn.Module):
    """Shared 280→32→16 trunk with mean and non-negative instability heads."""

    hidden_dims = (32, 16)

    def __init__(self, input_dim: int = 280) -> None:
        super().__init__()
        if isinstance(input_dim, bool) or not isinstance(input_dim, int):
            raise TypeError("input_dim must be an integer")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.input_dim = int(input_dim)

        self.trunk = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(self.hidden_dims[1], 1)
        self.instability_head = nn.Linear(self.hidden_dims[1], 1)

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
        if not bool(torch.isfinite(inputs).all().item()):
            raise ValueError("multi-task surrogate input contains a non-finite value")

    def forward(self, inputs: torch.Tensor) -> MultiTaskPrediction:
        self.validate_inputs(inputs)
        shared = self.trunk(inputs)

        # Deliberately unconstrained regression output, matching SA-RE's mean
        # predictor. Do not add a sigmoid here.
        predicted_mean = self.mean_head(shared).squeeze(-1)

        # d(a)=|y1-y2| is non-negative; Softplus enforces d_hat >= 0 while
        # retaining useful gradients.
        instability_raw = self.instability_head(shared).squeeze(-1)
        predicted_instability = F.softplus(instability_raw)

        if not bool(torch.isfinite(predicted_mean).all().item()):
            raise FloatingPointError("predicted mean contains a non-finite value")
        if not bool(torch.isfinite(predicted_instability).all().item()):
            raise FloatingPointError(
                "predicted instability contains a non-finite value"
            )

        return MultiTaskPrediction(
            predicted_mean=predicted_mean,
            predicted_instability=predicted_instability,
        )


__all__ = ["MultiTaskPrediction", "MultiTaskSurrogate"]
