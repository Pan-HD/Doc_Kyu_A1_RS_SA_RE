"""Small deterministic accuracy surrogate used by SA-RE."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class AccuracySurrogate(nn.Module):
    """MLP mapping architecture encodings to predicted validation accuracy."""

    def __init__(
        self,
        input_dim: int = 280,
        hidden_dims: Sequence[int] = (32, 16),
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        normalized_hidden_dims = tuple(int(value) for value in hidden_dims)
        if not normalized_hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")
        if any(value <= 0 for value in normalized_hidden_dims):
            raise ValueError("all hidden dimensions must be positive")

        self.input_dim = int(input_dim)
        self.hidden_dims = normalized_hidden_dims

        dimensions = (self.input_dim, *self.hidden_dims, 1)
        layers: list[nn.Module] = []
        for index, (in_features, out_features) in enumerate(
            zip(dimensions[:-1], dimensions[1:])
        ):
            layers.append(nn.Linear(in_features, out_features))
            if index < len(dimensions) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2:
            raise ValueError("surrogate input must have shape [batch, input_dim]")
        if inputs.shape[1] != self.input_dim:
            raise ValueError(
                f"surrogate expected input_dim={self.input_dim}, "
                f"got {inputs.shape[1]}"
            )
        return self.net(inputs).squeeze(-1)


__all__ = ["AccuracySurrogate"]
