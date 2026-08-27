"""Deterministic CPU full-batch training for the SA-RE surrogate."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .dataset import SurrogateDataset
from .model import AccuracySurrogate


@dataclass(frozen=True)
class SurrogateTrainingConfig:
    input_dim: int = 280
    hidden_dims: tuple[int, ...] = (32, 16)
    optimizer: str = "Adam"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    steps: int = 200
    seed: int = 900000

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "hidden_dims",
            tuple(int(value) for value in self.hidden_dims),
        )
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not self.hidden_dims or any(value <= 0 for value in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive values")
        if self.optimizer.lower() != "adam":
            raise ValueError("only optimizer=Adam is supported")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True)
class SurrogateTrainingResult:
    model: AccuracySurrogate
    training_mse: float
    observation_count: int
    seed: int


def train_accuracy_surrogate(
    dataset: SurrogateDataset,
    config: SurrogateTrainingConfig,
) -> SurrogateTrainingResult:
    """Retrain from a fixed initialization using all real observations."""

    if dataset.input_dim != config.input_dim:
        raise ValueError("dataset and surrogate input dimensions differ")
    features, targets = dataset.tensors()

    # fork_rng restores the caller's global CPU RNG state on exit. The tiny
    # provisional model deliberately remains on CPU for deterministic debug.
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(config.seed)
        model = AccuracySurrogate(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
        ).cpu()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        loss_function = nn.MSELoss()

        model.train()
        for _ in range(config.steps):
            optimizer.zero_grad(set_to_none=True)
            predictions = model(features)
            loss = loss_function(predictions, targets)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("surrogate training produced non-finite loss")
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            training_mse = float(
                loss_function(model(features), targets).item()
            )

    if not math.isfinite(training_mse):
        raise RuntimeError("surrogate training MSE is non-finite")
    return SurrogateTrainingResult(
        model=model,
        training_mse=training_mse,
        observation_count=len(dataset),
        seed=config.seed,
    )


def predict_accuracy(
    model: AccuracySurrogate,
    encodings: Sequence[Any],
) -> tuple[float, ...]:
    if not encodings:
        raise ValueError("at least one candidate encoding is required")
    tensors = [
        torch.as_tensor(value, dtype=torch.float32, device="cpu")
        .detach()
        .reshape(-1)
        for value in encodings
    ]
    if any(tensor.numel() != model.input_dim for tensor in tensors):
        raise ValueError("candidate encoding dimension does not match model")
    features = torch.stack(tensors)
    model.eval()
    with torch.inference_mode():
        predictions = model(features)
    if not bool(torch.isfinite(predictions).all().item()):
        raise RuntimeError("surrogate prediction contains a non-finite value")
    return tuple(float(value) for value in predictions.tolist())


__all__ = [
    "SurrogateTrainingConfig",
    "SurrogateTrainingResult",
    "predict_accuracy",
    "train_accuracy_surrogate",
]
