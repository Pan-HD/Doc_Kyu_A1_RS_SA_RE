"""Deterministic CPU full-batch training for the RS-SA-RE surrogate."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .multitask_dataset import MultiTaskSurrogateDataset
from .multitask_loss import masked_multitask_mse_loss
from .multitask_model import MultiTaskPrediction, MultiTaskSurrogate


@dataclass(frozen=True)
class MultiTaskTrainingConfig:
    input_dim: int = 280
    hidden_dims: tuple[int, int] = (32, 16)
    optimizer: str = "Adam"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    steps: int = 200
    seed: int = 900000

    def __post_init__(self) -> None:
        hidden_dims = tuple(int(value) for value in self.hidden_dims)
        object.__setattr__(self, "hidden_dims", hidden_dims)
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hidden_dims != MultiTaskSurrogate.hidden_dims:
            raise ValueError("hidden_dims must remain fixed at (32, 16)")
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
class MultiTaskTrainingResult:
    model: MultiTaskSurrogate
    training_loss: float
    mean_training_mse: float
    instability_training_mse: float
    observation_count: int
    paired_count: int
    seed: int


def train_multitask_surrogate(
    dataset: MultiTaskSurrogateDataset,
    config: MultiTaskTrainingConfig,
) -> MultiTaskTrainingResult:
    """Retrain both heads from a fixed initialization using all records."""

    if not isinstance(dataset, MultiTaskSurrogateDataset):
        raise TypeError("dataset must be a MultiTaskSurrogateDataset")
    if not isinstance(config, MultiTaskTrainingConfig):
        raise TypeError("config must be a MultiTaskTrainingConfig")
    if dataset.input_dim != config.input_dim:
        raise ValueError("dataset and surrogate input dimensions differ")
    features, mean_targets, instability_targets, stability_mask = dataset.tensors()

    # This restores the caller's global CPU RNG state on exit. Search RNG is a
    # separate random.Random object and is never passed into this trainer.
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(config.seed)
        model = MultiTaskSurrogate(input_dim=config.input_dim).cpu()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        model.train()
        for _ in range(config.steps):
            optimizer.zero_grad(set_to_none=True)
            loss = masked_multitask_mse_loss(
                model(features),
                mean_targets,
                instability_targets,
                stability_mask,
            )
            if not bool(torch.isfinite(loss.total_loss).item()):
                raise RuntimeError("surrogate training produced non-finite loss")
            loss.total_loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            final_loss = masked_multitask_mse_loss(
                model(features),
                mean_targets,
                instability_targets,
                stability_mask,
            )
            training_loss = float(final_loss.total_loss.item())
            mean_training_mse = float(final_loss.mean_loss.item())
            instability_training_mse = float(
                final_loss.instability_loss.item()
            )

    metrics = (
        training_loss,
        mean_training_mse,
        instability_training_mse,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in metrics):
        raise RuntimeError("multi-task surrogate produced invalid metrics")
    return MultiTaskTrainingResult(
        model=model,
        training_loss=training_loss,
        mean_training_mse=mean_training_mse,
        instability_training_mse=instability_training_mse,
        observation_count=len(dataset),
        paired_count=int(stability_mask.sum().item()),
        seed=config.seed,
    )


def predict_multitask(
    model: MultiTaskSurrogate,
    encodings: Sequence[Any],
) -> MultiTaskPrediction:
    """Predict finite ``mu_hat`` and non-negative ``d_hat`` on CPU."""

    if not isinstance(model, MultiTaskSurrogate):
        raise TypeError("model must be a MultiTaskSurrogate")
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
    if not bool(torch.isfinite(features).all().item()):
        raise ValueError("candidate encoding contains a non-finite value")

    model.eval()
    with torch.inference_mode():
        prediction = model(features)
    if not bool(torch.isfinite(prediction.predicted_mean).all().item()):
        raise RuntimeError("predicted mean contains a non-finite value")
    if not bool(torch.isfinite(prediction.predicted_instability).all().item()):
        raise RuntimeError("predicted instability contains a non-finite value")
    if bool((prediction.predicted_instability < 0.0).any().item()):
        raise RuntimeError("predicted instability must be non-negative")
    return prediction


__all__ = [
    "MultiTaskTrainingConfig",
    "MultiTaskTrainingResult",
    "predict_multitask",
    "train_multitask_surrogate",
]
