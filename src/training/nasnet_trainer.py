"""Provisional reduced-budget trainer for CIFAR-10 NASNet evaluation.

This module intentionally excludes DropPath, auxiliary heads, and
``torch.compile``. Search fitness is the validation accuracy at the final
fixed-budget epoch, while best validation accuracy is retained only as
diagnostic metadata. CUDA is synchronized around timed regions so cumulative
epoch times can be used as reliable T1/T5 benchmark measurements.
"""

from __future__ import annotations

import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class TrainingConfig:
    """Fixed-horizon NASNet training configuration."""

    epochs: int
    batch_size: int = 128
    optimizer: str = "SGD"
    learning_rate: float = 0.025
    momentum: float = 0.9
    weight_decay: float = 0.0005
    scheduler: str = "cosine"
    gradient_clip_norm: float = 5.0
    amp: bool = True
    training_seed: int = 20_260_826
    drop_path: bool = False
    auxiliary_head: bool = False
    torch_compile: bool = False

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.optimizer.upper() != "SGD":
            raise ValueError("Part C/D provisional trainer supports only SGD")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.scheduler.lower() != "cosine":
            raise ValueError(
                "Part C/D provisional trainer supports only cosine scheduling"
            )
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.drop_path:
            raise ValueError("DropPath is disabled for Part C/D benchmarks")
        if self.auxiliary_head:
            raise ValueError("auxiliary heads are disabled for Part C/D")
        if self.torch_compile:
            raise ValueError(
                "torch.compile is disabled for Part C/D benchmarks"
            )


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    learning_rate: float
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    epoch_time_seconds: float
    cumulative_time_seconds: float


@dataclass(frozen=True)
class TrainingResult:
    """Training outputs; accuracy values are fractions in the range [0, 1]."""

    final_val_accuracy: float
    best_val_accuracy: float
    final_val_loss: float
    best_epoch: int
    epoch_metrics: tuple[EpochMetrics, ...]
    training_time_seconds: float
    parameter_count: int
    amp_enabled: bool

    @property
    def fitness(self) -> float:
        """Return fixed-horizon search fitness, never best-so-far accuracy."""

        return self.final_val_accuracy


def synchronize_device(device: str | torch.device) -> None:
    """Synchronize CUDA for accurate wall-clock timing; CPU is a no-op."""

    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _seed_training(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise TypeError("each loader batch must contain inputs and targets")
    return batch[0], batch[1]


def _make_grad_scaler(enabled: bool):
    """Create a CUDA GradScaler across current and older PyTorch APIs."""

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_context(enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "autocast"):
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        )
    return torch.cuda.amp.autocast(enabled=True)


def _run_training_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    amp_enabled: bool,
    gradient_clip_norm: float,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in loader:
        inputs, targets = _unpack_batch(batch)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(amp_enabled):
            logits = model(inputs)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=gradient_clip_norm,
        )
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum())
        total_examples += batch_size

    if total_examples == 0:
        raise RuntimeError("training loader produced no examples")

    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def evaluate_nasnet(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool = False,
) -> tuple[float, float]:
    """Evaluate on search validation data and return loss and accuracy."""

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in loader:
        inputs, targets = _unpack_batch(batch)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with _autocast_context(amp_enabled):
            logits = model(inputs)
            loss = criterion(logits, targets)

        batch_size = targets.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum())
        total_examples += batch_size

    if total_examples == 0:
        raise RuntimeError("validation loader produced no examples")

    return total_loss / total_examples, total_correct / total_examples


def train_nasnet(
    model: nn.Module,
    train_loader,
    val_loader,
    config: TrainingConfig,
    device: str | torch.device,
) -> TrainingResult:
    """Train/evaluate one architecture for one continuous fixed epoch budget.

    Timing begins immediately before epoch 1 and ends after the final epoch's
    validation. Model construction and data loading are intentionally excluded.
    The function accepts no official-test loader.
    """

    _seed_training(config.training_seed)
    device = torch.device(device)
    model = model.to(device)

    amp_enabled = bool(config.amp and device.type == "cuda")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
    )
    scaler = _make_grad_scaler(amp_enabled)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if parameter_count <= 0:
        raise ValueError("model has no trainable parameters")

    metrics = []
    best_val_accuracy = float("-inf")
    best_epoch = 0

    synchronize_device(device)
    training_start = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        synchronize_device(device)
        epoch_start = time.perf_counter()
        learning_rate = float(optimizer.param_groups[0]["lr"])

        train_loss, train_accuracy = _run_training_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        val_loss, val_accuracy = evaluate_nasnet(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
        )

        synchronize_device(device)
        epoch_end = time.perf_counter()

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch

        metrics.append(
            EpochMetrics(
                epoch=epoch,
                learning_rate=learning_rate,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                epoch_time_seconds=epoch_end - epoch_start,
                cumulative_time_seconds=epoch_end - training_start,
            )
        )
        scheduler.step()

    final_metrics = metrics[-1]

    return TrainingResult(
        final_val_accuracy=final_metrics.val_accuracy,
        best_val_accuracy=best_val_accuracy,
        final_val_loss=final_metrics.val_loss,
        best_epoch=best_epoch,
        epoch_metrics=tuple(metrics),
        training_time_seconds=final_metrics.cumulative_time_seconds,
        parameter_count=parameter_count,
        amp_enabled=amp_enabled,
    )


__all__ = [
    "EpochMetrics",
    "TrainingConfig",
    "TrainingResult",
    "evaluate_nasnet",
    "synchronize_device",
    "train_nasnet",
]
