"""CPU-only structural tests for the provisional NASNet trainer."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.nasnet_trainer import (
    TrainingConfig,
    TrainingResult,
    train_nasnet,
)


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(3 * 8 * 8, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return logits; CrossEntropyLoss applies the required normalization.
        return self.classifier(torch.flatten(x, 1))


def _build_tiny_loaders() -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(20260826)
    train_inputs = torch.randn(24, 3, 8, 8, generator=generator)
    train_targets = torch.randint(0, 10, (24,), generator=generator)
    val_inputs = torch.randn(12, 3, 8, 8, generator=generator)
    val_targets = torch.randint(0, 10, (12,), generator=generator)

    train_loader = DataLoader(
        TensorDataset(train_inputs, train_targets),
        batch_size=8,
        shuffle=False,
    )
    val_loader = DataLoader(
        TensorDataset(val_inputs, val_targets),
        batch_size=6,
        shuffle=False,
    )
    return train_loader, val_loader


def test_training_result_fitness_is_final_accuracy() -> None:
    result = TrainingResult(
        final_val_accuracy=0.40,
        best_val_accuracy=0.90,
        final_val_loss=1.0,
        best_epoch=1,
        epoch_metrics=(),
        training_time_seconds=2.0,
        parameter_count=100,
        amp_enabled=False,
    )

    assert result.fitness == 0.40
    assert result.fitness == result.final_val_accuracy
    assert result.fitness != result.best_val_accuracy


def test_two_epoch_cpu_training_returns_required_metrics(monkeypatch) -> None:
    train_loader, val_loader = _build_tiny_loaders()
    model = TinyClassifier()
    config = TrainingConfig(
        epochs=2,
        batch_size=8,
        learning_rate=0.025,
        momentum=0.9,
        weight_decay=0.0005,
        gradient_clip_norm=5.0,
        amp=True,
        training_seed=20260826,
    )

    original_clip = torch.nn.utils.clip_grad_norm_
    clip_calls = []

    def recording_clip(parameters, max_norm, *args, **kwargs):
        clip_calls.append(max_norm)
        return original_clip(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        recording_clip,
    )

    result = train_nasnet(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device="cpu",
    )

    assert len(result.epoch_metrics) == 2
    assert result.parameter_count > 0
    assert result.training_time_seconds > 0
    assert result.amp_enabled is False
    assert result.fitness == result.final_val_accuracy
    assert result.best_val_accuracy >= result.final_val_accuracy
    assert result.best_epoch in (1, 2)
    assert 0.0 <= result.final_val_accuracy <= 1.0
    assert result.final_val_loss >= 0.0
    assert result.epoch_metrics[1].learning_rate < (
        result.epoch_metrics[0].learning_rate
    )
    assert clip_calls
    assert all(value == 5.0 for value in clip_calls)


@pytest.mark.parametrize(
    "disabled_feature",
    ("drop_path", "auxiliary_head", "torch_compile"),
)
def test_complex_training_features_remain_disabled(
    disabled_feature: str,
) -> None:
    with pytest.raises(ValueError, match="disabled"):
        TrainingConfig(
            epochs=1,
            **{disabled_feature: True},
        )


def test_provisional_optimizer_and_scheduler_are_fixed() -> None:
    with pytest.raises(ValueError, match="only SGD"):
        TrainingConfig(epochs=1, optimizer="Adam")

    with pytest.raises(ValueError, match="only cosine"):
        TrainingConfig(epochs=1, scheduler="step")
