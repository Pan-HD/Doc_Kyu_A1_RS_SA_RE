from __future__ import annotations

import pytest
import torch

from src.surrogate.multitask_dataset import (
    MultiTaskSurrogateDataset,
    PairedEvaluationRecord,
)
from src.surrogate.multitask_trainer import (
    MultiTaskTrainingConfig,
    predict_multitask,
    train_multitask_surrogate,
)


def _dataset(*, paired: bool) -> MultiTaskSurrogateDataset:
    dataset = MultiTaskSurrogateDataset(input_dim=4)
    first = PairedEvaluationRecord(
        base_evaluation_index=1,
        architecture="A",
        seed_1=101,
        accuracy_1=0.70,
    )
    if paired:
        first.add_repeat(seed_2=201, accuracy_2=0.76)
    dataset.add_paired_evaluation(record=first, encoding=[1, 0, 0, 0])
    dataset.add_paired_evaluation(
        record=PairedEvaluationRecord(
            base_evaluation_index=2,
            architecture="B",
            seed_1=102,
            accuracy_1=0.72,
        ),
        encoding=[0, 1, 0, 0],
    )
    return dataset


@pytest.mark.parametrize("paired", [False, True])
def test_multitask_training_and_prediction_are_finite(paired: bool) -> None:
    result = train_multitask_surrogate(
        _dataset(paired=paired),
        MultiTaskTrainingConfig(input_dim=4, steps=5, seed=123),
    )

    assert result.observation_count == 2
    assert result.paired_count == int(paired)
    assert result.training_loss >= 0.0
    assert result.mean_training_mse >= 0.0
    assert result.instability_training_mse >= 0.0
    prediction = predict_multitask(
        result.model,
        ([1, 0, 0, 0], [0, 1, 0, 0]),
    )
    assert prediction.predicted_mean.shape == (2,)
    assert prediction.predicted_instability.shape == (2,)
    assert torch.isfinite(prediction.predicted_mean).all()
    assert torch.isfinite(prediction.predicted_instability).all()
    assert torch.all(prediction.predicted_instability >= 0.0)


def test_zero_paired_training_has_finite_zero_instability_loss() -> None:
    result = train_multitask_surrogate(
        _dataset(paired=False),
        MultiTaskTrainingConfig(input_dim=4, steps=2, seed=321),
    )

    assert result.paired_count == 0
    assert result.instability_training_mse == pytest.approx(0.0)
    assert result.training_loss == pytest.approx(result.mean_training_mse)


def test_training_is_reproducible_for_same_surrogate_seed() -> None:
    config = MultiTaskTrainingConfig(input_dim=4, steps=5, seed=456)
    first = train_multitask_surrogate(_dataset(paired=True), config)
    second = train_multitask_surrogate(_dataset(paired=True), config)

    first_prediction = predict_multitask(
        first.model,
        ([1, 0, 0, 0], [0, 1, 0, 0]),
    )
    second_prediction = predict_multitask(
        second.model,
        ([1, 0, 0, 0], [0, 1, 0, 0]),
    )
    assert torch.equal(
        first_prediction.predicted_mean,
        second_prediction.predicted_mean,
    )
    assert torch.equal(
        first_prediction.predicted_instability,
        second_prediction.predicted_instability,
    )
