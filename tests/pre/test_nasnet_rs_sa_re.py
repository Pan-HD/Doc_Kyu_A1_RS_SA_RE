from __future__ import annotations

import csv
import json
from dataclasses import dataclass

import pytest
import torch

from src.evolution.repeat_policy import (
    RepeatPolicyConfig,
    derive_training_seed,
)
from src.search.nasnet_re import NASNetTrainingEvaluator
from src.search.nasnet_rs_sa_re import run_nasnet_rs_sa_re
from src.search.regularized_evolution import EvaluationOutcome
from src.surrogate.multitask_model import MultiTaskPrediction


@dataclass(frozen=True)
class Architecture:
    value: int

    def to_dict(self):
        return {"value": self.value}


@dataclass(frozen=True)
class Mutation:
    architecture: Architecture
    mutation_type: str = "test_mutation"


@dataclass(frozen=True)
class FitResult:
    model: object
    training_loss: float
    mean_training_mse: float
    instability_training_mse: float
    observation_count: int
    paired_count: int


class FakeExplicitSeedEvaluator:
    def __init__(self) -> None:
        self.real_training_runs = 0
        self.calls = []

    def evaluate_with_seed(self, architecture, training_seed):
        self.real_training_runs += 1
        self.calls.append((architecture, training_seed))
        accuracy = 0.60 + (architecture.value % 10) / 100.0
        return EvaluationOutcome(
            fitness=accuracy,
            metadata={
                "training_seed": training_seed,
                "final_val_accuracy": accuracy,
                "best_val_accuracy": accuracy + 0.01,
                "parameter_count": 1234,
                "training_time_seconds": 0.5,
            },
        )


def _fit(records, encodings):
    paired = sum(record.has_pair for record in records)
    return FitResult(
        model=object(),
        training_loss=0.03,
        mean_training_mse=0.02,
        instability_training_mse=0.01,
        observation_count=len(records),
        paired_count=paired,
    )


def _predict(model, encodings):
    del model
    count = len(encodings)
    return MultiTaskPrediction(
        predicted_mean=torch.tensor(
            [0.80 - 0.03 * index for index in range(count)]
        ),
        predicted_instability=torch.tensor(
            [0.10 if index == 0 else 0.01 for index in range(count)]
        ),
    )


def test_nasnet_runner_writes_separate_first_repeat_and_candidate_logs(
    tmp_path,
) -> None:
    evaluator = FakeExplicitSeedEvaluator()
    initialization = iter((Architecture(10), Architecture(20)))
    mutations = iter(range(100, 1000))
    result = run_nasnet_rs_sa_re(
        evaluator=evaluator,
        output_dir=tmp_path / "run",
        config_text="experiment:\n  method: RS-SA-RE",
        method="RS-SA-RE",
        search_seed=123,
        repeat_seed=77,
        training_seed_base=10_000,
        population_size=2,
        tournament_size=1,
        budget=6,
        candidate_count=2,
        stability_penalty_lambda=1.0,
        surrogate_config_values={"input_dim": 1, "steps": 1},
        repeat_policy_config=RepeatPolicyConfig(
            initial_population_size=2,
            warmup_pairs=1,
            repeat_interval=2,
            repeat_rate_beta=0.5,
        ),
        random_architecture_fn=lambda rng: next(initialization),
        mutate_fn=lambda parent, rng: Mutation(
            Architecture(next(mutations))
        ),
        encode_fn=lambda architecture: [float(architecture.value)],
        fit_surrogate_fn=_fit,
        predict_surrogate_fn=_predict,
        print_fn=lambda message: None,
    )

    with result.evaluations_csv_path.open(newline="", encoding="utf-8") as stream:
        evaluations = list(csv.DictReader(stream))
    with result.candidate_predictions_csv_path.open(
        newline="", encoding="utf-8"
    ) as stream:
        candidates = list(csv.DictReader(stream))
    with result.repeat_evaluations_csv_path.open(
        newline="", encoding="utf-8"
    ) as stream:
        repeats = list(csv.DictReader(stream))

    assert len(evaluations) == 6
    assert len([row for row in evaluations if row["event_type"] == "first_evaluation"]) == 4
    assert len([row for row in evaluations if row["event_type"] == "repeat_evaluation"]) == 2
    for row in evaluations:
        if row["event_type"] == "repeat_evaluation":
            assert row["population_inserted"] == "False"
            assert row["fitness"] == ""
        else:
            assert row["population_inserted"] == "True"

    assert len(candidates) == 4
    assert sum(row["selected"] == "True" for row in candidates) == 2
    for row in candidates:
        expected = float(row["predicted_mu"]) - float(row["lambda"]) * float(
            row["predicted_d"]
        )
        assert float(row["score"]) == pytest.approx(expected)

    assert len(repeats) == 2
    for row in repeats:
        base_index = int(row["base_evaluation_index"])
        assert int(row["seed_2"]) == derive_training_seed(
            training_seed_base=10_000,
            base_evaluation_index=base_index,
            replicate_id=2,
        )
        assert float(row["mean_target"]) == pytest.approx(
            (float(row["accuracy_1"]) + float(row["accuracy_2"])) / 2.0
        )

    history = json.loads(result.history_json_path.read_text(encoding="utf-8"))
    assert history["completed"] is True
    assert history["real_training_runs"] == 6
    assert history["first_evaluations"] == 4
    assert history["repeat_evaluations"] == 2


@dataclass(frozen=True)
class TrainingResult:
    final_val_accuracy: float = 0.70
    best_val_accuracy: float = 0.71
    parameter_count: int = 100
    training_time_seconds: float = 1.0


def _make_training_evaluator(seed_calls):
    return NASNetTrainingEvaluator(
        loader_factory=lambda seed: ("train", "val"),
        training_config_values={},
        training_seed_base=500,
        device="cpu",
        model_builder=lambda architecture, N, F, num_classes: object(),
        training_config_factory=lambda **values: values,
        trainer_fn=lambda model, train, val, config, device: TrainingResult(),
        seed_fn=seed_calls.append,
        cleanup_fn=lambda model, device: None,
    )


def test_nasnet_evaluator_preserves_old_call_schedule_and_accepts_explicit_seed() -> None:
    sequential_seeds = []
    sequential = _make_training_evaluator(sequential_seeds)
    sequential(Architecture(1))
    sequential(Architecture(2))
    assert sequential_seeds == [500, 501]
    assert sequential.completed_training_seeds == [500, 501]

    explicit_seeds = []
    explicit = _make_training_evaluator(explicit_seeds)
    explicit.evaluate_with_seed(Architecture(1), 1_000_500)
    assert explicit_seeds == [1_000_500]
    assert explicit.completed_training_seeds == [1_000_500]
