from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from scripts.check_rs_sa_re_smoke import (
    ExpectedBudgetFlow,
    audit_rs_sa_re_smoke,
    expected_budget_flow,
)
from scripts.run_rs_sa_re import load_config, validate_debug_config
from src.evolution.repeat_policy import RepeatPolicyConfig
from src.search.nasnet_rs_sa_re import run_nasnet_rs_sa_re
from src.search.regularized_evolution import EvaluationOutcome
from src.surrogate.multitask_model import MultiTaskPrediction


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_CONFIG_PATH = PROJECT_ROOT / "configs/debug/rs_sa_re_debug.yaml"


@dataclass(frozen=True)
class Architecture:
    value: int

    def to_dict(self):
        return {"value": self.value}


@dataclass(frozen=True)
class Mutation:
    architecture: Architecture
    mutation_type: str


@dataclass(frozen=True)
class FitResult:
    model: object
    training_loss: float
    mean_training_mse: float
    instability_training_mse: float
    observation_count: int
    paired_count: int


class FakeExplicitSeedEvaluator:
    def __init__(self, training_seed_base: int) -> None:
        self.training_seed_base = training_seed_base
        self.real_training_runs = 0

    def evaluate_with_seed(self, architecture, training_seed):
        self.real_training_runs += 1
        repeat_adjustment = (
            0.002
            if training_seed >= self.training_seed_base + 1_000_000
            else 0.0
        )
        accuracy = 0.60 + (architecture.value % 10) / 100.0
        accuracy += repeat_adjustment
        return EvaluationOutcome(
            fitness=accuracy,
            metadata={
                "training_seed": training_seed,
                "final_val_accuracy": accuracy,
                "best_val_accuracy": accuracy + 0.01,
                "parameter_count": 1234,
                "training_time_seconds": 0.01,
            },
        )


def _fit(records, encodings):
    assert len(records) == len(encodings)
    paired_count = sum(record.has_pair for record in records)
    return FitResult(
        model=object(),
        training_loss=0.03,
        mean_training_mse=0.02,
        instability_training_mse=0.01,
        observation_count=len(records),
        paired_count=paired_count,
    )


def _predict(model, encodings):
    del model
    assert len(encodings) == 5
    return MultiTaskPrediction(
        predicted_mean=torch.tensor(
            [0.80, 0.77, 0.75, 0.73, 0.71],
            dtype=torch.float32,
        ),
        predicted_instability=torch.tensor(
            [0.10, 0.01, 0.03, 0.02, 0.04],
            dtype=torch.float32,
        ),
    )


def test_debug_config_and_exact_b30_flow_are_frozen() -> None:
    config = load_config(DEBUG_CONFIG_PATH)
    validate_debug_config(config)

    assert config["experiment"]["search_seed"] == 2910
    assert config["training"]["epochs"] == 1
    assert config["stability"]["lambda"] == 1.0
    assert expected_budget_flow(
        population_size=20,
        warmup_pairs=4,
        repeat_interval=4,
        budget=30,
    ) == ExpectedBudgetFlow(
        initial_first_evaluations=20,
        warmup_repeats=4,
        evolutionary_children=5,
        periodic_repeats=1,
    )


def test_fake_b30_smoke_produces_auditable_real_run_logs(tmp_path) -> None:
    config_text = DEBUG_CONFIG_PATH.read_text(encoding="utf-8")
    config = load_config(DEBUG_CONFIG_PATH)
    training_seed_base = int(config["training"]["training_seed_base"])
    evaluator = FakeExplicitSeedEvaluator(training_seed_base)
    initialization = iter(Architecture(index) for index in range(1, 21))
    mutation_counter = iter(range(1000, 10_000))
    diagnostics: list[str] = []

    def mutate(parent, rng):
        del parent, rng
        index = next(mutation_counter)
        return Mutation(
            architecture=Architecture(index),
            mutation_type=("op" if index % 2 == 0 else "input"),
        )

    result = run_nasnet_rs_sa_re(
        evaluator=evaluator,
        output_dir=tmp_path / "rs_sa_re_2910",
        config_text=config_text,
        method="RS-SA-RE",
        search_seed=2910,
        repeat_seed=1_802_910,
        training_seed_base=training_seed_base,
        population_size=20,
        tournament_size=5,
        budget=30,
        candidate_count=5,
        stability_penalty_lambda=1.0,
        surrogate_config_values={"input_dim": 1, "steps": 1},
        repeat_policy_config=RepeatPolicyConfig(),
        random_architecture_fn=lambda rng: next(initialization),
        mutate_fn=mutate,
        encode_fn=lambda architecture: [float(architecture.value)],
        fit_surrogate_fn=_fit,
        predict_surrogate_fn=_predict,
        print_fn=diagnostics.append,
    )

    summary = audit_rs_sa_re_smoke(result.output_dir)
    assert summary.real_training_runs == 30
    assert summary.first_evaluations == 25
    assert summary.warmup_repeats == 4
    assert summary.periodic_repeats == 1
    assert summary.final_population_size == 20
    assert summary.candidate_rows == 25
    assert summary.selected_candidate_rows == 5

    assert any(line == "[Surrogate training]" for line in diagnostics)
    assert any(line.startswith("mu_target:") for line in diagnostics)
    assert any(line.startswith("d_target:") for line in diagnostics)
    assert any(line.startswith("predicted_mu:") for line in diagnostics)
    assert any(line.startswith("predicted_d:") for line in diagnostics)
    assert any(
        line == "[RS-SA-RE budget 25 / first eval 21]"
        for line in diagnostics
    )
    assert any(line == "Selected: candidate 1" for line in diagnostics)
    assert sum(line == "[Repeat]" for line in diagnostics) == 5
    assert sum("population unchanged" in line for line in diagnostics) == 5

    run_log = result.run_log_path.read_text(encoding="utf-8")
    assert "predicted_d:" in run_log
    assert "Selected: candidate 1" in run_log
    assert "population unchanged" in run_log
