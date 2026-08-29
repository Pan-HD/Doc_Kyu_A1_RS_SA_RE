from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from scripts.check_rs_sa_re_pilot import audit_matched_rs_sa_re_pilot
from scripts.check_rs_sa_re_smoke import (
    ExpectedBudgetFlow,
    expected_budget_flow,
)
from scripts.run_rs_sa_re import (
    load_config,
    validate_rs_sa_re_config,
)
from src.evolution.repeat_policy import RepeatPolicyConfig, RepeatScheduler
from src.search.nasnet_rs_sa_re import run_nasnet_rs_sa_re
from src.search.regularized_evolution import EvaluationOutcome
from src.surrogate.multitask_dataset import PairedEvaluationRecord
from src.surrogate.multitask_model import MultiTaskPrediction


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PILOT_CONFIGS = {
    2701: PROJECT_ROOT / "configs/pilot/rs_sa_re_2701.yaml",
    2702: PROJECT_ROOT / "configs/pilot/rs_sa_re_2702.yaml",
}


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


class FakeEvaluator:
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
    return FitResult(
        model=object(),
        training_loss=0.03,
        mean_training_mse=0.02,
        instability_training_mse=0.01,
        observation_count=len(records),
        paired_count=sum(record.has_pair for record in records),
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


def _repeat_records() -> list[PairedEvaluationRecord]:
    return [
        PairedEvaluationRecord(
            base_evaluation_index=index,
            architecture=Architecture(index),
            seed_1=20_260_827 + index - 1,
            accuracy_1=0.60,
        )
        for index in range(1, 21)
    ]


def test_matched_pilot_configs_and_rng_namespaces_are_frozen() -> None:
    for search_seed, path in PILOT_CONFIGS.items():
        config = load_config(path)
        validate_rs_sa_re_config(config)
        assert config["experiment"]["mode"] == "pilot"
        assert config["experiment"]["search_seed"] == search_seed
        assert config["training"]["epochs"] == 5
        assert config["training"]["training_seed_base"] == 20_260_827
        assert config["surrogate"]["seed_offset"] == 900_000
        assert config["stability"]["repeat_seed"] == 1_800_000 + search_seed
        assert config["stability"]["lambda"] == 1.0

    assert expected_budget_flow(
        population_size=20,
        warmup_pairs=4,
        repeat_interval=4,
        budget=30,
    ) == ExpectedBudgetFlow(20, 4, 5, 1)


def test_repeat_selection_does_not_shift_matched_search_rng() -> None:
    for search_seed in PILOT_CONFIGS:
        search_rng_a = random.Random(search_seed)
        search_rng_b = random.Random(search_seed)
        RepeatScheduler(
            repeat_seed=1_800_000 + search_seed
        ).select_warmup(_repeat_records())
        assert search_rng_a.getstate() == search_rng_b.getstate()
        assert search_rng_a.random() == search_rng_b.random()


def _write_reference_initialization(
    *,
    source_csv: Path,
    destination_dir: Path,
) -> None:
    with source_csv.open(newline="", encoding="utf-8") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["phase"] == "initialization"
        ][:20]
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / "evaluations.csv"
    with destination_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_fake_pilot_passes_internal_and_matched_initialization_audits(
    tmp_path,
) -> None:
    config_path = PILOT_CONFIGS[2701]
    config = load_config(config_path)
    training_seed_base = int(config["training"]["training_seed_base"])
    evaluator = FakeEvaluator(training_seed_base)
    initialization = iter(Architecture(index) for index in range(1, 21))
    mutation_counter = iter(range(1000, 10_000))

    def mutate(parent, rng):
        del parent, rng
        value = next(mutation_counter)
        return Mutation(
            architecture=Architecture(value),
            mutation_type=("op" if value % 2 == 0 else "input"),
        )

    result = run_nasnet_rs_sa_re(
        evaluator=evaluator,
        output_dir=tmp_path / "rs_sa_re_2701",
        config_text=config_path.read_text(encoding="utf-8"),
        method="RS-SA-RE",
        search_seed=2701,
        repeat_seed=1_802_701,
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
        print_fn=lambda message: None,
    )
    re_dir = tmp_path / "re_2701"
    sa_re_dir = tmp_path / "sa_re_2701"
    _write_reference_initialization(
        source_csv=result.evaluations_csv_path,
        destination_dir=re_dir,
    )
    _write_reference_initialization(
        source_csv=result.evaluations_csv_path,
        destination_dir=sa_re_dir,
    )

    summary = audit_matched_rs_sa_re_pilot(
        result.output_dir,
        re_output_dir=re_dir,
        sa_re_output_dir=sa_re_dir,
    )
    assert summary.search_seed == 2701
    assert summary.run.real_training_runs == 30
    assert summary.run.first_evaluations == 25
    assert summary.run.warmup_repeats == 4
    assert summary.run.periodic_repeats == 1
    assert summary.run.final_population_size == 20
    assert summary.architecture_matches_re == 20
    assert summary.architecture_matches_sa_re == 20
    assert summary.training_seed_matches_re == 20
    assert summary.training_seed_matches_sa_re == 20
