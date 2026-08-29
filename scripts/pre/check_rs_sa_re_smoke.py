"""Audit a completed RS-SA-RE smoke from its persisted logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import yaml


@dataclass(frozen=True)
class ExpectedBudgetFlow:
    initial_first_evaluations: int
    warmup_repeats: int
    evolutionary_children: int
    periodic_repeats: int

    @property
    def first_evaluations(self) -> int:
        return self.initial_first_evaluations + self.evolutionary_children

    @property
    def repeat_evaluations(self) -> int:
        return self.warmup_repeats + self.periodic_repeats

    @property
    def real_training_runs(self) -> int:
        return self.first_evaluations + self.repeat_evaluations


@dataclass(frozen=True)
class SmokeAuditSummary:
    real_training_runs: int
    first_evaluations: int
    warmup_repeats: int
    periodic_repeats: int
    final_population_size: int
    candidate_rows: int
    selected_candidate_rows: int


def expected_budget_flow(
    *,
    population_size: int,
    warmup_pairs: int,
    repeat_interval: int,
    budget: int,
) -> ExpectedBudgetFlow:
    if population_size <= 0 or repeat_interval <= 0:
        raise ValueError("population_size and repeat_interval must be positive")
    if not 0 <= warmup_pairs <= population_size:
        raise ValueError("warmup_pairs is inconsistent with population_size")
    used = population_size + warmup_pairs
    if budget < used:
        raise ValueError("budget cannot complete initialization and warm-up")

    children = 0
    periodic_repeats = 0
    while used < budget:
        used += 1
        children += 1
        if used < budget and children % repeat_interval == 0:
            used += 1
            periodic_repeats += 1
    if used != budget:
        raise RuntimeError("expected budget-flow calculation exceeded budget")
    return ExpectedBudgetFlow(
        initial_first_evaluations=population_size,
        warmup_repeats=warmup_pairs,
        evolutionary_children=children,
        periodic_repeats=periodic_repeats,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _parse_bool(value: str, *, field_name: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{field_name} must contain True or False")


def _finite_float(value: str, *, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def _json_object(value: str, *, field_name: str) -> dict[str, Any]:
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise ValueError(f"{field_name} must encode a JSON object")
    return decoded


def audit_rs_sa_re_smoke(output_dir: str | Path) -> SmokeAuditSummary:
    output_dir = Path(output_dir)
    required_paths = {
        "config": output_dir / "config.yaml",
        "evaluations": output_dir / "evaluations.csv",
        "candidates": output_dir / "candidate_predictions.csv",
        "repeats": output_dir / "repeat_evaluations.csv",
        "history": output_dir / "history.json",
        "run_log": output_dir / "run.log",
    }
    missing = [name for name, path in required_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "smoke output is missing required files: " + ", ".join(missing)
        )

    config = yaml.safe_load(required_paths["config"].read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a mapping")
    experiment = config["experiment"]
    training = config["training"]
    evolution = config["evolution"]
    stability = config["stability"]
    if str(experiment["method"]).upper() != "RS-SA-RE":
        raise ValueError("audit expected experiment.method=RS-SA-RE")
    if str(experiment["mode"]).lower() != "debug":
        raise ValueError("audit expected experiment.mode=debug")
    if int(training["epochs"]) != 1:
        raise ValueError("real smoke must use one training epoch")

    population_size = int(evolution["population_size"])
    tournament_size = int(evolution["tournament_size"])
    budget = int(evolution["budget"])
    candidate_count = int(evolution["candidate_count"])
    warmup_pairs = int(stability["warmup_pairs"])
    repeat_interval = int(stability["repeat_interval"])
    penalty_lambda = float(stability["lambda"])
    if (
        population_size,
        tournament_size,
        budget,
        candidate_count,
    ) != (20, 5, 30, 5):
        raise ValueError("smoke evolution config must be P=20, S=5, B=30, K=5")
    if not math.isclose(penalty_lambda, 1.0, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("smoke audit requires debug lambda=1.0")

    expected = expected_budget_flow(
        population_size=population_size,
        warmup_pairs=warmup_pairs,
        repeat_interval=repeat_interval,
        budget=budget,
    )
    if expected != ExpectedBudgetFlow(20, 4, 5, 1):
        raise RuntimeError("B=30 flow differs from 20+4+5+1")

    evaluations = _read_csv(required_paths["evaluations"])
    candidates = _read_csv(required_paths["candidates"])
    repeats = _read_csv(required_paths["repeats"])
    history = json.loads(required_paths["history"].read_text(encoding="utf-8"))
    run_log_text = required_paths["run_log"].read_text(encoding="utf-8")

    if len(evaluations) != budget:
        raise RuntimeError("evaluations.csv must contain exactly 30 real events")
    budget_indices = [int(row["budget_index"]) for row in evaluations]
    if budget_indices != list(range(1, budget + 1)):
        raise RuntimeError("evaluation budget indices are not contiguous 1..30")

    first_rows = [
        row for row in evaluations if row["event_type"] == "first_evaluation"
    ]
    repeat_event_rows = [
        row for row in evaluations if row["event_type"] == "repeat_evaluation"
    ]
    unknown_event_rows = [
        row
        for row in evaluations
        if row["event_type"] not in {"first_evaluation", "repeat_evaluation"}
    ]
    if unknown_event_rows:
        raise RuntimeError("evaluations.csv contains an unknown event_type")
    if len(first_rows) != expected.first_evaluations:
        raise RuntimeError("first-evaluation count is inconsistent")
    if len(repeat_event_rows) != expected.repeat_evaluations:
        raise RuntimeError("repeat-evaluation count is inconsistent")

    warmup_event_rows = [
        row for row in repeat_event_rows if row["phase"] == "warmup_repeat"
    ]
    periodic_event_rows = [
        row for row in repeat_event_rows if row["phase"] == "periodic_repeat"
    ]
    if len(warmup_event_rows) != expected.warmup_repeats:
        raise RuntimeError("warm-up repeat count is inconsistent")
    if len(periodic_event_rows) != expected.periodic_repeats:
        raise RuntimeError("periodic repeat count is inconsistent")

    for row in first_rows:
        if not _parse_bool(
            row["population_inserted"], field_name="population_inserted"
        ):
            raise RuntimeError("every first evaluation must enter population")
    for row in repeat_event_rows:
        if _parse_bool(
            row["population_inserted"], field_name="population_inserted"
        ):
            raise RuntimeError("repeat evaluation entered the population")
        if row["fitness"] != "":
            raise RuntimeError("repeat evaluation must not contain population fitness")

    population_orders = []
    for row in evaluations:
        age_order = _json_object(
            row["population_age_order"],
            field_name="population_age_order",
        )
        order = age_order.get("oldest_to_youngest_evaluation_indices")
        if not isinstance(order, list):
            raise ValueError("population age order omitted evaluation indices")
        population_orders.append(tuple(int(value) for value in order))
    for index, row in enumerate(evaluations):
        if row["event_type"] == "repeat_evaluation":
            if index == 0 or population_orders[index] != population_orders[index - 1]:
                raise RuntimeError("repeat changed the population architecture/order")
            if len(population_orders[index]) != population_size:
                raise RuntimeError("repeat population size differs from P")

    child_evaluation_indices = list(
        range(population_size + 1, expected.first_evaluations + 1)
    )
    candidate_groups: dict[int, list[dict[str, str]]] = {}
    for row in candidates:
        evaluation_index = int(row["evaluation_index"])
        candidate_groups.setdefault(evaluation_index, []).append(row)
        predicted_mu = _finite_float(
            row["predicted_mu"], field_name="predicted_mu"
        )
        predicted_d = _finite_float(
            row["predicted_d"], field_name="predicted_d"
        )
        row_lambda = _finite_float(row["lambda"], field_name="lambda")
        score = _finite_float(row["score"], field_name="score")
        if predicted_d < 0.0:
            raise RuntimeError("predicted_d must be non-negative")
        if not math.isclose(row_lambda, penalty_lambda, rel_tol=0.0, abs_tol=0.0):
            raise RuntimeError("candidate row uses the wrong lambda")
        expected_score = predicted_mu - row_lambda * predicted_d
        if not math.isclose(score, expected_score, rel_tol=1e-6, abs_tol=1e-7):
            raise RuntimeError("candidate score is not mu_hat - lambda*d_hat")

    if sorted(candidate_groups) != child_evaluation_indices:
        raise RuntimeError("candidate batches do not match first-evaluated children")
    for evaluation_index, rows in candidate_groups.items():
        if len(rows) != candidate_count:
            raise RuntimeError(
                f"evaluation {evaluation_index} does not contain K candidates"
            )
        indices = sorted(int(row["candidate_index"]) for row in rows)
        if indices != list(range(candidate_count)):
            raise RuntimeError("candidate indices must be 0..K-1")
        selected_rows = [
            row
            for row in rows
            if _parse_bool(row["selected"], field_name="selected")
        ]
        if len(selected_rows) != 1:
            raise RuntimeError("each candidate batch must select exactly one row")

    expected_candidate_rows = expected.evolutionary_children * candidate_count
    if len(candidates) != expected_candidate_rows:
        raise RuntimeError("candidate_predictions.csv has the wrong row count")
    selected_candidate_rows = sum(
        _parse_bool(row["selected"], field_name="selected")
        for row in candidates
    )
    if selected_candidate_rows != expected.evolutionary_children:
        raise RuntimeError("selected-candidate count is inconsistent")

    if len(repeats) != expected.repeat_evaluations:
        raise RuntimeError("repeat_evaluations.csv has the wrong row count")
    repeat_base_indices = [int(row["base_evaluation_index"]) for row in repeats]
    if len(set(repeat_base_indices)) != len(repeat_base_indices):
        raise RuntimeError("one base evaluation received multiple scheduled repeats")
    repeat_budget_indices = [int(row["budget_index"]) for row in repeats]
    event_repeat_budget_indices = [
        int(row["budget_index"]) for row in repeat_event_rows
    ]
    if repeat_budget_indices != event_repeat_budget_indices:
        raise RuntimeError("repeat CSV and event log budget indices differ")
    for row in repeats:
        seed_1 = int(row["seed_1"])
        seed_2 = int(row["seed_2"])
        if seed_1 == seed_2:
            raise RuntimeError("repeat seed must differ from first seed")
        accuracy_1 = _finite_float(row["accuracy_1"], field_name="accuracy_1")
        accuracy_2 = _finite_float(row["accuracy_2"], field_name="accuracy_2")
        mean_target = _finite_float(row["mean_target"], field_name="mean_target")
        instability = _finite_float(
            row["instability_target"], field_name="instability_target"
        )
        if not 0.0 <= accuracy_1 <= 1.0 or not 0.0 <= accuracy_2 <= 1.0:
            raise RuntimeError("repeat accuracy lies outside [0, 1]")
        if not math.isclose(
            mean_target,
            (accuracy_1 + accuracy_2) / 2.0,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise RuntimeError("repeat mean target is incorrect")
        if not math.isclose(
            instability,
            abs(accuracy_1 - accuracy_2),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise RuntimeError("repeat instability target is incorrect")

    if not isinstance(history, dict) or history.get("completed") is not True:
        raise RuntimeError("history.json does not mark the smoke complete")
    if int(history["real_training_runs"]) != budget:
        raise RuntimeError("history.json real-training count is incorrect")
    if int(history["first_evaluations"]) != expected.first_evaluations:
        raise RuntimeError("history.json first-evaluation count is incorrect")
    if int(history["repeat_evaluations"]) != expected.repeat_evaluations:
        raise RuntimeError("history.json repeat-evaluation count is incorrect")
    final_population_order = tuple(
        int(value) for value in history["final_population_order"]
    )
    if len(final_population_order) != population_size:
        raise RuntimeError("final population size differs from P=20")
    if final_population_order != population_orders[-1]:
        raise RuntimeError("history/evaluation final population orders differ")

    required_diagnostics = (
        "[Surrogate training]",
        "paired_label_count=",
        "mu_target:",
        "d_target:",
        "predicted_mu:",
        "predicted_d:",
        "[Repeat]",
        "population unchanged",
        "Selected: candidate",
    )
    missing_diagnostics = [
        marker for marker in required_diagnostics if marker not in run_log_text
    ]
    if missing_diagnostics:
        raise RuntimeError(
            "run.log omitted diagnostics: " + ", ".join(missing_diagnostics)
        )

    return SmokeAuditSummary(
        real_training_runs=len(evaluations),
        first_evaluations=len(first_rows),
        warmup_repeats=len(warmup_event_rows),
        periodic_repeats=len(periodic_event_rows),
        final_population_size=len(final_population_order),
        candidate_rows=len(candidates),
        selected_candidate_rows=selected_candidate_rows,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit completed RS-SA-RE debug-smoke logs."
    )
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    summary = audit_rs_sa_re_smoke(parse_args().output_dir)
    print("RS-SA-RE smoke audit: PASS")
    print(f"real training runs: {summary.real_training_runs}")
    print(f"first evaluations: {summary.first_evaluations}")
    print(f"warm-up repeats: {summary.warmup_repeats}")
    print(f"periodic repeats: {summary.periodic_repeats}")
    print(f"final population: {summary.final_population_size}")
    print(f"candidate rows: {summary.candidate_rows}")
    print(f"selected rows: {summary.selected_candidate_rows}")


if __name__ == "__main__":
    main()
