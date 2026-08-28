"""Compare matched RE and SA-RE pilot runs and create Part C artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence

import yaml


EXPECTED_RUNS = (
    ("RE", 2701, "re_2701"),
    ("SA-RE", 2701, "sa_re_2701"),
    ("RE", 2702, "re_2702"),
    ("SA-RE", 2702, "sa_re_2702"),
)
SUMMARY_FIELDS = (
    "method",
    "search_seed",
    "initial_best",
    "final_best",
    "final_population_best",
    "best_at_budget_20",
    "best_at_budget_25",
    "best_at_budget_30",
    "best_evaluation_index",
    "runtime_seconds",
    "runtime_minutes",
    "mean_training_time_seconds",
    "parameter_count_of_best",
    "real_training_runs",
    "population_size",
    "budget",
)


class ComparisonError(ValueError):
    """Raised when the pilot artifacts cannot support a valid comparison."""


@dataclass(frozen=True)
class RunData:
    method: str
    search_seed: int
    run_dir: Path
    config: dict[str, object]
    evaluations: tuple[dict[str, str], ...]
    history: dict[str, object]
    accuracies: tuple[float, ...]
    best_so_far: tuple[float, ...]
    final_population_order: tuple[int, ...]
    runtime_seconds: float
    summary_row: dict[str, object]
    artifact_checks: dict[str, bool]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ComparisonError(f"missing required file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ComparisonError(f"CSV contains no rows: {path}")
    return rows


def _require_columns(
    rows: Sequence[dict[str, str]], required: Iterable[str], path: Path
) -> None:
    missing = sorted(set(required) - set(rows[0]))
    if missing:
        raise ComparisonError(
            f"{path} is missing required columns: {', '.join(missing)}"
        )


def _integer(value: object, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ComparisonError(f"{label} must be an integer, got {value!r}") from error


def _finite_float(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ComparisonError(f"{label} must be numeric, got {value!r}") from error
    if not math.isfinite(result):
        raise ComparisonError(f"{label} must be finite, got {value!r}")
    return result


def _boolean(value: object, label: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ComparisonError(f"{label} must be boolean, got {value!r}")


def _unique(values: Iterable[object], label: str) -> object:
    unique_values = {str(value).strip() for value in values}
    if len(unique_values) != 1:
        raise ComparisonError(f"expected one {label}, got {sorted(unique_values)!r}")
    return unique_values.pop()


def _load_yaml(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ComparisonError(f"missing required file: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ComparisonError(f"YAML root must be a mapping: {path}")
    return data


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ComparisonError(f"missing required file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ComparisonError(f"JSON root must be an object: {path}")
    return data


def _best_so_far(accuracies: Sequence[float]) -> list[float]:
    running_best = float("-inf")
    result: list[float] = []
    for accuracy in accuracies:
        running_best = max(running_best, accuracy)
        result.append(running_best)
    return result


def _parse_runtime_seconds(path: Path) -> float:
    if not path.is_file():
        raise ComparisonError(f"missing required file: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    start_lines = [line for line in lines if " START " in f" {line} "]
    completed_lines = [line for line in lines if re.search(r"\bcompleted\b", line, re.I)]
    if len(start_lines) != 1 or len(completed_lines) != 1:
        raise ComparisonError(
            f"run.log must contain exactly one START and one completed line: {path}"
        )
    try:
        started = datetime.fromisoformat(start_lines[0].split(maxsplit=1)[0])
        completed = datetime.fromisoformat(completed_lines[0].split(maxsplit=1)[0])
    except ValueError as error:
        raise ComparisonError(f"invalid ISO-8601 timestamp in {path}") from error
    runtime = (completed - started).total_seconds()
    if runtime <= 0.0:
        raise ComparisonError(f"runtime must be positive: {path}")
    return runtime


def _expected_fifo_order(budget: int, population_size: int) -> list[int]:
    return list(range(budget - population_size + 1, budget + 1))


def _load_run(pilot_root: Path, method: str, seed: int, directory_name: str) -> RunData:
    run_dir = pilot_root / directory_name
    config = _load_yaml(run_dir / "config.yaml")
    history = _load_json(run_dir / "history.json")
    rows = _read_csv(run_dir / "evaluations.csv")
    _require_columns(
        rows,
        {
            "method",
            "search_seed",
            "training_seed",
            "evaluation_index",
            "budget",
            "phase",
            "architecture",
            "mutation_type",
            "final_val_accuracy",
            "parameter_count",
            "training_time",
        },
        run_dir / "evaluations.csv",
    )
    rows.sort(key=lambda row: _integer(row["evaluation_index"], "evaluation_index"))
    actual_method = str(_unique((row["method"] for row in rows), "method"))
    actual_seed = _integer(_unique((row["search_seed"] for row in rows), "search seed"), "search seed")
    if actual_method != method or actual_seed != seed:
        raise ComparisonError(
            f"directory {directory_name} contains {actual_method} seed {actual_seed}, "
            f"expected {method} seed {seed}"
        )

    budget = _integer(_unique((row["budget"] for row in rows), "budget"), "budget")
    indices = [_integer(row["evaluation_index"], "evaluation_index") for row in rows]
    if indices != list(range(1, budget + 1)):
        raise ComparisonError(f"{directory_name}: evaluations must be contiguous 1..budget")
    if len(rows) != budget:
        raise ComparisonError(f"{directory_name}: real evaluation count does not equal budget")

    evolution_config = config.get("evolution")
    if not isinstance(evolution_config, dict):
        raise ComparisonError(f"{directory_name}: config.evolution must be a mapping")
    population_size = _integer(
        evolution_config.get("population_size"), "population_size"
    )
    if population_size != 20 or budget != 30:
        raise ComparisonError(
            f"{directory_name}: Part C expects population_size=20 and budget=30"
        )
    expected_phases = ["initialization"] * population_size + ["evolution"] * (
        budget - population_size
    )
    if [row["phase"] for row in rows] != expected_phases:
        raise ComparisonError(f"{directory_name}: phase sequence is inconsistent")

    accuracies: list[float] = []
    training_times: list[float] = []
    parameter_counts: list[int] = []
    for row in rows:
        index = _integer(row["evaluation_index"], "evaluation_index")
        accuracy = _finite_float(
            row["final_val_accuracy"], f"final_val_accuracy at evaluation {index}"
        )
        if not 0.0 <= accuracy <= 1.0:
            raise ComparisonError(
                f"{directory_name}: final_val_accuracy must use the 0.0-1.0 scale"
            )
        training_time = _finite_float(
            row["training_time"], f"training_time at evaluation {index}"
        )
        if training_time <= 0.0:
            raise ComparisonError(f"{directory_name}: training_time must be positive")
        parameter_count = _integer(row["parameter_count"], "parameter_count")
        if parameter_count <= 0:
            raise ComparisonError(f"{directory_name}: parameter_count must be positive")
        accuracies.append(accuracy)
        training_times.append(training_time)
        parameter_counts.append(parameter_count)

    best_values = _best_so_far(accuracies)
    final_best = best_values[-1]
    best_evaluation_index = accuracies.index(final_best) + 1
    final_population_raw = history.get("final_population_order")
    if not isinstance(final_population_raw, list):
        raise ComparisonError(f"{directory_name}: missing final_population_order")
    final_population_order = tuple(
        _integer(index, "final population index") for index in final_population_raw
    )
    if len(final_population_order) != population_size or len(set(final_population_order)) != population_size:
        raise ComparisonError(f"{directory_name}: invalid final population order")
    if any(index < 1 or index > budget for index in final_population_order):
        raise ComparisonError(f"{directory_name}: final population index outside budget")

    history_complete = bool(history.get("completed"))
    history_real_runs = _integer(history.get("real_training_runs"), "history real_training_runs")
    runtime_seconds = _parse_runtime_seconds(run_dir / "run.log")
    summary_row: dict[str, object] = {
        "method": method,
        "search_seed": seed,
        "initial_best": best_values[population_size - 1],
        "final_best": final_best,
        "final_population_best": max(accuracies[index - 1] for index in final_population_order),
        "best_at_budget_20": best_values[19],
        "best_at_budget_25": best_values[24],
        "best_at_budget_30": best_values[29],
        "best_evaluation_index": best_evaluation_index,
        "runtime_seconds": runtime_seconds,
        "runtime_minutes": runtime_seconds / 60.0,
        "mean_training_time_seconds": mean(training_times),
        "parameter_count_of_best": parameter_counts[best_evaluation_index - 1],
        "real_training_runs": len(rows),
        "population_size": population_size,
        "budget": budget,
    }
    artifact_checks = {
        "history_completed": history_complete,
        "history_real_training_runs_match_budget": history_real_runs == budget,
        "evaluation_count_matches_budget": len(rows) == budget,
        "accuracy_targets_finite_and_in_unit_interval": True,
        "training_times_finite_and_positive": True,
        "final_population_size_matches": len(final_population_order) == population_size,
        "fifo_final_population_order_valid": list(final_population_order)
        == _expected_fifo_order(budget, population_size),
        "run_log_complete": runtime_seconds > 0.0,
    }
    return RunData(
        method=method,
        search_seed=seed,
        run_dir=run_dir,
        config=config,
        evaluations=tuple(rows),
        history=history,
        accuracies=tuple(accuracies),
        best_so_far=tuple(best_values),
        final_population_order=final_population_order,
        runtime_seconds=runtime_seconds,
        summary_row=summary_row,
        artifact_checks=artifact_checks,
    )


def _shared_config_checks(re_run: RunData, sa_run: RunData) -> dict[str, bool]:
    re_evolution = re_run.config["evolution"]
    sa_evolution = sa_run.config["evolution"]
    assert isinstance(re_evolution, dict) and isinstance(sa_evolution, dict)
    return {
        "dataset_config_matches": re_run.config.get("dataset") == sa_run.config.get("dataset"),
        "network_config_matches": re_run.config.get("network") == sa_run.config.get("network"),
        "training_config_matches": re_run.config.get("training") == sa_run.config.get("training"),
        "device_config_matches": re_run.config.get("device") == sa_run.config.get("device"),
        "population_size_matches": re_evolution.get("population_size")
        == sa_evolution.get("population_size"),
        "tournament_size_matches": re_evolution.get("tournament_size")
        == sa_evolution.get("tournament_size"),
        "real_training_budget_matches": re_evolution.get("budget")
        == sa_evolution.get("budget"),
    }


def _matched_initialization(re_run: RunData, sa_run: RunData) -> dict[str, object]:
    population_size = _integer(re_run.summary_row["population_size"], "population size")
    architecture_matches = 0
    training_seed_matches = 0
    accuracy_matches = 0
    accuracy_deltas: list[float] = []
    mismatch_indices: list[int] = []
    for offset in range(population_size):
        re_row = re_run.evaluations[offset]
        sa_row = sa_run.evaluations[offset]
        architecture_match = re_row["architecture"] == sa_row["architecture"]
        seed_match = re_row["training_seed"] == sa_row["training_seed"]
        re_accuracy = re_run.accuracies[offset]
        sa_accuracy = sa_run.accuracies[offset]
        accuracy_match = re_accuracy == sa_accuracy
        architecture_matches += int(architecture_match)
        training_seed_matches += int(seed_match)
        accuracy_matches += int(accuracy_match)
        accuracy_deltas.append(abs(re_accuracy - sa_accuracy))
        if not (architecture_match and seed_match and accuracy_match):
            mismatch_indices.append(offset + 1)
    return {
        "population_size": population_size,
        "architecture_matches": architecture_matches,
        "training_seed_matches": training_seed_matches,
        "accuracy_matches_exact": accuracy_matches,
        "maximum_accuracy_absolute_delta": max(accuracy_deltas),
        "mean_accuracy_absolute_delta": mean(accuracy_deltas),
        "diagnostic_mismatch_indices": mismatch_indices,
        "hard_gate_pass": architecture_matches == population_size
        and training_seed_matches == population_size,
        "accuracy_is_diagnostic_only": True,
    }


def _candidate_checks(sa_run: RunData) -> dict[str, object]:
    path = sa_run.run_dir / "candidate_predictions.csv"
    rows = _read_csv(path)
    _require_columns(
        rows,
        {
            "method",
            "search_seed",
            "evaluation_index",
            "candidate_index",
            "predicted_mu",
            "selected",
            "surrogate_training_size",
        },
        path,
    )
    population_size = _integer(sa_run.summary_row["population_size"], "population size")
    budget = _integer(sa_run.summary_row["budget"], "budget")
    evolution_config = sa_run.config.get("evolution")
    if not isinstance(evolution_config, dict):
        raise ComparisonError("SA-RE evolution config must be a mapping")
    candidate_count = _integer(evolution_config.get("candidate_count"), "candidate_count")
    groups: dict[int, list[dict[str, str]]] = {}
    all_predictions: list[float] = []
    for row in rows:
        if row["method"] != "SA-RE" or _integer(row["search_seed"], "candidate seed") != sa_run.search_seed:
            raise ComparisonError("candidate method/search seed does not match its SA-RE run")
        evaluation_index = _integer(row["evaluation_index"], "candidate evaluation_index")
        groups.setdefault(evaluation_index, []).append(row)
        all_predictions.append(
            _finite_float(row["predicted_mu"], "candidate predicted_mu")
        )
    expected_indices = list(range(population_size + 1, budget + 1))
    group_indices_match = sorted(groups) == expected_indices
    rows_per_step_match = group_indices_match and all(
        len(groups[index]) == candidate_count for index in expected_indices
    )
    selected_per_step_match = rows_per_step_match and all(
        sum(_boolean(row["selected"], "selected") for row in groups[index]) == 1
        for index in expected_indices
    )
    selected_is_max_prediction = selected_per_step_match and all(
        max(_finite_float(row["predicted_mu"], "predicted_mu") for row in groups[index])
        == next(
            _finite_float(row["predicted_mu"], "selected predicted_mu")
            for row in groups[index]
            if _boolean(row["selected"], "selected")
        )
        for index in expected_indices
    )
    training_sizes_match = rows_per_step_match and all(
        {
            _integer(row["surrogate_training_size"], "surrogate_training_size")
            for row in groups[index]
        }
        == {index - 1}
        for index in expected_indices
    )
    return {
        "configured_candidate_count": candidate_count,
        "candidate_rows": len(rows),
        "expected_candidate_rows": (budget - population_size) * candidate_count,
        "selected_rows": sum(_boolean(row["selected"], "selected") for row in rows),
        "expected_selected_rows": budget - population_size,
        "candidate_groups_cover_evolution": group_indices_match,
        "candidate_rows_per_step_match_config": rows_per_step_match,
        "exactly_one_selected_per_step": selected_per_step_match,
        "selected_candidate_has_max_predicted_mu": selected_is_max_prediction,
        "prediction_time_training_sizes_match": training_sizes_match,
        "final_surrogate_dataset_size": budget,
        "predictions_all_finite": all(math.isfinite(value) for value in all_predictions),
        "prediction_minimum": min(all_predictions),
        "prediction_maximum": max(all_predictions),
        "prediction_mean": mean(all_predictions),
        "hard_gate_pass": all(
            (
                candidate_count == 5,
                len(rows) == (budget - population_size) * candidate_count,
                group_indices_match,
                rows_per_step_match,
                selected_per_step_match,
                selected_is_max_prediction,
                training_sizes_match,
                all(math.isfinite(value) for value in all_predictions),
            )
        ),
    }


def _write_summary(path: Path, runs: Sequence[RunData]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(run.summary_row for run in runs)


def _write_plot(path: Path, runs: Sequence[RunData]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        ("RE", 2701): ("#3569a8", "-"),
        ("SA-RE", 2701): ("#d97706", "-"),
        ("RE", 2702): ("#3569a8", "--"),
        ("SA-RE", 2702): ("#d97706", "--"),
    }
    fig, ax = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    x_values = list(range(1, 31))
    all_best_values: list[float] = []
    ax.axvspan(1, 20, color="#64748b", alpha=0.06)
    ax.axvspan(20, 30, color="#f59e0b", alpha=0.05)
    for run in runs:
        color, linestyle = styles[(run.method, run.search_seed)]
        ax.plot(
            x_values,
            run.best_so_far,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            marker="o",
            markersize=3.0,
            label=f"{run.method} {run.search_seed}",
        )
        all_best_values.extend(run.best_so_far)
    ax.axvline(20, color="#475569", linestyle=":", linewidth=1.5)
    phase_label_style = {
        "transform": ax.get_xaxis_transform(),
        "ha": "center",
        "va": "top",
        "fontsize": 9.5,
        "bbox": {"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.0},
    }
    ax.text(10.5, 0.97, "Matched initialization", color="#475569", **phase_label_style)
    ax.text(25.0, 0.97, "Evolution", color="#92400e", **phase_label_style)
    ax.set_title("RE vs SA-RE pilot: best-so-far validation accuracy")
    ax.set_xlabel("Real CNN training budget")
    ax.set_ylabel("Best-so-far validation accuracy")
    ax.set_xlim(1, 30)
    lower = max(0.0, min(all_best_values) - 0.02)
    upper = min(1.0, max(all_best_values) + 0.02)
    ax.set_ylim(lower, upper)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.grid(axis="y", color="#cbd5e1", alpha=0.65, linewidth=0.8)
    ax.legend(loc="lower right", frameon=True, ncol=2)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def summarize_pilots(
    pilot_root: Path | str,
    *,
    output_dir: Path | str | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Validate four matched pilots and write the Part C outputs."""

    pilot_root = Path(pilot_root)
    output_dir = Path(output_dir) if output_dir is not None else pilot_root
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "pilot_comparison_summary.csv"
    plot_path = output_dir / "pilot_best_so_far_diagnostic.png"
    audit_path = output_dir / "pilot_comparison_audit.json"
    for path in (summary_path, plot_path, audit_path):
        if path.exists() and not overwrite:
            raise ComparisonError(f"output already exists: {path}; use overwrite=True")

    runs = tuple(_load_run(pilot_root, *specification) for specification in EXPECTED_RUNS)
    run_map = {(run.method, run.search_seed): run for run in runs}
    pair_audits: dict[str, object] = {}
    all_pair_gates: list[bool] = []
    candidate_audits: dict[str, object] = {}
    all_candidate_gates: list[bool] = []
    for seed in (2701, 2702):
        re_run = run_map[("RE", seed)]
        sa_run = run_map[("SA-RE", seed)]
        config_checks = _shared_config_checks(re_run, sa_run)
        initialization = _matched_initialization(re_run, sa_run)
        pair_gate = all(config_checks.values()) and bool(initialization["hard_gate_pass"])
        pair_audits[str(seed)] = {
            "config_checks": config_checks,
            "matched_initialization": initialization,
            "hard_gate_pass": pair_gate,
        }
        all_pair_gates.append(pair_gate)
        candidate_audit = _candidate_checks(sa_run)
        candidate_audits[str(seed)] = candidate_audit
        all_candidate_gates.append(bool(candidate_audit["hard_gate_pass"]))

    run_checks = {
        f"{run.method}_{run.search_seed}": run.artifact_checks for run in runs
    }
    all_run_gates = [all(checks.values()) for checks in run_checks.values()]
    overall_checks = {
        "all_four_expected_runs_present": len(runs) == 4,
        "all_runs_complete_and_budget_exact": all(all_run_gates),
        "matched_initial_conditions": all(all_pair_gates),
        "same_dataset_network_trainer_and_core_evolution_config": all(all_pair_gates),
        "fifo_aging_artifacts_valid": all(
            checks["fifo_final_population_order_valid"] for checks in run_checks.values()
        ),
        "k5_candidate_screening_valid": all(all_candidate_gates),
        "one_selected_child_consumes_each_evolution_budget_step": all(
            bool(audit["exactly_one_selected_per_step"])
            for audit in candidate_audits.values()
        ),
        "surrogate_predictions_finite": all(
            bool(audit["predictions_all_finite"])
            for audit in candidate_audits.values()
        ),
        "logs_complete": all(checks["run_log_complete"] for checks in run_checks.values()),
    }
    overall_pass = all(overall_checks.values())
    audit: dict[str, object] = {
        "schema_version": 1,
        "audit_name": "RE vs SA-RE Pilot Comparison",
        "paired_search_seeds": [2701, 2702],
        "performance_superiority_required_for_pass": False,
        "performance_claim_limit": (
            "These n=2, budget=30 pilots are implementation diagnostics only. "
            "They do not support a claim that SA-RE significantly outperforms RE."
        ),
        "artifact_scope_note": (
            "Parent-selection and mutation-function code identity remain covered by "
            "the existing implementation regression tests; this audit verifies their "
            "recorded configurations and resulting pilot artifacts."
        ),
        "run_checks": run_checks,
        "matched_pair_checks": pair_audits,
        "sa_re_candidate_checks": candidate_audits,
        "overall_checks": overall_checks,
        "result": "PASS" if overall_pass else "FAIL",
    }
    if not overall_pass:
        failed = [name for name, passed in overall_checks.items() if not passed]
        raise ComparisonError(f"pilot comparison audit failed: {', '.join(failed)}")

    _write_summary(summary_path, runs)
    _write_plot(plot_path, runs)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-root",
        type=Path,
        default=Path("experiments/pilot"),
        help="Directory containing re_2701, re_2702, sa_re_2701, sa_re_2702",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory; defaults to --pilot-root",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        audit = summarize_pilots(
            args.pilot_root,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except ComparisonError as error:
        raise SystemExit(f"Pilot comparison failed: {error}") from error
    print("RE vs SA-RE Pilot Comparison")
    print("Matched seeds: 2701, 2702")
    print(f"Result:        {audit['result']}")
    print("Claim scope:   implementation diagnostic only (n=2, budget=30)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
