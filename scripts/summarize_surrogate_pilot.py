"""Create online surrogate diagnostics for one completed SA-RE pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev, stdev
from typing import Iterable, Sequence


class DiagnosticError(ValueError):
    """Raised when a pilot artifact violates a diagnostic invariant."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise DiagnosticError(f"missing input file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise DiagnosticError(f"input CSV is empty: {path}")
    return rows


def _require_columns(
    rows: Sequence[dict[str, str]], required: Iterable[str], path: Path
) -> None:
    missing = sorted(set(required) - set(rows[0]))
    if missing:
        raise DiagnosticError(
            f"{path} is missing required columns: {', '.join(missing)}"
        )


def _integer(value: str, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise DiagnosticError(f"{label} must be an integer, got {value!r}") from error


def _finite_float(value: str, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise DiagnosticError(f"{label} must be numeric, got {value!r}") from error
    if not math.isfinite(result):
        raise DiagnosticError(f"{label} must be finite, got {value!r}")
    return result


def _selected(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise DiagnosticError(f"selected must be boolean, got {value!r}")


def _unique(values: Iterable[str], label: str) -> str:
    unique_values = {str(value).strip() for value in values}
    if len(unique_values) != 1:
        raise DiagnosticError(f"expected one {label}, got {sorted(unique_values)!r}")
    return unique_values.pop()


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_ss * right_ss)
    return None if denominator == 0.0 else numerator / denominator


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[ordered[position][0]] = average_rank
        start = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    return _pearson(_average_ranks(left), _average_ranks(right))


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames = [
        "evaluation_index",
        "predicted_mu",
        "observed_accuracy",
        "error",
        "absolute_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize_experiment(
    experiment_dir: Path | str, *, overwrite: bool = False
) -> dict[str, object]:
    """Validate one SA-RE run and write its CSV/JSON diagnostic artifacts."""

    experiment_dir = Path(experiment_dir)
    evaluations_path = experiment_dir / "evaluations.csv"
    candidates_path = experiment_dir / "candidate_predictions.csv"
    online_path = experiment_dir / "surrogate_online_diagnostic.csv"
    summary_path = experiment_dir / "surrogate_diagnostic_summary.json"

    for output_path in (online_path, summary_path):
        if output_path.exists() and not overwrite:
            raise DiagnosticError(
                f"output already exists: {output_path}; pass overwrite=True"
            )

    evaluations = _read_csv(evaluations_path)
    candidates = _read_csv(candidates_path)
    _require_columns(
        evaluations,
        {
            "method",
            "search_seed",
            "evaluation_index",
            "budget",
            "phase",
            "mutation_type",
            "observed_final_val_accuracy",
            "predicted_mu_before_training",
            "surrogate_training_size",
            "selected_candidate_index",
        },
        evaluations_path,
    )
    _require_columns(
        candidates,
        {
            "method",
            "search_seed",
            "evaluation_index",
            "candidate_index",
            "mutation_type",
            "predicted_mu",
            "selected",
            "surrogate_training_size",
        },
        candidates_path,
    )

    method = _unique((row["method"] for row in evaluations), "method")
    if method != "SA-RE":
        raise DiagnosticError(f"method must be SA-RE, got {method!r}")
    search_seed = _integer(
        _unique((row["search_seed"] for row in evaluations), "search seed"),
        "search seed",
    )
    candidate_method = _unique((row["method"] for row in candidates), "candidate method")
    candidate_seed = _integer(
        _unique((row["search_seed"] for row in candidates), "candidate search seed"),
        "candidate search seed",
    )
    if candidate_method != method or candidate_seed != search_seed:
        raise DiagnosticError("candidate CSV method/search seed does not match evaluations")

    budget = _integer(
        _unique((row["budget"] for row in evaluations), "budget"), "budget"
    )
    if len(evaluations) != budget:
        raise DiagnosticError(
            f"real evaluation count must equal budget ({budget}), got {len(evaluations)}"
        )
    evaluations_by_index: dict[int, dict[str, str]] = {}
    observed_targets: list[float] = []
    initialization_indices: list[int] = []
    evolution_indices: list[int] = []
    for row in evaluations:
        index = _integer(row["evaluation_index"], "evaluation_index")
        if index in evaluations_by_index:
            raise DiagnosticError(f"duplicate evaluation_index: {index}")
        evaluations_by_index[index] = row
        target = _finite_float(
            row["observed_final_val_accuracy"],
            f"observed_final_val_accuracy at evaluation {index}",
        )
        if not 0.0 <= target <= 1.0:
            raise DiagnosticError(
                "target accuracy must use the 0.0-1.0 scale; "
                f"evaluation {index} has {target}"
            )
        observed_targets.append(target)
        if row["phase"] == "initialization":
            initialization_indices.append(index)
        elif row["phase"] == "evolution":
            evolution_indices.append(index)
        else:
            raise DiagnosticError(f"unknown phase at evaluation {index}: {row['phase']!r}")

    if sorted(evaluations_by_index) != list(range(1, budget + 1)):
        raise DiagnosticError("evaluation indices must be contiguous from 1 through budget")
    population_size = len(initialization_indices)
    if initialization_indices != list(range(1, population_size + 1)):
        raise DiagnosticError("initialization evaluations must precede evolution")
    expected_evolution_indices = list(range(population_size + 1, budget + 1))
    if evolution_indices != expected_evolution_indices:
        raise DiagnosticError("evolution evaluation indices are inconsistent with budget")

    candidate_groups: dict[int, list[dict[str, str]]] = defaultdict(list)
    all_predictions: list[float] = []
    for row in candidates:
        index = _integer(row["evaluation_index"], "candidate evaluation_index")
        candidate_groups[index].append(row)
        all_predictions.append(
            _finite_float(
                row["predicted_mu"],
                f"predicted_mu at evaluation {index}, candidate {row['candidate_index']}",
            )
        )
    if sorted(candidate_groups) != expected_evolution_indices:
        raise DiagnosticError("candidate groups must cover every evolution evaluation exactly")
    group_sizes = {len(group) for group in candidate_groups.values()}
    if len(group_sizes) != 1:
        raise DiagnosticError(f"candidate count varies across evaluations: {sorted(group_sizes)}")
    candidate_count = group_sizes.pop()
    if candidate_count <= 0:
        raise DiagnosticError("candidate count must be positive")

    prediction_time_sizes: list[int] = []
    online_rows: list[dict[str, object]] = []
    predicted_selected: list[float] = []
    observed_selected: list[float] = []
    for offset, evaluation_index in enumerate(expected_evolution_indices):
        group = candidate_groups[evaluation_index]
        selected_rows = [row for row in group if _selected(row["selected"])]
        if len(selected_rows) != 1:
            raise DiagnosticError(
                f"evaluation {evaluation_index} must have exactly one selected candidate"
            )
        sizes = {
            _integer(row["surrogate_training_size"], "surrogate_training_size")
            for row in group
        }
        if len(sizes) != 1:
            raise DiagnosticError(
                f"surrogate training size varies within evaluation {evaluation_index}"
            )
        training_size = sizes.pop()
        expected_training_size = population_size + offset
        if training_size != expected_training_size:
            raise DiagnosticError(
                f"evaluation {evaluation_index} prediction-time surrogate size must be "
                f"{expected_training_size}, got {training_size}"
            )
        prediction_time_sizes.append(training_size)

        selected_row = selected_rows[0]
        evaluation_row = evaluations_by_index[evaluation_index]
        selected_index = _integer(selected_row["candidate_index"], "candidate_index")
        logged_selected_index = _integer(
            evaluation_row["selected_candidate_index"], "selected_candidate_index"
        )
        if selected_index != logged_selected_index:
            raise DiagnosticError(
                f"selected candidate index mismatch at evaluation {evaluation_index}"
            )
        if selected_row["mutation_type"] != evaluation_row["mutation_type"]:
            raise DiagnosticError(
                f"selected mutation type mismatch at evaluation {evaluation_index}"
            )
        evaluation_training_size = _integer(
            evaluation_row["surrogate_training_size"], "evaluation surrogate_training_size"
        )
        if evaluation_training_size != training_size:
            raise DiagnosticError(
                f"surrogate training size mismatch at evaluation {evaluation_index}"
            )
        predicted = _finite_float(
            selected_row["predicted_mu"],
            f"selected predicted_mu at evaluation {evaluation_index}",
        )
        predicted_before_training = _finite_float(
            evaluation_row["predicted_mu_before_training"],
            f"predicted_mu_before_training at evaluation {evaluation_index}",
        )
        if not math.isclose(predicted, predicted_before_training, rel_tol=0.0, abs_tol=1e-12):
            raise DiagnosticError(
                f"pre-training prediction mismatch at evaluation {evaluation_index}"
            )
        observed = _finite_float(
            evaluation_row["observed_final_val_accuracy"],
            f"observed accuracy at evaluation {evaluation_index}",
        )
        error = predicted - observed
        predicted_selected.append(predicted)
        observed_selected.append(observed)
        online_rows.append(
            {
                "evaluation_index": evaluation_index,
                "predicted_mu": predicted,
                "observed_accuracy": observed,
                "error": error,
                "absolute_error": abs(error),
            }
        )

    errors = [predicted - observed for predicted, observed in zip(predicted_selected, observed_selected)]
    absolute_errors = [abs(error) for error in errors]
    squared_errors = [error * error for error in errors]
    selected_count = len(online_rows)
    expected_candidate_rows = selected_count * candidate_count
    if len(candidates) != expected_candidate_rows:
        raise DiagnosticError(
            f"expected {expected_candidate_rows} candidate rows, got {len(candidates)}"
        )

    summary: dict[str, object] = {
        "schema_version": 1,
        "method": method,
        "search_seed": search_seed,
        "source_files": {
            "evaluations": evaluations_path.name,
            "candidate_predictions": candidates_path.name,
        },
        "budget_sanity": {
            "budget": budget,
            "real_evaluations": len(evaluations),
            "initial_real_evaluations": population_size,
            "offspring_real_evaluations": selected_count,
            "candidate_count_per_step": candidate_count,
            "candidate_rows": len(candidates),
            "selected_candidate_rows": selected_count,
        },
        "target_accuracy": {
            "unit": "fraction_0_to_1",
            "count": len(observed_targets),
            "minimum": min(observed_targets),
            "maximum": max(observed_targets),
            "all_finite": True,
            "all_in_unit_interval": True,
        },
        "prediction_all_candidates": {
            "count": len(all_predictions),
            "minimum": min(all_predictions),
            "maximum": max(all_predictions),
            "mean": mean(all_predictions),
            "population_std": pstdev(all_predictions),
            "sample_std": stdev(all_predictions) if len(all_predictions) > 1 else 0.0,
            "all_finite": True,
            "all_in_unit_interval": all(0.0 <= value <= 1.0 for value in all_predictions),
        },
        "surrogate_training_size": {
            "prediction_time_sizes": prediction_time_sizes,
            "expected_prediction_time_sizes": list(range(population_size, budget)),
            "final_dataset_size_after_last_real_evaluation": budget,
        },
        "selected_online_diagnostic": {
            "count": selected_count,
            "mae": mean(absolute_errors),
            "rmse": math.sqrt(mean(squared_errors)),
            "mean_error_predicted_minus_observed": mean(errors),
            "pearson_correlation": _pearson(predicted_selected, observed_selected),
            "spearman_rank_correlation": _spearman(predicted_selected, observed_selected),
            "interpretation": (
                "Small, selection-biased online diagnostic: only surrogate-selected "
                "candidates receive real evaluations; this is not an unbiased "
                "surrogate generalization estimate."
            ),
        },
    }

    experiment_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(online_path, online_rows)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Directory containing evaluations.csv and candidate_predictions.csv",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing diagnostic output files",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        summary = summarize_experiment(args.experiment_dir, overwrite=args.overwrite)
    except DiagnosticError as error:
        raise SystemExit(f"Surrogate diagnostic failed: {error}") from error

    predictions = summary["prediction_all_candidates"]
    online = summary["selected_online_diagnostic"]
    print("Surrogate Diagnostic")
    print(f"Search seed:       {summary['search_seed']}")
    print(
        "Predictions:       "
        f"n={predictions['count']} "
        f"min={predictions['minimum']:.6f} "
        f"max={predictions['maximum']:.6f} "
        f"mean={predictions['mean']:.6f} "
        f"std={predictions['sample_std']:.6f}"
    )
    print(
        "Selected online:   "
        f"n={online['count']} MAE={online['mae']:.6f} "
        f"RMSE={online['rmse']:.6f} "
        f"Spearman={online['spearman_rank_correlation']:.6f}"
    )
    print(f"Result:            PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
