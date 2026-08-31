#!/usr/bin/env python3
"""Build offline RS-SA-RE pilot diagnostics without running CNN training.

Inputs (read-only):
  experiments/pilot/rs_sa_re_2701/evaluations.csv
  experiments/pilot/rs_sa_re_2701/candidate_predictions.csv
  experiments/pilot/rs_sa_re_2702/evaluations.csv
  experiments/pilot/rs_sa_re_2702/candidate_predictions.csv

The script extracts the five real repeat pairs per seed, summarises the true
instability targets, analyses candidate mu/d scale, and recomputes selection at
lambda in {0, 0.5, 1, 2}. It never imports the trainer or starts a CNN run.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


SEARCH_SEEDS = (2701, 2702)
LAMBDA_VALUES = (0.0, 0.5, 1.0, 2.0)
DEFAULT_K = 5
DEFAULT_PAIRS_PER_SEED = 5
DEFAULT_STEPS_PER_SEED = 5
DEFAULT_EPSILON = 1e-12

LABEL_COLUMNS = (
    "search_seed",
    "base_evaluation_index",
    "accuracy_seed_1",
    "accuracy_seed_2",
    "mean_target",
    "instability_target",
)

SUMMARY_COLUMNS = ("group", "n", "mean", "median", "std", "min", "max", "q1", "q3", "iqr")

DIAGNOSTIC_COLUMNS = (
    "search_seed",
    "budget",
    "paired_label_count",
    "true_d_mean",
    "true_d_std",
    "true_d_min",
    "true_d_max",
    "candidate_mu_mean",
    "candidate_mu_std",
    "candidate_mu_range",
    "candidate_d_mean",
    "candidate_d_std",
    "candidate_d_range",
    "score_mean",
    "score_std",
    "lambda",
    "penalty_to_mu_range_ratio",
)

SENSITIVITY_COLUMNS = (
    "search_seed",
    "budget",
    "candidate_count",
    "argmax_mu",
    "argmax_lambda_0",
    "argmax_lambda_0_5",
    "argmax_lambda_1",
    "argmax_lambda_2",
    "changed_lambda_0_5",
    "changed_lambda_1",
    "changed_lambda_2",
    "pilot_selected_candidate",
)

SENSITIVITY_SUMMARY_COLUMNS = (
    "group",
    "lambda",
    "search_steps",
    "ranking_change_count",
    "ranking_change_frequency",
)

INDEX_ALIASES = (
    "evaluation_index",
    "real_evaluation_index",
    "real_evaluation",
    "training_run_index",
    "budget_index",
    "eval_index",
    "budget",
)
ACCURACY_ALIASES = (
    "accuracy",
    "val_accuracy",
    "final_val_accuracy",
    "validation_accuracy",
    "fitness",
    "acc",
)
EVENT_ALIASES = ("event", "event_type", "evaluation_type", "kind")
PHASE_ALIASES = ("phase", "stage", "search_phase")
REPEAT_ALIASES = ("is_repeat", "repeat", "repeated")
INSERTED_ALIASES = ("inserted", "population_inserted", "entered_population", "is_inserted")
BASE_INDEX_ALIASES = (
    "base_evaluation_index",
    "base_eval",
    "repeated_evaluation_index",
    "repeat_of_evaluation_index",
    "paired_with_evaluation",
)

GROUP_ALIASES = (
    "budget",
    "real_budget",
    "evaluation_budget",
    "target_budget",
    "selection_budget",
    "search_budget",
    "search_step",
    "evolution_step",
    "step",
)
CANDIDATE_ID_ALIASES = (
    "candidate_index",
    "candidate_id",
    "candidate",
    "index",
    "k_index",
)
MU_ALIASES = (
    "predicted_mu",
    "mu_hat",
    "mu",
    "predicted_mean",
    "predicted_accuracy",
    "prediction",
)
D_ALIASES = (
    "predicted_d",
    "d_hat",
    "d",
    "predicted_instability",
    "instability_prediction",
)
SCORE_ALIASES = ("score", "predicted_score", "selection_score")
SELECTED_ALIASES = ("selected", "is_selected", "chosen", "selected_candidate")
PAIRED_COUNT_ALIASES = (
    "paired_label_count",
    "pair_count",
    "instability_label_count",
    "d_label_count",
)


@dataclass(frozen=True)
class Evaluation:
    real_index: int
    accuracy: float
    is_repeat: bool
    inserted: bool | None
    base_evaluation_index: int | None


@dataclass(frozen=True)
class LabelPair:
    search_seed: int
    base_evaluation_index: int
    accuracy_seed_1: float
    accuracy_seed_2: float
    mean_target: float
    instability_target: float
    repeat_real_index: int


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    predicted_mu: float
    predicted_d: float
    saved_score: float | None
    selected: bool | None
    paired_label_count: int | None


def normalise_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def normalise_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {normalise_key(str(key)): value for key, value in record.items()}


def pick(record: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    for key in aliases:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def as_float(value: Any, *, field: str, required: bool = False) -> float | None:
    if value in (None, ""):
        if required:
            raise ValueError(f"missing required field {field}")
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"non-finite {field}: {value!r}")
    return result


def as_int(value: Any, *, field: str, required: bool = False) -> int | None:
    number = as_float(value, field=field, required=required)
    if number is None:
        return None
    if not number.is_integer():
        raise ValueError(f"{field} must be integral, got {value!r}")
    return int(number)


def as_bool(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "t"}:
        return True
    if text in {"0", "false", "no", "n", "f"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing required input: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [normalise_record(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"no data rows in {path}")
    return rows


def select_real_index(rows: Sequence[Mapping[str, Any]]) -> tuple[str | None, int]:
    """Ignore a repeated total-budget field and select a true row index."""

    for key in INDEX_ALIASES:
        values: list[int] = []
        for row in rows:
            value = as_int(row.get(key), field=key)
            if value is None:
                values = []
                break
            values.append(value)
        if len(values) != len(rows):
            continue
        if sorted(values) == list(range(1, len(rows) + 1)):
            return key, 0
        if sorted(values) == list(range(len(rows))):
            return key, 1
    return None, 0


def classify_repeat(row: Mapping[str, Any]) -> bool:
    if as_bool(pick(row, REPEAT_ALIASES)) is True:
        return True
    text = f"{pick(row, EVENT_ALIASES) or ''} {pick(row, PHASE_ALIASES) or ''}".lower()
    if "repeat" in text or "retrain" in text:
        return True
    inserted = as_bool(pick(row, INSERTED_ALIASES))
    base_index = pick(row, BASE_INDEX_ALIASES)
    return inserted is False and base_index not in (None, "")


def parse_evaluations(path: Path) -> list[Evaluation]:
    rows = load_csv(path)
    index_key, index_offset = select_real_index(rows)
    evaluations: list[Evaluation] = []

    for ordinal, row in enumerate(rows, start=1):
        if index_key is None:
            real_index = ordinal
        else:
            raw_index = as_int(row[index_key], field=index_key, required=True)
            assert raw_index is not None
            real_index = raw_index + index_offset

        accuracy = as_float(pick(row, ACCURACY_ALIASES), field="accuracy", required=True)
        assert accuracy is not None
        if not 0.0 <= accuracy <= 1.0:
            raise ValueError(f"{path}: accuracy outside [0,1] at real index {real_index}")

        base_index = as_int(pick(row, BASE_INDEX_ALIASES), field="base_evaluation_index")
        if base_index is not None and index_offset:
            base_index += index_offset
        evaluations.append(
            Evaluation(
                real_index=real_index,
                accuracy=accuracy,
                is_repeat=classify_repeat(row),
                inserted=as_bool(pick(row, INSERTED_ALIASES)),
                base_evaluation_index=base_index,
            )
        )

    evaluations.sort(key=lambda item: item.real_index)
    indices = [item.real_index for item in evaluations]
    if indices != list(range(1, len(evaluations) + 1)):
        raise ValueError(f"{path}: real evaluation indices are not contiguous 1..{len(evaluations)}")
    return evaluations


def make_label_pair(
    search_seed: int,
    base_evaluation_index: int,
    accuracy_seed_1: float,
    accuracy_seed_2: float,
    repeat_real_index: int,
) -> LabelPair:
    return LabelPair(
        search_seed=search_seed,
        base_evaluation_index=base_evaluation_index,
        accuracy_seed_1=accuracy_seed_1,
        accuracy_seed_2=accuracy_seed_2,
        mean_target=(accuracy_seed_1 + accuracy_seed_2) / 2.0,
        instability_target=abs(accuracy_seed_1 - accuracy_seed_2),
        repeat_real_index=repeat_real_index,
    )


def extract_label_pairs(
    evaluations: Sequence[Evaluation], search_seed: int, expected_pairs: int
) -> list[LabelPair]:
    first_by_index = {item.real_index: item for item in evaluations if not item.is_repeat}
    pairs: list[LabelPair] = []
    for repeated in (item for item in evaluations if item.is_repeat):
        if repeated.base_evaluation_index is None:
            raise ValueError(
                f"RS-SA-RE {search_seed}: repeat at real index {repeated.real_index} "
                "has no base_evaluation_index"
            )
        base = first_by_index.get(repeated.base_evaluation_index)
        if base is None:
            raise ValueError(
                f"RS-SA-RE {search_seed}: repeat at {repeated.real_index} refers to "
                f"missing/non-first base {repeated.base_evaluation_index}"
            )
        pairs.append(
            make_label_pair(
                search_seed,
                base.real_index,
                base.accuracy,
                repeated.accuracy,
                repeated.real_index,
            )
        )
    pairs.sort(key=lambda item: item.repeat_real_index)
    if len(pairs) != expected_pairs:
        raise ValueError(
            f"RS-SA-RE {search_seed}: found {len(pairs)} real repeat pairs; "
            f"expected {expected_pairs}"
        )
    return pairs


def percentile(values: Sequence[float], probability: float) -> float:
    """Linear percentile equivalent to the common (n-1)*p definition."""

    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate percentile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def describe(values: Sequence[float], group: str | int) -> dict[str, Any]:
    if not values:
        raise ValueError(f"no values for group {group}")
    q1 = percentile(values, 0.25)
    q3 = percentile(values, 0.75)
    return {
        "group": str(group),
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
    }


def _candidate_id_sort_key(value: str) -> tuple[int, float | str]:
    try:
        return 0, float(value)
    except ValueError:
        return 1, value


def split_candidate_groups(
    rows: Sequence[Mapping[str, Any]], expected_steps: int, candidate_count: int
) -> list[list[Mapping[str, Any]]]:
    for key in GROUP_ALIASES:
        if any(row.get(key) in (None, "") for row in rows):
            continue
        groups: dict[str, list[Mapping[str, Any]]] = {}
        first_seen: dict[str, int] = {}
        for position, row in enumerate(rows):
            value = str(row[key]).strip()
            groups.setdefault(value, []).append(row)
            first_seen.setdefault(value, position)
        if len(groups) == expected_steps and all(len(group) == candidate_count for group in groups.values()):
            def group_sort(value: str) -> tuple[int, float | int]:
                try:
                    return 0, float(value)
                except ValueError:
                    return 1, first_seen[value]

            return [groups[value] for value in sorted(groups, key=group_sort)]

    expected_rows = expected_steps * candidate_count
    if len(rows) != expected_rows:
        raise ValueError(
            f"candidate_predictions.csv has {len(rows)} rows; expected {expected_rows} "
            "and no usable group column was found"
        )
    return [
        list(rows[start : start + candidate_count])
        for start in range(0, len(rows), candidate_count)
    ]


def parse_candidate_group(rows: Sequence[Mapping[str, Any]]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for ordinal, row in enumerate(rows, start=1):
        candidate_id = str(pick(row, CANDIDATE_ID_ALIASES) or ordinal).strip()
        mu = as_float(pick(row, MU_ALIASES), field="predicted_mu", required=True)
        predicted_d = as_float(pick(row, D_ALIASES), field="predicted_d", required=True)
        assert mu is not None and predicted_d is not None
        candidates.append(
            Candidate(
                candidate_id=candidate_id,
                predicted_mu=mu,
                predicted_d=predicted_d,
                saved_score=as_float(pick(row, SCORE_ALIASES), field="saved_score"),
                selected=as_bool(pick(row, SELECTED_ALIASES)),
                paired_label_count=as_int(
                    pick(row, PAIRED_COUNT_ALIASES), field="paired_label_count"
                ),
            )
        )
    ids = [item.candidate_id for item in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate candidate ids in one K-set: {ids}")
    return sorted(candidates, key=lambda item: _candidate_id_sort_key(item.candidate_id))


def choose_candidate(candidates: Sequence[Candidate], lambda_value: float) -> Candidate:
    """Argmax with deterministic lowest-candidate-id tie breaking."""

    if not candidates:
        raise ValueError("empty candidate set")
    ordered = sorted(candidates, key=lambda item: _candidate_id_sort_key(item.candidate_id))
    return min(
        ordered,
        key=lambda item: (-(item.predicted_mu - lambda_value * item.predicted_d), _candidate_id_sort_key(item.candidate_id)),
    )


def range_ratio(mu_values: Sequence[float], d_values: Sequence[float], lambda_value: float, epsilon: float) -> float:
    mu_range = max(mu_values) - min(mu_values)
    d_range = max(d_values) - min(d_values)
    return lambda_value * d_range / (mu_range + epsilon)


def _sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def analyse_candidate_sets(
    search_seed: int,
    evaluations: Sequence[Evaluation],
    labels: Sequence[LabelPair],
    candidate_path: Path,
    lambda_value: float,
    epsilon: float,
    candidate_count: int,
    expected_steps: int,
    score_tolerance: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows = load_csv(candidate_path)
    groups = split_candidate_groups(raw_rows, expected_steps, candidate_count)
    evolution_budgets = [item.real_index for item in evaluations if not item.is_repeat][20:]
    if len(evolution_budgets) != expected_steps:
        raise ValueError(
            f"RS-SA-RE {search_seed}: found {len(evolution_budgets)} evolutionary "
            f"first evaluations; expected {expected_steps}"
        )

    diagnostics: list[dict[str, Any]] = []
    sensitivities: list[dict[str, Any]] = []
    for group_rows, budget in zip(groups, evolution_budgets):
        candidates = parse_candidate_group(group_rows)
        explicit_counts = {
            item.paired_label_count
            for item in candidates
            if item.paired_label_count is not None
        }
        if len(explicit_counts) > 1:
            raise ValueError(
                f"RS-SA-RE {search_seed} budget {budget}: inconsistent paired_label_count"
            )
        paired_count = (
            next(iter(explicit_counts))
            if explicit_counts
            else sum(item.repeat_real_index < budget for item in labels)
        )
        if paired_count is None or not 1 <= paired_count <= len(labels):
            raise ValueError(
                f"RS-SA-RE {search_seed} budget {budget}: invalid paired_label_count={paired_count}"
            )

        true_d = [item.instability_target for item in labels[:paired_count]]
        mu_values = [item.predicted_mu for item in candidates]
        d_values = [item.predicted_d for item in candidates]
        scores = [item.predicted_mu - lambda_value * item.predicted_d for item in candidates]

        if math.isclose(lambda_value, 1.0, abs_tol=1e-12):
            for candidate, recomputed in zip(candidates, scores):
                if candidate.saved_score is not None and not math.isclose(
                    candidate.saved_score, recomputed, rel_tol=0.0, abs_tol=score_tolerance
                ):
                    raise ValueError(
                        f"RS-SA-RE {search_seed} budget {budget} candidate "
                        f"{candidate.candidate_id}: saved score {candidate.saved_score} "
                        f"!= recomputed {recomputed}"
                    )

        mu_argmax = choose_candidate(candidates, 0.0).candidate_id
        choices = {
            value: choose_candidate(candidates, value).candidate_id
            for value in LAMBDA_VALUES
        }
        selected = [item.candidate_id for item in candidates if item.selected is True]
        if len(selected) > 1:
            raise ValueError(
                f"RS-SA-RE {search_seed} budget {budget}: multiple selected candidates {selected}"
            )

        diagnostics.append(
            {
                "search_seed": search_seed,
                "budget": budget,
                "paired_label_count": paired_count,
                "true_d_mean": statistics.fmean(true_d),
                "true_d_std": _sample_std(true_d),
                "true_d_min": min(true_d),
                "true_d_max": max(true_d),
                "candidate_mu_mean": statistics.fmean(mu_values),
                "candidate_mu_std": _sample_std(mu_values),
                "candidate_mu_range": max(mu_values) - min(mu_values),
                "candidate_d_mean": statistics.fmean(d_values),
                "candidate_d_std": _sample_std(d_values),
                "candidate_d_range": max(d_values) - min(d_values),
                "score_mean": statistics.fmean(scores),
                "score_std": _sample_std(scores),
                "lambda": lambda_value,
                "penalty_to_mu_range_ratio": range_ratio(
                    mu_values, d_values, lambda_value, epsilon
                ),
            }
        )
        sensitivities.append(
            {
                "search_seed": search_seed,
                "budget": budget,
                "candidate_count": len(candidates),
                "argmax_mu": mu_argmax,
                "argmax_lambda_0": choices[0.0],
                "argmax_lambda_0_5": choices[0.5],
                "argmax_lambda_1": choices[1.0],
                "argmax_lambda_2": choices[2.0],
                "changed_lambda_0_5": choices[0.5] != mu_argmax,
                "changed_lambda_1": choices[1.0] != mu_argmax,
                "changed_lambda_2": choices[2.0] != mu_argmax,
                "pilot_selected_candidate": selected[0] if selected else "",
            }
        )

    return diagnostics, sensitivities


def build_sensitivity_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    group_rows: list[tuple[str, list[Mapping[str, Any]]]] = []
    for seed in SEARCH_SEEDS:
        group_rows.append((str(seed), [row for row in rows if int(row["search_seed"]) == seed]))
    group_rows.append(("combined", list(rows)))

    column_for_lambda = {
        0.0: None,
        0.5: "changed_lambda_0_5",
        1.0: "changed_lambda_1",
        2.0: "changed_lambda_2",
    }
    for group, selected_rows in group_rows:
        for lambda_value in LAMBDA_VALUES:
            column = column_for_lambda[lambda_value]
            changes = 0 if column is None else sum(bool(row[column]) for row in selected_rows)
            output.append(
                {
                    "group": group,
                    "lambda": lambda_value,
                    "search_steps": len(selected_rows),
                    "ranking_change_count": changes,
                    "ranking_change_frequency": changes / len(selected_rows),
                }
            )
    return output


def _format(value: Any) -> Any:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.10g}"
    if value is None:
        return ""
    return value


def write_csv_atomic(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({column: _format(row.get(column)) for column in columns})
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def render_notes(
    label_summaries: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    sensitivity_summary: Sequence[Mapping[str, Any]],
    lambda_value: float,
    epsilon: float,
) -> str:
    combined = next(row for row in label_summaries if row["group"] == "combined")
    true_min = float(combined["min"])
    true_max = float(combined["max"])
    candidate_d_means = [float(row["candidate_d_mean"]) for row in diagnostics]
    ratios = [float(row["penalty_to_mu_range_ratio"]) for row in diagnostics]

    lines = [
        "# RS-SA-RE Pilot Diagnostics",
        "",
        "## Scope and data integrity",
        "",
        "This is an offline analysis of the frozen RS-SA-RE pilots for matched search seeds 2701 and 2702. No CNN training was run. The analysis contains 5 paired repeat labels per seed (10 combined) and 5 candidate-selection steps per seed (10 combined).",
        "",
        "`instability_target = abs(accuracy_seed_1 - accuracy_seed_2)` and `mean_target = (accuracy_seed_1 + accuracy_seed_2) / 2`.",
        "",
        "## True instability-label statistics",
        "",
        "| Group | n | Mean | Median | Sample SD | Min | Max | IQR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in label_summaries:
        lines.append(
            f"| {row['group']} | {row['n']} | {float(row['mean']):.6f} | "
            f"{float(row['median']):.6f} | {float(row['std']):.6f} | "
            f"{float(row['min']):.6f} | {float(row['max']):.6f} | "
            f"{float(row['iqr']):.6f} |"
        )

    lines.extend(
        [
            "",
            "## Surrogate scale diagnostic",
            "",
            f"Across the 10 candidate sets, the true instability targets range from {true_min:.6f} to {true_max:.6f}. Candidate-set mean predicted instability ranges from {min(candidate_d_means):.6f} to {max(candidate_d_means):.6f}.",
            "",
            f"For each K=5 set, the ranking-force ratio is `R = lambda * range(d_hat) / (range(mu_hat) + epsilon)`, with `lambda={lambda_value:g}` and `epsilon={epsilon:g}`. Observed R ranges from {min(ratios):.6f} to {max(ratios):.6f}, with median {statistics.median(ratios):.6f}.",
            "",
            "A small training-set instability MSE does not demonstrate calibrated candidate predictions. Only 4–5 paired labels are available at a selection step, and the candidate architectures are out-of-sample surrogate inputs.",
            "",
            "## Offline lambda sensitivity",
            "",
            "| Group | Lambda | Steps | Ranking changes | Frequency |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in sensitivity_summary:
        lines.append(
            f"| {row['group']} | {float(row['lambda']):g} | {row['search_steps']} | "
            f"{row['ranking_change_count']} | {float(row['ranking_change_frequency']):.3f} |"
        )

    lines.extend(
        [
            "",
            "`lambda=0` is an invariant check and must always select the same candidate as `argmax(mu_hat)`.",
            "",
            "## Interpretation constraint",
            "",
            "Offline lambda sensitivity is a mechanism diagnostic, not performance tuning. Only the selected candidate has a real CNN accuracy; the other four candidates in each set have no outcome ground truth. A ranking change therefore shows that the stability penalty has a nonzero selection effect, but it cannot show that one lambda gives better performance.",
            "",
            "## Provisional decision",
            "",
            "Unless the CSV diagnostics reveal numerical collapse, NaN/Inf values, or complete domination by the instability prediction, retain `lambda=1.0` provisionally until the 8/31 formal freeze. The justification is its unit interpretation: a predicted instability increase of 0.01 receives the same penalty as a predicted mean-accuracy decrease of 0.01. This is not an outcome-driven retuning decision.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, default=Path("experiments/pilot"))
    parser.add_argument("--lambda-value", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--candidate-count", type=int, default=DEFAULT_K)
    parser.add_argument("--pairs-per-seed", type=int, default=DEFAULT_PAIRS_PER_SEED)
    parser.add_argument("--steps-per-seed", type=int, default=DEFAULT_STEPS_PER_SEED)
    parser.add_argument("--score-tolerance", type=float, default=1e-5)
    parser.add_argument("--notes-output", type=Path, default=Path("notes/rs_sa_re_pilot_diagnostics.md"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not math.isfinite(args.lambda_value):
        raise ValueError("lambda-value must be finite")

    all_labels: list[LabelPair] = []
    all_diagnostics: list[dict[str, Any]] = []
    all_sensitivity: list[dict[str, Any]] = []
    seed_label_values: dict[int, list[float]] = {}

    for search_seed in SEARCH_SEEDS:
        run_dir = args.pilot_root / f"rs_sa_re_{search_seed}"
        evaluations = parse_evaluations(run_dir / "evaluations.csv")
        labels = extract_label_pairs(evaluations, search_seed, args.pairs_per_seed)
        diagnostics, sensitivity = analyse_candidate_sets(
            search_seed=search_seed,
            evaluations=evaluations,
            labels=labels,
            candidate_path=run_dir / "candidate_predictions.csv",
            lambda_value=args.lambda_value,
            epsilon=args.epsilon,
            candidate_count=args.candidate_count,
            expected_steps=args.steps_per_seed,
            score_tolerance=args.score_tolerance,
        )
        all_labels.extend(labels)
        all_diagnostics.extend(diagnostics)
        all_sensitivity.extend(sensitivity)
        seed_label_values[search_seed] = [item.instability_target for item in labels]

    expected_labels = len(SEARCH_SEEDS) * args.pairs_per_seed
    expected_steps = len(SEARCH_SEEDS) * args.steps_per_seed
    if len(all_labels) != expected_labels or len(all_diagnostics) != expected_steps:
        raise AssertionError("combined diagnostic row-count invariant failed")

    label_summaries = [describe(seed_label_values[seed], seed) for seed in SEARCH_SEEDS]
    label_summaries.append(
        describe([item.instability_target for item in all_labels], "combined")
    )
    sensitivity_summary = build_sensitivity_summary(all_sensitivity)

    label_rows = [
        {
            "search_seed": item.search_seed,
            "base_evaluation_index": item.base_evaluation_index,
            "accuracy_seed_1": item.accuracy_seed_1,
            "accuracy_seed_2": item.accuracy_seed_2,
            "mean_target": item.mean_target,
            "instability_target": item.instability_target,
        }
        for item in all_labels
    ]

    outputs = (
        (args.pilot_root / "rs_sa_re_instability_labels.csv", LABEL_COLUMNS, label_rows),
        (args.pilot_root / "rs_sa_re_instability_summary.csv", SUMMARY_COLUMNS, label_summaries),
        (args.pilot_root / "rs_sa_re_surrogate_diagnostic.csv", DIAGNOSTIC_COLUMNS, all_diagnostics),
        (args.pilot_root / "rs_sa_re_lambda_sensitivity.csv", SENSITIVITY_COLUMNS, all_sensitivity),
        (
            args.pilot_root / "rs_sa_re_lambda_sensitivity_summary.csv",
            SENSITIVITY_SUMMARY_COLUMNS,
            sensitivity_summary,
        ),
    )
    for path, columns, rows in outputs:
        write_csv_atomic(path, columns, rows)
        print(f"WROTE {path} ({len(rows)} rows)")

    notes = render_notes(
        label_summaries,
        all_diagnostics,
        sensitivity_summary,
        args.lambda_value,
        args.epsilon,
    )
    write_text_atomic(args.notes_output, notes)
    print(f"WROTE {args.notes_output}")
    print("RS-SA-RE offline pilot diagnostics: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

