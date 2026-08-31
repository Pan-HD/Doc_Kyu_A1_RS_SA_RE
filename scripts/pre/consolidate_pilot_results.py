#!/usr/bin/env python3
"""Consolidate the six matched RE/SA-RE/RS-SA-RE pilot runs.

This script treats every row in each run's evaluations.csv as one real CNN
training budget unit.  Repeat evaluations remain on the budget axis, but they
never update the search best because they are not inserted into the population.

Run from the project root:

    python scripts/consolidate_pilot_results.py --pilot-root experiments/pilot
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence


METHOD_DIRS: tuple[tuple[str, int, str], ...] = (
    ("re_2701", 2701, "RE"),
    ("sa_re_2701", 2701, "SA-RE"),
    ("rs_sa_re_2701", 2701, "RS-SA-RE"),
    ("re_2702", 2702, "RE"),
    ("sa_re_2702", 2702, "SA-RE"),
    ("rs_sa_re_2702", 2702, "RS-SA-RE"),
)

SUMMARY_COLUMNS: tuple[str, ...] = (
    "method",
    "search_seed",
    "initial_best",
    "final_best",
    "final_population_best",
    "best_at_budget_20",
    "best_at_budget_25",
    "best_at_budget_30",
    "real_training_runs",
    "first_evaluation_count",
    "repeat_evaluation_count",
    "runtime",
    "mean_training_time",
    "parameter_count_of_best",
)

CURVE_COLUMNS: tuple[str, ...] = (
    "method",
    "search_seed",
    "budget",
    "event_type",
    "inserted",
    "accuracy",
    "search_best_so_far",
)

ALIASES: dict[str, tuple[str, ...]] = {
    "budget": (
        "budget",
        "real_budget",
        "real_evaluation",
        "real_evaluation_index",
        "evaluation",
        "evaluation_index",
        "eval_index",
        "training_run",
        "training_run_index",
        "global_evaluation_index",
    ),
    "accuracy": (
        "accuracy",
        "val_accuracy",
        "final_val_accuracy",
        "validation_accuracy",
        "fitness",
        "acc",
    ),
    "event": ("event", "event_type", "evaluation_type", "kind"),
    "phase": ("phase", "stage", "search_phase"),
    "inserted": (
        "inserted",
        "population_inserted",
        "entered_population",
        "is_inserted",
    ),
    "is_repeat": ("is_repeat", "repeat", "repeated"),
    "base_evaluation_index": (
        "base_evaluation_index",
        "base_eval",
        "repeated_evaluation_index",
    ),
    "training_time": (
        "training_time",
        "training_time_seconds",
        "train_time",
        "duration",
        "duration_seconds",
        "elapsed_seconds",
    ),
    "parameter_count": (
        "parameter_count",
        "parameter_count_of_architecture",
        "params",
        "num_parameters",
        "n_parameters",
    ),
}


@dataclass(frozen=True)
class Evaluation:
    budget: int
    accuracy: float
    event_type: str
    is_repeat: bool
    inserted: bool | None
    training_time: float | None
    parameter_count: int | None

    @property
    def updates_search(self) -> bool:
        """Whether this evaluation may update search best and population."""

        return not self.is_repeat and self.inserted is not False


def _normalise_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _normalise_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {_normalise_key(str(key)): value for key, value in record.items()}


def _pick(record: Mapping[str, Any], logical_name: str) -> Any:
    for key in ALIASES[logical_name]:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _as_float(value: Any, *, field: str, required: bool = False) -> float | None:
    if value in (None, ""):
        if required:
            raise ValueError(f"missing required {field}")
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"non-finite {field}: {value!r}")
    return result


def _as_int(value: Any, *, field: str) -> int | None:
    number = _as_float(value, field=field)
    if number is None:
        return None
    if not number.is_integer():
        raise ValueError(f"{field} must be integral, got {value!r}")
    return int(number)


def _as_bool(value: Any) -> bool | None:
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


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no evaluation rows in {path}")
    return rows


def _find_record_list(value: Any) -> list[dict[str, Any]] | None:
    """Find the most likely list of evaluation dictionaries in a JSON object."""

    if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        normalised = [_normalise_record(item) for item in value]
        if any(_pick(item, "accuracy") is not None for item in normalised):
            return value
    if isinstance(value, dict):
        preferred = ("evaluations", "records", "history", "events", "results")
        for key in preferred:
            if key in value:
                found = _find_record_list(value[key])
                if found is not None:
                    return found
        for child in value.values():
            found = _find_record_list(child)
            if found is not None:
                return found
    return None


def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = _find_record_list(payload)
    if not rows:
        raise ValueError(f"could not find evaluation records in {path}")
    return rows


def load_raw_records(run_dir: Path) -> tuple[list[dict[str, Any]], Path]:
    candidates = (
        run_dir / "evaluations.csv",
        run_dir / "evaluation_log.csv",
        run_dir / "results.csv",
        run_dir / "history.json",
    )
    for path in candidates:
        if path.is_file():
            loader = _load_csv if path.suffix.lower() == ".csv" else _load_json
            return loader(path), path
    raise FileNotFoundError(
        f"{run_dir}: expected evaluations.csv, evaluation_log.csv, "
        "results.csv, or history.json"
    )


def parse_evaluations(raw_rows: Sequence[Mapping[str, Any]], source: Path) -> list[Evaluation]:
    evaluations: list[Evaluation] = []
    for ordinal, raw in enumerate(raw_rows, start=1):
        row = _normalise_record(raw)
        accuracy = _as_float(_pick(row, "accuracy"), field="accuracy", required=True)
        assert accuracy is not None
        if not 0.0 <= accuracy <= 1.0:
            raise ValueError(
                f"{source}: row {ordinal} accuracy must be in [0, 1], got {accuracy}"
            )

        budget = _as_int(_pick(row, "budget"), field="budget") or ordinal
        event = str(_pick(row, "event") or "").strip()
        phase = str(_pick(row, "phase") or "").strip()
        inserted = _as_bool(_pick(row, "inserted"))
        explicit_repeat = _as_bool(_pick(row, "is_repeat"))
        repeat_text = f"{event} {phase}".lower()
        base_index = _pick(row, "base_evaluation_index")

        is_repeat = bool(explicit_repeat)
        is_repeat = is_repeat or "repeat" in repeat_text or "retrain" in repeat_text
        # Fallback for logs that mark repeats only as inserted=False and refer
        # back to an earlier/base evaluation.
        is_repeat = is_repeat or (inserted is False and base_index not in (None, ""))

        event_type = event or ("repeat_evaluation" if is_repeat else "first_evaluation")
        training_time = _as_float(_pick(row, "training_time"), field="training_time")
        parameter_count = _as_int(_pick(row, "parameter_count"), field="parameter_count")

        evaluations.append(
            Evaluation(
                budget=budget,
                accuracy=accuracy,
                event_type=event_type,
                is_repeat=is_repeat,
                inserted=inserted,
                training_time=training_time,
                parameter_count=parameter_count,
            )
        )

    evaluations.sort(key=lambda item: item.budget)
    budgets = [item.budget for item in evaluations]
    if len(budgets) != len(set(budgets)):
        raise ValueError(f"{source}: duplicate budget indices")
    expected = list(range(1, len(evaluations) + 1))
    if budgets != expected:
        raise ValueError(
            f"{source}: budgets must be contiguous 1..{len(evaluations)}, got {budgets}"
        )
    return evaluations


def build_curve(
    evaluations: Sequence[Evaluation], method: str, search_seed: int
) -> list[dict[str, Any]]:
    best: float | None = None
    rows: list[dict[str, Any]] = []
    for evaluation in evaluations:
        if evaluation.updates_search:
            best = evaluation.accuracy if best is None else max(best, evaluation.accuracy)
        if best is None:
            raise ValueError("the first real training row did not enter the search population")
        rows.append(
            {
                "method": method,
                "search_seed": search_seed,
                "budget": evaluation.budget,
                "event_type": evaluation.event_type,
                "inserted": evaluation.inserted,
                "accuracy": evaluation.accuracy,
                "search_best_so_far": best,
            }
        )
    return rows


def _best_at(curve: Sequence[Mapping[str, Any]], budget: int) -> float:
    matches = [row for row in curve if int(row["budget"]) == budget]
    if len(matches) != 1:
        raise ValueError(f"curve does not contain exactly one row at budget {budget}")
    return float(matches[0]["search_best_so_far"])


def _final_population(evaluations: Sequence[Evaluation], population_size: int) -> list[Evaluation]:
    population: list[Evaluation] = []
    for evaluation in evaluations:
        if not evaluation.updates_search:
            continue
        if len(population) >= population_size:
            population.pop(0)  # Aging Evolution FIFO removal.
        population.append(evaluation)
    if len(population) != population_size:
        raise ValueError(
            f"final population has {len(population)} members; expected {population_size}"
        )
    return population


def summarise_run(
    evaluations: Sequence[Evaluation],
    method: str,
    search_seed: int,
    population_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    curve = build_curve(evaluations, method, search_seed)
    first_evaluations = [item for item in evaluations if not item.is_repeat]
    initial = [item for item in evaluations if item.updates_search][:population_size]
    if len(initial) != population_size:
        raise ValueError(
            f"{method} {search_seed}: only {len(initial)} initial population records"
        )

    searchable = [item for item in evaluations if item.updates_search]
    best_record = max(searchable, key=lambda item: item.accuracy)
    population = _final_population(evaluations, population_size)
    training_times = [
        item.training_time for item in evaluations if item.training_time is not None
    ]

    summary = {
        "method": method,
        "search_seed": search_seed,
        "initial_best": max(item.accuracy for item in initial),
        "final_best": curve[-1]["search_best_so_far"],
        "final_population_best": max(item.accuracy for item in population),
        "best_at_budget_20": _best_at(curve, 20),
        "best_at_budget_25": _best_at(curve, 25),
        "best_at_budget_30": _best_at(curve, 30),
        "real_training_runs": len(evaluations),
        "first_evaluation_count": len(first_evaluations),
        "repeat_evaluation_count": sum(item.is_repeat for item in evaluations),
        # Sum of per-run real CNN training times. A blank value means that the
        # source log did not expose training time.
        "runtime": sum(training_times) if training_times else None,
        "mean_training_time": fmean(training_times) if training_times else None,
        "parameter_count_of_best": best_record.parameter_count,
    }
    return summary, curve


def _format_cell(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.10g}"
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    return value


def write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_cell(row.get(key)) for key in columns})


def plot_curves(path: Path, curves: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to create pilot_best_so_far_diagnostic.png"
        ) from exc

    method_styles = {
        "RE": {"marker": "o", "color": "#1f77b4"},
        "SA-RE": {"marker": "s", "color": "#ff7f0e"},
        "RS-SA-RE": {"marker": "^", "color": "#2ca02c"},
    }
    seeds = sorted({int(row["search_seed"]) for row in curves})
    fig, axes = plt.subplots(1, len(seeds), figsize=(12, 4.8), sharex=True, sharey=True)
    if len(seeds) == 1:
        axes = [axes]

    for axis, seed in zip(axes, seeds):
        for method in ("RE", "SA-RE", "RS-SA-RE"):
            rows = [
                row
                for row in curves
                if int(row["search_seed"]) == seed and row["method"] == method
            ]
            rows.sort(key=lambda row: int(row["budget"]))
            if not rows:
                continue
            style = method_styles[method]
            axis.plot(
                [int(row["budget"]) for row in rows],
                [float(row["search_best_so_far"]) for row in rows],
                label=method,
                linewidth=1.8,
                marker=style["marker"],
                markevery=5,
                markersize=4.5,
                color=style["color"],
            )
        axis.set_title(f"Matched search seed {seed}")
        axis.set_xlabel("Real CNN training budget")
        axis.set_xlim(1, 30)
        axis.set_xticks([1, 5, 10, 15, 20, 25, 30])
        axis.grid(True, alpha=0.25)

    axes[0].set_ylabel("Best-so-far validation accuracy")
    axes[-1].legend(loc="best", frameon=True)
    fig.suptitle("Three-method pilot diagnostic (search fitness)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def consolidate(
    pilot_root: Path,
    population_size: int = 20,
    expected_budget: int = 30,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for directory_name, seed, method in METHOD_DIRS:
        run_dir = pilot_root / directory_name
        if not run_dir.is_dir():
            raise FileNotFoundError(f"missing required pilot directory: {run_dir}")
        raw, source = load_raw_records(run_dir)
        evaluations = parse_evaluations(raw, source)
        if len(evaluations) != expected_budget:
            raise ValueError(
                f"{run_dir}: {len(evaluations)} real training rows; expected {expected_budget}"
            )
        summary, curve = summarise_run(evaluations, method, seed, population_size)
        summaries.append(summary)
        curves.extend(curve)

    if len(summaries) != 6:
        raise AssertionError(f"expected six summaries, got {len(summaries)}")
    return summaries, curves


def _warn_incomplete_metadata(summaries: Sequence[Mapping[str, Any]]) -> None:
    for row in summaries:
        missing = [
            name
            for name in ("runtime", "mean_training_time", "parameter_count_of_best")
            if row.get(name) is None
        ]
        if missing:
            print(
                f"WARNING: {row['method']} {row['search_seed']} has no "
                + ", ".join(missing)
                + " in its evaluation log; the corresponding summary cells are blank.",
                file=sys.stderr,
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, default=Path("experiments/pilot"))
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--expected-budget", type=int, default=30)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="default: PILOT_ROOT/pilot_comparison_summary.csv",
    )
    parser.add_argument(
        "--curve-output",
        type=Path,
        default=None,
        help="default: PILOT_ROOT/pilot_best_so_far_curves.csv",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="default: PILOT_ROOT/pilot_best_so_far_diagnostic.png",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    pilot_root = args.pilot_root
    summary_output = args.summary_output or pilot_root / "pilot_comparison_summary.csv"
    curve_output = args.curve_output or pilot_root / "pilot_best_so_far_curves.csv"
    plot_output = args.plot_output or pilot_root / "pilot_best_so_far_diagnostic.png"

    summaries, curves = consolidate(
        pilot_root=pilot_root,
        population_size=args.population_size,
        expected_budget=args.expected_budget,
    )
    _warn_incomplete_metadata(summaries)
    write_csv(summary_output, SUMMARY_COLUMNS, summaries)
    write_csv(curve_output, CURVE_COLUMNS, curves)
    plot_curves(plot_output, curves)

    print(f"WROTE {summary_output} ({len(summaries)} rows)")
    print(f"WROTE {curve_output} ({len(curves)} rows)")
    print(f"WROTE {plot_output}")
    print("Pilot consolidation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
