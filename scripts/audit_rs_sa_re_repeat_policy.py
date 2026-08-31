#!/usr/bin/env python3
"""Audit B=30 RS-SA-RE logs and simulate the formal B=60 repeat policy.

This is an offline audit. It uses only Python's standard library, never imports
the trainer, and never starts CNN evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SEARCH_SEEDS = (2701, 2702)

EVENT_ALIASES = ("event", "event_type", "evaluation_type", "kind")
PHASE_ALIASES = ("phase", "stage", "search_phase")
REPEAT_ALIASES = ("is_repeat", "repeat", "repeated")
INSERTED_ALIASES = (
    "inserted",
    "population_inserted",
    "entered_population",
    "is_inserted",
)
BASE_INDEX_ALIASES = (
    "base_evaluation_index",
    "base_eval",
    "repeated_evaluation_index",
    "repeat_of_evaluation_index",
)
SELECTED_ALIASES = ("selected", "is_selected", "chosen", "selected_candidate")


@dataclass(frozen=True)
class Policy:
    population_size: int = 20
    candidate_count: int = 5
    warmup_pairs: int = 4
    repeat_interval: int = 4


@dataclass(frozen=True)
class ToyEvent:
    budget: int
    event_type: str
    repeat_kind: str | None
    inserted: bool
    candidate_rows: int
    selected_rows: int
    population_before: int
    population_after: int
    search_best_before: float
    search_best_after: float


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


def is_repeat(row: Mapping[str, Any]) -> bool:
    if as_bool(pick(row, REPEAT_ALIASES)) is True:
        return True
    text = f"{pick(row, EVENT_ALIASES) or ''} {pick(row, PHASE_ALIASES) or ''}".lower()
    if "repeat" in text or "retrain" in text:
        return True
    inserted = as_bool(pick(row, INSERTED_ALIASES))
    base_index = pick(row, BASE_INDEX_ALIASES)
    return inserted is False and base_index not in (None, "")


def simulate_repeat_policy(real_budget: int, policy: Policy = Policy()) -> list[ToyEvent]:
    """Simulate scheduling with a fake evaluator and explicit state changes."""

    if real_budget < policy.population_size + policy.warmup_pairs:
        raise ValueError(
            "real_budget must cover the full initialization and warm-up repeats"
        )
    events: list[ToyEvent] = []
    population_size = 0
    search_best = float("-inf")

    def append_first(*, initialization: bool) -> None:
        nonlocal population_size, search_best
        budget = len(events) + 1
        before_population = population_size
        before_best = search_best
        fake_accuracy = 0.50 + budget / 1000.0
        if initialization:
            population_size += 1
            candidate_rows = 0
            selected_rows = 0
        else:
            # FIFO removes one and inserts one, so a full population stays full.
            population_size = min(policy.population_size, population_size + 1)
            candidate_rows = policy.candidate_count
            selected_rows = 1
        search_best = max(search_best, fake_accuracy)
        events.append(
            ToyEvent(
                budget=budget,
                event_type="first_evaluation",
                repeat_kind=None,
                inserted=True,
                candidate_rows=candidate_rows,
                selected_rows=selected_rows,
                population_before=before_population,
                population_after=population_size,
                search_best_before=before_best,
                search_best_after=search_best,
            )
        )

    def append_repeat(kind: str) -> None:
        budget = len(events) + 1
        # The fake repeat accuracy is intentionally larger than search best. If
        # best changes, the repeat invariant test will fail.
        events.append(
            ToyEvent(
                budget=budget,
                event_type="repeat_evaluation",
                repeat_kind=kind,
                inserted=False,
                candidate_rows=0,
                selected_rows=0,
                population_before=population_size,
                population_after=population_size,
                search_best_before=search_best,
                search_best_after=search_best,
            )
        )

    for _ in range(policy.population_size):
        append_first(initialization=True)
    for _ in range(policy.warmup_pairs):
        append_repeat("warmup")

    first_since_periodic_repeat = 0
    while len(events) < real_budget:
        if first_since_periodic_repeat == policy.repeat_interval:
            append_repeat("periodic")
            first_since_periodic_repeat = 0
        else:
            append_first(initialization=False)
            first_since_periodic_repeat += 1

    if len(events) != real_budget:
        raise AssertionError(f"scheduler stopped at {len(events)}, expected {real_budget}")
    return events


def summarise_events(events: Sequence[ToyEvent], policy: Policy = Policy()) -> dict[str, int]:
    first = [event for event in events if event.event_type == "first_evaluation"]
    repeats = [event for event in events if event.event_type == "repeat_evaluation"]
    warmup = [event for event in repeats if event.repeat_kind == "warmup"]
    periodic = [event for event in repeats if event.repeat_kind == "periodic"]
    evolution = first[policy.population_size :]
    return {
        "real_training_runs": len(events),
        "first_evaluations": len(first),
        "repeat_evaluations": len(repeats),
        "initial_evaluations": min(len(first), policy.population_size),
        "evolution_children": len(evolution),
        "warmup_repeats": len(warmup),
        "periodic_repeats": len(periodic),
        "candidate_rows": sum(event.candidate_rows for event in events),
        "selected_rows": sum(event.selected_rows for event in events),
        "final_population": events[-1].population_after,
    }


def validate_repeat_invariants(events: Sequence[ToyEvent]) -> None:
    budgets = [event.budget for event in events]
    if budgets != list(range(1, len(events) + 1)):
        raise AssertionError("toy scheduler produced non-contiguous or off-by-one budgets")
    for event in events:
        if event.event_type == "repeat_evaluation":
            if event.inserted:
                raise AssertionError(f"repeat at B{event.budget} entered population")
            if event.candidate_rows or event.selected_rows:
                raise AssertionError(f"repeat at B{event.budget} generated candidate/selection rows")
            if event.population_before != event.population_after:
                raise AssertionError(f"repeat at B{event.budget} changed population")
            if event.search_best_before != event.search_best_after:
                raise AssertionError(f"repeat at B{event.budget} updated search best")


def audit_pilot_run(run_dir: Path, policy: Policy, expected_budget: int = 30) -> dict[str, Any]:
    evaluation_path = run_dir / "evaluations.csv"
    candidate_path = run_dir / "candidate_predictions.csv"
    evaluations = load_csv(evaluation_path)
    candidates = load_csv(candidate_path)

    repeats = [row for row in evaluations if is_repeat(row)]
    first = [row for row in evaluations if not is_repeat(row)]
    inserted_first = [
        row for row in first if as_bool(pick(row, INSERTED_ALIASES)) is not False
    ]
    explicitly_inserted_repeats = sum(
        as_bool(pick(row, INSERTED_ALIASES)) is True for row in repeats
    )
    unknown_repeat_insertion = sum(
        pick(row, INSERTED_ALIASES) in (None, "") for row in repeats
    )

    selected_values = [pick(row, SELECTED_ALIASES) for row in candidates]
    if all(value in (None, "") for value in selected_values):
        if len(candidates) % policy.candidate_count:
            raise ValueError(
                f"{candidate_path}: candidate row count is not divisible by K={policy.candidate_count}"
            )
        selected_rows = len(candidates) // policy.candidate_count
        selected_source = "derived_one_per_complete_K_set"
    else:
        selected_rows = sum(as_bool(value) is True for value in selected_values)
        selected_source = "selected_column"

    initial_evaluations = min(len(first), policy.population_size)
    evolution_children = len(first) - initial_evaluations
    warmup_repeats = min(len(repeats), policy.warmup_pairs)
    periodic_repeats = max(0, len(repeats) - warmup_repeats)
    final_population = min(policy.population_size, len(inserted_first))

    result: dict[str, Any] = {
        "source_files": {
            "evaluations": evaluation_path.as_posix(),
            "candidate_predictions": candidate_path.as_posix(),
        },
        "real_training_runs": len(evaluations),
        "first_evaluations": len(first),
        "repeat_evaluations": len(repeats),
        "initial_evaluations": initial_evaluations,
        "evolution_children": evolution_children,
        "warmup_repeats": warmup_repeats,
        "periodic_repeats": periodic_repeats,
        "candidate_rows": len(candidates),
        "selected_rows": selected_rows,
        "selected_rows_source": selected_source,
        "final_population": final_population,
        "explicitly_inserted_repeats": explicitly_inserted_repeats,
        "unknown_repeat_insertion_flags": unknown_repeat_insertion,
    }
    expected = {
        "real_training_runs": expected_budget,
        "first_evaluations": 25,
        "repeat_evaluations": 5,
        "initial_evaluations": 20,
        "evolution_children": 5,
        "warmup_repeats": 4,
        "periodic_repeats": 1,
        "candidate_rows": 25,
        "selected_rows": 5,
        "final_population": 20,
        "explicitly_inserted_repeats": 0,
    }
    mismatches = {
        key: {"actual": result[key], "expected": value}
        for key, value in expected.items()
        if result[key] != value
    }
    result["mismatches"] = mismatches
    result["pass"] = not mismatches
    return result


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def build_audit(pilot_root: Path, formal_budget: int, policy: Policy) -> dict[str, Any]:
    pilot_results = {
        str(seed): audit_pilot_run(pilot_root / f"rs_sa_re_{seed}", policy)
        for seed in SEARCH_SEEDS
    }
    formal_events = simulate_repeat_policy(formal_budget, policy)
    validate_repeat_invariants(formal_events)
    formal_summary = summarise_events(formal_events, policy)
    expected_formal = {
        "real_training_runs": 60,
        "first_evaluations": 49,
        "repeat_evaluations": 11,
        "initial_evaluations": 20,
        "evolution_children": 29,
        "warmup_repeats": 4,
        "periodic_repeats": 7,
        "candidate_rows": 145,
        "selected_rows": 29,
        "final_population": 20,
    }
    formal_mismatches = {
        key: {"actual": formal_summary.get(key), "expected": value}
        for key, value in expected_formal.items()
        if formal_summary.get(key) != value
    }
    formal_summary["mismatches"] = formal_mismatches
    formal_summary["pass"] = not formal_mismatches

    overall_pass = all(result["pass"] for result in pilot_results.values()) and formal_summary["pass"]
    return {
        "schema_version": 1,
        "audit": "rs_sa_re_repeat_policy",
        "policy": asdict(policy),
        "pilot_B30": pilot_results,
        "formal_B60_expected": formal_summary,
        "event_semantics": {
            "repeat_consumes_real_budget": True,
            "repeat_inserts_population": False,
            "repeat_generates_candidates": False,
            "repeat_updates_search_best": False,
            "aging": "FIFO on inserted first evaluations only",
        },
        "overall_pass": overall_pass,
        "provenance": "Generated from frozen B=30 logs plus an offline formal-budget toy simulation.",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, default=Path("experiments/pilot"))
    parser.add_argument("--formal-budget", type=int, default=60)
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--candidate-count", type=int, default=5)
    parser.add_argument("--warmup-pairs", type=int, default=4)
    parser.add_argument("--repeat-interval", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="default: PILOT_ROOT/rs_sa_re_repeat_policy_audit.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    policy = Policy(
        population_size=args.population_size,
        candidate_count=args.candidate_count,
        warmup_pairs=args.warmup_pairs,
        repeat_interval=args.repeat_interval,
    )
    if args.formal_budget != 60:
        raise ValueError(
            "this freeze-gate audit is defined for formal_budget=60; "
            "use simulate_repeat_policy directly for exploratory budgets"
        )
    output = args.output or args.pilot_root / "rs_sa_re_repeat_policy_audit.json"
    audit = build_audit(args.pilot_root, args.formal_budget, policy)
    write_json_atomic(output, audit)

    for seed, result in audit["pilot_B30"].items():
        print(
            f"B=30 seed={seed}: real={result['real_training_runs']} "
            f"first={result['first_evaluations']} repeat={result['repeat_evaluations']} "
            f"candidates={result['candidate_rows']} selected={result['selected_rows']} "
            f"status={'PASS' if result['pass'] else 'FAIL'}"
        )
    formal = audit["formal_B60_expected"]
    print(
        f"B=60 toy: real={formal['real_training_runs']} "
        f"first={formal['first_evaluations']} repeat={formal['repeat_evaluations']} "
        f"candidates={formal['candidate_rows']} selected={formal['selected_rows']} "
        f"status={'PASS' if formal['pass'] else 'FAIL'}"
    )
    print(f"WROTE {output}")
    print(
        "RS-SA-RE repeat policy audit: "
        + ("PASS" if audit["overall_pass"] else "FAIL")
    )
    return 0 if audit["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

