#!/usr/bin/env python3
"""Audit matched initialization across RE, SA-RE, and RS-SA-RE pilots.

The audit compares the first 20 non-repeat initialization records position by
position. Architecture/genotype and training seed must match exactly. Accuracy
is intentionally excluded because GPU training can be nondeterministic.

Run from the project root:

    python scripts/audit_three_method_pilots.py --pilot-root experiments/pilot
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


RUNS: dict[int, dict[str, str]] = {
    2701: {
        "RE": "re_2701",
        "SA-RE": "sa_re_2701",
        "RS-SA-RE": "rs_sa_re_2701",
    },
    2702: {
        "RE": "re_2702",
        "SA-RE": "sa_re_2702",
        "RS-SA-RE": "rs_sa_re_2702",
    },
}

INDEX_ALIASES = (
    "evaluation_index",
    "real_evaluation_index",
    "real_evaluation",
    "training_run_index",
    "eval_index",
    "budget_index",
    "budget",
)

ARCHITECTURE_ALIASES = (
    "architecture",
    "architecture_json",
    "genotype",
    "genotype_json",
    "arch",
    "architecture_encoding",
    "encoding",
)

TRAINING_SEED_ALIASES = (
    "training_seed",
    "train_seed",
    "cnn_training_seed",
    "cnn_seed",
    "model_seed",
    "evaluation_seed",
)

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
)


@dataclass(frozen=True)
class InitializationRecord:
    position: int
    architecture: str
    training_seed: str


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
    raise ValueError(f"invalid boolean value {value!r}")


def canonicalise(value: Any) -> str:
    """Return a stable representation for JSON, Python literals, or strings."""

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("empty architecture value")
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
            if parsed != text:
                return canonicalise(parsed)
        # Genotype class reprs are sometimes neither JSON nor Python literals.
        return re.sub(r"\s+", "", text)
    if isinstance(value, Mapping):
        normalised = {
            str(key): json.loads(canonicalise(child))
            if canonicalise(child).startswith(("{", "[", '"'))
            else canonicalise(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
        return json.dumps(normalised, sort_keys=True, separators=(",", ":"))
    if isinstance(value, (list, tuple)):
        return json.dumps(
            [json.loads(canonicalise(item)) if canonicalise(item).startswith(("{", "[", '"')) else canonicalise(item) for item in value],
            sort_keys=True,
            separators=(",", ":"),
        )
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def extract_architecture(row: Mapping[str, Any]) -> str:
    direct = pick(row, ARCHITECTURE_ALIASES)
    if direct not in (None, ""):
        return canonicalise(direct)

    excluded_fragments = (
        "accuracy",
        "fitness",
        "parameter",
        "count",
        "seed",
        "time",
        "budget",
        "index",
        "inserted",
    )
    components = {
        key: value
        for key, value in row.items()
        if value not in (None, "")
        and (
            key.startswith("normal_")
            or key.startswith("reduction_")
            or key.startswith("architecture_")
            or key.startswith("genotype_")
        )
        and not any(fragment in key for fragment in excluded_fragments)
    }
    if components:
        return canonicalise(components)

    raise ValueError(
        "could not identify architecture/genotype column; available columns: "
        + ", ".join(sorted(row))
    )


def extract_training_seed(row: Mapping[str, Any]) -> str:
    value = pick(row, TRAINING_SEED_ALIASES)
    if value in (None, ""):
        raise ValueError(
            "could not identify training-seed column; available columns: "
            + ", ".join(sorted(row))
        )
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if math.isfinite(number) and number.is_integer():
        return str(int(number))
    return text


def is_repeat(row: Mapping[str, Any]) -> bool:
    explicit = as_bool(pick(row, REPEAT_ALIASES))
    if explicit is True:
        return True
    event = str(pick(row, EVENT_ALIASES) or "").lower()
    phase = str(pick(row, PHASE_ALIASES) or "").lower()
    if "repeat" in f"{event} {phase}" or "retrain" in f"{event} {phase}":
        return True
    inserted = as_bool(pick(row, INSERTED_ALIASES))
    base_index = pick(row, BASE_INDEX_ALIASES)
    return inserted is False and base_index not in (None, "")


def _find_record_list(value: Any) -> list[dict[str, Any]] | None:
    if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        keys = {normalise_key(str(key)) for item in value for key in item}
        if keys.intersection(ARCHITECTURE_ALIASES) or keys.intersection(TRAINING_SEED_ALIASES):
            return value
    if isinstance(value, dict):
        for key in ("evaluations", "records", "history", "events", "results"):
            if key in value:
                found = _find_record_list(value[key])
                if found is not None:
                    return found
        for child in value.values():
            found = _find_record_list(child)
            if found is not None:
                return found
    return None


def load_records(run_dir: Path) -> tuple[list[dict[str, Any]], Path]:
    for name in ("evaluations.csv", "evaluation_log.csv", "results.csv"):
        path = run_dir / name
        if path.is_file():
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                records = list(csv.DictReader(handle))
            if not records:
                raise ValueError(f"no records in {path}")
            return [normalise_record(record) for record in records], path

    path = run_dir / "history.json"
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        records = _find_record_list(payload)
        if not records:
            raise ValueError(f"could not find evaluation records in {path}")
        return [normalise_record(record) for record in records], path

    raise FileNotFoundError(
        f"{run_dir}: expected evaluations.csv, evaluation_log.csv, results.csv, or history.json"
    )


def order_records(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Use a genuine 1..N/0..N-1 index; otherwise preserve file row order."""

    for key in INDEX_ALIASES:
        values: list[int] = []
        for row in rows:
            value = row.get(key)
            try:
                number = float(value)
            except (TypeError, ValueError):
                values = []
                break
            if not math.isfinite(number) or not number.is_integer():
                values = []
                break
            values.append(int(number))
        if len(values) != len(rows):
            continue
        if sorted(values) in (list(range(1, len(rows) + 1)), list(range(len(rows)))):
            return [row for _, row in sorted(zip(values, rows), key=lambda pair: pair[0])]
    return list(rows)


def select_initialization_rows(
    rows: Sequence[Mapping[str, Any]], initialization_size: int
) -> list[Mapping[str, Any]]:
    ordered = order_records(rows)
    first_evaluations = [row for row in ordered if not is_repeat(row)]

    phase_marked = [
        row
        for row in first_evaluations
        if "init" in str(pick(row, PHASE_ALIASES) or "").lower()
        or "init" in str(pick(row, EVENT_ALIASES) or "").lower()
    ]
    selected = phase_marked if len(phase_marked) >= initialization_size else first_evaluations
    if len(selected) < initialization_size:
        raise ValueError(
            f"only {len(selected)} non-repeat initialization rows; expected {initialization_size}"
        )
    return selected[:initialization_size]


def extract_initialization(
    run_dir: Path, initialization_size: int
) -> tuple[list[InitializationRecord], Path]:
    rows, source = load_records(run_dir)
    selected = select_initialization_rows(rows, initialization_size)
    records = [
        InitializationRecord(
            position=position,
            architecture=extract_architecture(row),
            training_seed=extract_training_seed(row),
        )
        for position, row in enumerate(selected, start=1)
    ]
    return records, source


def compare_initializations(
    left: Sequence[InitializationRecord], right: Sequence[InitializationRecord]
) -> dict[str, Any]:
    if len(left) != len(right):
        raise ValueError(f"initialization length mismatch: {len(left)} vs {len(right)}")
    architecture_mismatches = [
        a.position for a, b in zip(left, right) if a.architecture != b.architecture
    ]
    seed_mismatches = [
        a.position for a, b in zip(left, right) if a.training_seed != b.training_seed
    ]
    total = len(left)
    result = {
        "architecture_matches": total - len(architecture_mismatches),
        "architecture_total": total,
        "architecture_mismatch_indices": architecture_mismatches,
        "training_seed_matches": total - len(seed_mismatches),
        "training_seed_total": total,
        "training_seed_mismatch_indices": seed_mismatches,
        "pass": not architecture_mismatches and not seed_mismatches,
    }
    return result


def build_audit(pilot_root: Path, initialization_size: int = 20) -> dict[str, Any]:
    seed_results: dict[str, Any] = {}
    overall_pass = True

    for search_seed, method_dirs in RUNS.items():
        initializations: dict[str, list[InitializationRecord]] = {}
        sources: dict[str, str] = {}
        source_directories: dict[str, str] = {}

        for method, directory_name in method_dirs.items():
            run_dir = pilot_root / directory_name
            if not run_dir.is_dir():
                raise FileNotFoundError(f"missing required pilot directory: {run_dir}")
            records, source = extract_initialization(run_dir, initialization_size)
            initializations[method] = records
            sources[method] = source.as_posix()
            source_directories[method] = run_dir.as_posix()

        comparisons = {
            "RE_vs_SA-RE": compare_initializations(
                initializations["RE"], initializations["SA-RE"]
            ),
            "RE_vs_RS-SA-RE": compare_initializations(
                initializations["RE"], initializations["RS-SA-RE"]
            ),
        }
        seed_pass = all(result["pass"] for result in comparisons.values())
        overall_pass = overall_pass and seed_pass
        seed_results[str(search_seed)] = {
            "source_directories": source_directories,
            "source_files": sources,
            "initialization_counts": {
                method: len(records) for method, records in initializations.items()
            },
            "comparisons": comparisons,
            "pass": seed_pass,
        }

    return {
        "schema_version": 1,
        "audit": "three_method_matched_initialization",
        "status": "pass" if overall_pass else "fail",
        "initialization_size": initialization_size,
        "comparison_policy": {
            "architecture": "canonical architecture/genotype equality by initialization position",
            "training_seed": "exact equality by initialization position",
            "accuracy": "not compared; nondeterministic GPU differences are permitted",
            "repeat_evaluations": "excluded from initialization",
        },
        "seeds": seed_results,
        "overall_pass": overall_pass,
        "provenance": "Generated directly from frozen pilot evaluation logs.",
    }


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
        try:
            Path(temporary_name).unlink(missing_ok=True)
        finally:
            raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, default=Path("experiments/pilot"))
    parser.add_argument("--initialization-size", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="default: PILOT_ROOT/pilot_comparison_audit.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output or args.pilot_root / "pilot_comparison_audit.json"
    audit = build_audit(args.pilot_root, args.initialization_size)
    write_json_atomic(output, audit)

    for search_seed, seed_result in audit["seeds"].items():
        for comparison_name, result in seed_result["comparisons"].items():
            print(
                f"{search_seed} {comparison_name}: "
                f"architecture={result['architecture_matches']}/{result['architecture_total']} "
                f"training_seed={result['training_seed_matches']}/{result['training_seed_total']} "
                f"status={'PASS' if result['pass'] else 'FAIL'}"
            )
    print(f"WROTE {output}")
    print(
        "Three-method matched initialization audit: "
        + ("PASS" if audit["overall_pass"] else "FAIL")
    )
    return 0 if audit["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

