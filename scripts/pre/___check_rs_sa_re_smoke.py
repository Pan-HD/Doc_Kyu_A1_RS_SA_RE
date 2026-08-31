"""Audit RS-SA-RE debug, pilot, and formal output directories.

The audit is deliberately read-only.  It validates the real-CNN budget from
the durable CSV files rather than trusting console output or an in-memory
result object.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


@dataclass(frozen=True)
class RSSAReAudit:
    mode: str
    real_training_runs: int
    first_evaluations: int
    repeat_evaluations: int
    warmup_repeats: int
    periodic_repeats: int
    candidate_rows: int
    selected_rows: int
    final_population_size: int


_EXPECTED = {
    "debug": RSSAReAudit("debug", 30, 25, 5, 4, 1, 25, 5, 20),
    "pilot": RSSAReAudit("pilot", 30, 25, 5, 4, 1, 25, 5, 20),
    "formal": RSSAReAudit("formal", 60, 49, 11, 4, 7, 145, 29, 20),
}


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = tuple(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    if not fieldnames:
        raise ValueError(f"{path} has no CSV header")
    return fieldnames, rows


def _first_present(row: Mapping[str, str], names: Iterable[str]) -> str | None:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return None


def _is_repeat_row(row: Mapping[str, str]) -> bool:
    value = _first_present(
        row,
        ("event_type", "evaluation_type", "event", "kind", "replica"),
    )
    if value is None:
        return False
    normalized = value.lower().replace("-", "_")
    return "repeat" in normalized or normalized in {"second", "replica_2", "2"}


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "selected"}


def _count_selected(rows: list[dict[str, str]]) -> int:
    if not rows:
        return 0
    names = ("selected", "is_selected", "chosen")
    if not any(name in rows[0] for name in names):
        raise ValueError("candidate_predictions.csv must contain selected status")
    return sum(
        _truthy(_first_present(row, names) or "")
        for row in rows
    )


def _repeat_phase_counts(rows: list[dict[str, str]]) -> tuple[int, int]:
    names = ("repeat_phase", "phase", "policy_phase", "reason")
    values = [_first_present(row, names) for row in rows]
    if all(value is None for value in values):
        # The frozen schedule always writes the four warm-up repeats first.
        warmup = min(4, len(rows))
        return warmup, len(rows) - warmup
    if any(value is None for value in values):
        raise ValueError("repeat phase is only partially recorded")
    warmup = 0
    periodic = 0
    for value in values:
        normalized = str(value).lower().replace("-", "_")
        if "warm" in normalized:
            warmup += 1
        elif "period" in normalized:
            periodic += 1
        else:
            raise ValueError(f"unknown repeat phase: {value}")
    return warmup, periodic


def _population_size(value: Any) -> int | None:
    if isinstance(value, dict):
        for key in ("final_population_size", "population_size"):
            candidate = value.get(key)
            if isinstance(candidate, int):
                return candidate
        for key in ("final_population", "population"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                return len(candidate)
        preferred = (
            "summary",
            "result",
            "evolution",
            "final_state",
            "state",
            "history",
            "events",
        )
        for key in preferred:
            if key in value:
                candidate = _population_size(value[key])
                if candidate is not None:
                    return candidate
        for candidate_value in value.values():
            candidate = _population_size(candidate_value)
            if candidate is not None:
                return candidate
    elif isinstance(value, list):
        for item in reversed(value):
            candidate = _population_size(item)
            if candidate is not None:
                return candidate
    return None


def _read_history_population(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    size = _population_size(value)
    if size is None:
        raise ValueError(f"cannot determine final population size from {path}")
    return size


def _assert_equal(name: str, actual: int, expected: int) -> None:
    if actual != expected:
        raise ValueError(f"{name}: expected {expected}, observed {actual}")


def audit_rs_sa_re_output(
    output_dir: str | Path,
    *,
    expected_mode: str | None = None,
) -> RSSAReAudit:
    output = Path(output_dir)
    config = _load_yaml(output / "config.yaml")
    mode = str(config["experiment"]["mode"]).lower()
    if expected_mode is not None and mode != expected_mode:
        raise ValueError(f"expected mode={expected_mode}, observed mode={mode}")
    if mode not in _EXPECTED:
        raise ValueError(f"unsupported RS-SA-RE audit mode: {mode}")
    if str(config["experiment"]["method"]).upper() != "RS-SA-RE":
        raise ValueError("config.yaml is not an RS-SA-RE run")

    _evaluation_fields, evaluation_rows = _read_csv(output / "evaluations.csv")
    _candidate_fields, candidate_rows = _read_csv(
        output / "candidate_predictions.csv"
    )
    _repeat_fields, repeat_rows = _read_csv(output / "repeat_evaluations.csv")

    if any(_is_repeat_row(row) for row in evaluation_rows):
        first_rows = [row for row in evaluation_rows if not _is_repeat_row(row)]
    else:
        first_rows = evaluation_rows
    warmup_repeats, periodic_repeats = _repeat_phase_counts(repeat_rows)
    selected_rows = _count_selected(candidate_rows)
    final_population_size = _read_history_population(output / "history.json")

    observed = RSSAReAudit(
        mode=mode,
        real_training_runs=len(first_rows) + len(repeat_rows),
        first_evaluations=len(first_rows),
        repeat_evaluations=len(repeat_rows),
        warmup_repeats=warmup_repeats,
        periodic_repeats=periodic_repeats,
        candidate_rows=len(candidate_rows),
        selected_rows=selected_rows,
        final_population_size=final_population_size,
    )
    expected = _EXPECTED[mode]
    for field in (
        "real_training_runs",
        "first_evaluations",
        "repeat_evaluations",
        "warmup_repeats",
        "periodic_repeats",
        "candidate_rows",
        "selected_rows",
        "final_population_size",
    ):
        _assert_equal(field, getattr(observed, field), getattr(expected, field))

    audit_expectations = config.get("audit_expectations", {})
    config_expected = {
        "real_training_runs": observed.real_training_runs,
        "first_evaluations": observed.first_evaluations,
        "repeat_evaluations": observed.repeat_evaluations,
        "candidate_rows": observed.candidate_rows,
        "selected_rows": observed.selected_rows,
        "final_population": observed.final_population_size,
    }
    for name, actual in config_expected.items():
        if name in audit_expectations:
            _assert_equal(
                f"config audit_expectations.{name}",
                actual,
                int(audit_expectations[name]),
            )
    return observed


def audit_rs_sa_re_smoke(output_dir: str | Path) -> RSSAReAudit:
    return audit_rs_sa_re_output(output_dir, expected_mode="debug")


def audit_rs_sa_re_pilot(output_dir: str | Path) -> RSSAReAudit:
    return audit_rs_sa_re_output(output_dir, expected_mode="pilot")


def audit_rs_sa_re_formal(output_dir: str | Path) -> RSSAReAudit:
    return audit_rs_sa_re_output(output_dir, expected_mode="formal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit an RS-SA-RE run directory.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--mode",
        choices=("auto", "debug", "pilot", "formal"),
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected_mode = None if args.mode == "auto" else args.mode
    audit = audit_rs_sa_re_output(args.output_dir, expected_mode=expected_mode)
    print(
        "RS-SA-RE audit: PASS "
        f"mode={audit.mode} real={audit.real_training_runs} "
        f"first={audit.first_evaluations} repeats={audit.repeat_evaluations} "
        f"candidate_rows={audit.candidate_rows} selected={audit.selected_rows} "
        f"population={audit.final_population_size}"
    )


if __name__ == "__main__":
    main()
