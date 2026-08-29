"""Audit a 5-epoch RS-SA-RE pilot and its matched initialization."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import yaml

from scripts.check_rs_sa_re_smoke import (
    SmokeAuditSummary,
    audit_rs_sa_re_pilot,
)


MATCHED_INITIALIZATION_SIZE = 20


@dataclass(frozen=True)
class MatchedPilotAuditSummary:
    search_seed: int
    architecture_matches_re: int
    architecture_matches_sa_re: int
    training_seed_matches_re: int
    training_seed_matches_sa_re: int
    run: SmokeAuditSummary


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _initialization_rows(path: Path, *, label: str) -> list[dict[str, str]]:
    rows = _read_csv(path)
    first_rows = [
        row
        for row in rows
        if row.get("event_type", "first_evaluation") == "first_evaluation"
    ]
    initialization = [
        row for row in first_rows if row.get("phase") == "initialization"
    ]
    if len(initialization) < MATCHED_INITIALIZATION_SIZE:
        raise RuntimeError(
            f"{label} contains fewer than 20 initialization evaluations"
        )
    return initialization[:MATCHED_INITIALIZATION_SIZE]


def _architecture(value: str, *, label: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} architecture is not valid JSON") from error


def _compare_initialization(
    *,
    rs_rows: list[dict[str, str]],
    reference_rows: list[dict[str, str]],
    label: str,
) -> tuple[int, int]:
    architecture_matches = []
    training_seed_matches = []
    for index, (rs_row, reference_row) in enumerate(
        zip(rs_rows, reference_rows, strict=True),
        start=1,
    ):
        architecture_matches.append(
            _architecture(rs_row["architecture"], label="RS-SA-RE")
            == _architecture(reference_row["architecture"], label=label)
        )
        training_seed_matches.append(
            int(rs_row["training_seed"])
            == int(reference_row["training_seed"])
        )

    bad_architectures = [
        index
        for index, matched in enumerate(architecture_matches, start=1)
        if not matched
    ]
    bad_training_seeds = [
        index
        for index, matched in enumerate(training_seed_matches, start=1)
        if not matched
    ]
    if bad_architectures:
        raise RuntimeError(
            f"{label}/RS-SA-RE initialization architectures differ at "
            f"rows {bad_architectures}"
        )
    if bad_training_seeds:
        raise RuntimeError(
            f"{label}/RS-SA-RE initialization training seeds differ at "
            f"rows {bad_training_seeds}"
        )
    return sum(architecture_matches), sum(training_seed_matches)


def audit_matched_rs_sa_re_pilot(
    rs_sa_re_output_dir: str | Path,
    *,
    re_output_dir: str | Path | None = None,
    sa_re_output_dir: str | Path | None = None,
) -> MatchedPilotAuditSummary:
    rs_sa_re_output_dir = Path(rs_sa_re_output_dir)
    config_path = rs_sa_re_output_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("RS-SA-RE config.yaml must contain a mapping")
    search_seed = int(config["experiment"]["search_seed"])
    if search_seed not in {2701, 2702}:
        raise ValueError("matched Part G pilot seed must be 2701 or 2702")

    re_dir = (
        Path(re_output_dir)
        if re_output_dir is not None
        else Path(f"experiments/pilot/re_{search_seed}")
    )
    sa_re_dir = (
        Path(sa_re_output_dir)
        if sa_re_output_dir is not None
        else Path(f"experiments/pilot/sa_re_{search_seed}")
    )

    run_summary = audit_rs_sa_re_pilot(rs_sa_re_output_dir)
    rs_rows = _initialization_rows(
        rs_sa_re_output_dir / "evaluations.csv",
        label="RS-SA-RE",
    )
    re_rows = _initialization_rows(
        re_dir / "evaluations.csv",
        label="RE",
    )
    sa_re_rows = _initialization_rows(
        sa_re_dir / "evaluations.csv",
        label="SA-RE",
    )
    re_architectures, re_training_seeds = _compare_initialization(
        rs_rows=rs_rows,
        reference_rows=re_rows,
        label="RE",
    )
    sa_architectures, sa_training_seeds = _compare_initialization(
        rs_rows=rs_rows,
        reference_rows=sa_re_rows,
        label="SA-RE",
    )
    return MatchedPilotAuditSummary(
        search_seed=search_seed,
        architecture_matches_re=re_architectures,
        architecture_matches_sa_re=sa_architectures,
        training_seed_matches_re=re_training_seeds,
        training_seed_matches_sa_re=sa_training_seeds,
        run=run_summary,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit RS-SA-RE pilot logs and matched initialization."
    )
    parser.add_argument("rs_sa_re_output_dir", type=Path)
    parser.add_argument("--re-dir", type=Path, default=None)
    parser.add_argument("--sa-re-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = audit_matched_rs_sa_re_pilot(
        args.rs_sa_re_output_dir,
        re_output_dir=args.re_dir,
        sa_re_output_dir=args.sa_re_dir,
    )
    print("RS-SA-RE pilot audit: PASS")
    print(f"search seed: {summary.search_seed}")
    print(f"real training runs: {summary.run.real_training_runs}")
    print(f"first evaluations: {summary.run.first_evaluations}")
    print(f"repeat evaluations: {summary.run.warmup_repeats + summary.run.periodic_repeats}")
    print(f"final population: {summary.run.final_population_size}")
    print(f"RE architecture matches: {summary.architecture_matches_re}/20")
    print(f"SA-RE architecture matches: {summary.architecture_matches_sa_re}/20")
    print(f"RE training-seed matches: {summary.training_seed_matches_re}/20")
    print(f"SA-RE training-seed matches: {summary.training_seed_matches_sa_re}/20")


if __name__ == "__main__":
    main()
