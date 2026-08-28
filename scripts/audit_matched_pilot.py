"""Audit matched RE/SA-RE initialization records.

The first ``population_size`` real evaluations must use the same architecture
sequence and training-seed schedule when RE and SA-RE share a search seed.
Final validation accuracy is reported as a diagnostic and can optionally be
made a hard gate for deterministic training protocols.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = (
    "evaluation_index",
    "architecture",
    "training_seed",
    "final_val_accuracy",
)


class AuditError(ValueError):
    """Raised when an experiment artifact cannot be audited safely."""


@dataclass(frozen=True)
class MatchedInitializationAudit:
    re_path: str
    sa_re_path: str
    search_seed: int | None
    population_size: int
    architecture_matches: int
    training_seed_matches: int
    accuracy_matches: int
    accuracy_tolerance: float
    require_accuracy_match: bool
    mismatches: tuple[dict[str, Any], ...]

    @property
    def architecture_passed(self) -> bool:
        return self.architecture_matches == self.population_size

    @property
    def training_seed_passed(self) -> bool:
        return self.training_seed_matches == self.population_size

    @property
    def accuracy_passed(self) -> bool:
        return self.accuracy_matches == self.population_size

    @property
    def passed(self) -> bool:
        core_passed = self.architecture_passed and self.training_seed_passed
        return core_passed and (
            self.accuracy_passed or not self.require_accuracy_match
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "architecture_passed": self.architecture_passed,
                "training_seed_passed": self.training_seed_passed,
                "accuracy_passed": self.accuracy_passed,
                "passed": self.passed,
            }
        )
        return payload


def _parse_int(value: str, *, field: str, path: Path) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise AuditError(f"{path}: invalid integer in {field}: {value!r}") from error


def _parse_float(value: str, *, field: str, path: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AuditError(f"{path}: invalid float in {field}: {value!r}") from error
    if not math.isfinite(result):
        raise AuditError(f"{path}: non-finite value in {field}: {value!r}")
    return result


def _canonical_architecture(value: str, *, path: Path) -> str:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise AuditError(f"{path}: architecture is not valid JSON") from error
    if not isinstance(parsed, dict):
        raise AuditError(f"{path}: architecture JSON must be an object")
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"))


def _load_initialization_rows(
    path: str | Path,
    *,
    population_size: int,
) -> tuple[dict[str, str], ...]:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise AuditError(f"missing evaluations CSV: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = tuple(reader.fieldnames or ())
        missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise AuditError(
                f"{csv_path}: missing required columns: {', '.join(missing)}"
            )
        rows = list(reader)

    by_index: dict[int, dict[str, str]] = {}
    for row in rows:
        index = _parse_int(
            row["evaluation_index"],
            field="evaluation_index",
            path=csv_path,
        )
        if 1 <= index <= population_size:
            if index in by_index:
                raise AuditError(
                    f"{csv_path}: duplicate evaluation_index={index}"
                )
            phase = row.get("phase", "")
            if phase and phase != "initialization":
                raise AuditError(
                    f"{csv_path}: evaluation {index} has phase={phase!r}, "
                    "expected 'initialization'"
                )
            by_index[index] = row

    expected = set(range(1, population_size + 1))
    missing_indices = sorted(expected.difference(by_index))
    if missing_indices:
        raise AuditError(
            f"{csv_path}: missing initialization evaluations: {missing_indices}"
        )
    return tuple(by_index[index] for index in range(1, population_size + 1))


def _single_search_seed(
    rows: tuple[dict[str, str], ...],
    *,
    path: Path,
) -> int | None:
    if "search_seed" not in rows[0]:
        return None
    seeds = {
        _parse_int(row["search_seed"], field="search_seed", path=path)
        for row in rows
    }
    if len(seeds) != 1:
        raise AuditError(f"{path}: initialization contains multiple search seeds")
    return next(iter(seeds))


def audit_matched_initialization(
    re_path: str | Path,
    sa_re_path: str | Path,
    *,
    population_size: int = 20,
    expected_search_seed: int | None = None,
    accuracy_tolerance: float = 0.0,
    require_accuracy_match: bool = False,
) -> MatchedInitializationAudit:
    """Compare the first P initialization evaluations from RE and SA-RE."""

    if population_size <= 0:
        raise AuditError("population_size must be positive")
    if not math.isfinite(accuracy_tolerance) or accuracy_tolerance < 0.0:
        raise AuditError("accuracy_tolerance must be finite and non-negative")

    re_csv = Path(re_path)
    sa_re_csv = Path(sa_re_path)
    re_rows = _load_initialization_rows(
        re_csv,
        population_size=population_size,
    )
    sa_re_rows = _load_initialization_rows(
        sa_re_csv,
        population_size=population_size,
    )

    re_seed = _single_search_seed(re_rows, path=re_csv)
    sa_re_seed = _single_search_seed(sa_re_rows, path=sa_re_csv)
    if re_seed != sa_re_seed:
        raise AuditError(
            f"search-seed mismatch: RE={re_seed!r}, SA-RE={sa_re_seed!r}"
        )
    if expected_search_seed is not None and re_seed != expected_search_seed:
        raise AuditError(
            f"expected search_seed={expected_search_seed}, found {re_seed!r}"
        )

    architecture_matches = 0
    training_seed_matches = 0
    accuracy_matches = 0
    mismatches: list[dict[str, Any]] = []

    for evaluation_index, (re_row, sa_re_row) in enumerate(
        zip(re_rows, sa_re_rows, strict=True),
        start=1,
    ):
        re_architecture = _canonical_architecture(
            re_row["architecture"],
            path=re_csv,
        )
        sa_re_architecture = _canonical_architecture(
            sa_re_row["architecture"],
            path=sa_re_csv,
        )
        architecture_equal = re_architecture == sa_re_architecture
        architecture_matches += int(architecture_equal)

        re_training_seed = _parse_int(
            re_row["training_seed"],
            field="training_seed",
            path=re_csv,
        )
        sa_re_training_seed = _parse_int(
            sa_re_row["training_seed"],
            field="training_seed",
            path=sa_re_csv,
        )
        training_seed_equal = re_training_seed == sa_re_training_seed
        training_seed_matches += int(training_seed_equal)

        re_accuracy = _parse_float(
            re_row["final_val_accuracy"],
            field="final_val_accuracy",
            path=re_csv,
        )
        sa_re_accuracy = _parse_float(
            sa_re_row["final_val_accuracy"],
            field="final_val_accuracy",
            path=sa_re_csv,
        )
        accuracy_delta = abs(re_accuracy - sa_re_accuracy)
        accuracy_equal = accuracy_delta <= accuracy_tolerance
        accuracy_matches += int(accuracy_equal)

        if not (architecture_equal and training_seed_equal and accuracy_equal):
            mismatches.append(
                {
                    "evaluation_index": evaluation_index,
                    "architecture_equal": architecture_equal,
                    "training_seed_equal": training_seed_equal,
                    "re_training_seed": re_training_seed,
                    "sa_re_training_seed": sa_re_training_seed,
                    "accuracy_equal": accuracy_equal,
                    "re_final_val_accuracy": re_accuracy,
                    "sa_re_final_val_accuracy": sa_re_accuracy,
                    "accuracy_absolute_delta": accuracy_delta,
                }
            )

    return MatchedInitializationAudit(
        re_path=str(re_csv),
        sa_re_path=str(sa_re_csv),
        search_seed=re_seed,
        population_size=population_size,
        architecture_matches=architecture_matches,
        training_seed_matches=training_seed_matches,
        accuracy_matches=accuracy_matches,
        accuracy_tolerance=accuracy_tolerance,
        require_accuracy_match=require_accuracy_match,
        mismatches=tuple(mismatches),
    )


def _print_report(result: MatchedInitializationAudit) -> None:
    seed = "unavailable" if result.search_seed is None else str(result.search_seed)
    print("Matched Initialization Audit")
    print(f"Search seed:          {seed}")
    print(f"Population size:      {result.population_size}")
    print(
        "Architecture matches: "
        f"{result.architecture_matches}/{result.population_size}"
    )
    print(
        "Training-seed matches: "
        f"{result.training_seed_matches}/{result.population_size}"
    )
    print(
        "Accuracy matches:      "
        f"{result.accuracy_matches}/{result.population_size} "
        f"(atol={result.accuracy_tolerance:g})"
    )
    if not result.require_accuracy_match and not result.accuracy_passed:
        print("Accuracy status:       diagnostic only (not a hard gate)")
    print(f"Result:                {'PASS' if result.passed else 'FAIL'}")
    if result.mismatches:
        indices = [item["evaluation_index"] for item in result.mismatches]
        print(f"Mismatched rows:       {indices}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit matched RE/SA-RE initialization evaluations."
    )
    parser.add_argument("--re", type=Path, required=True)
    parser.add_argument("--sa-re", dest="sa_re", type=Path, required=True)
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--expected-search-seed", type=int, default=None)
    parser.add_argument("--accuracy-tolerance", type=float, default=0.0)
    parser.add_argument("--require-accuracy-match", action="store_true")
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = audit_matched_initialization(
            args.re,
            args.sa_re,
            population_size=args.population_size,
            expected_search_seed=args.expected_search_seed,
            accuracy_tolerance=args.accuracy_tolerance,
            require_accuracy_match=args.require_accuracy_match,
        )
    except AuditError as error:
        print(f"AUDIT ERROR: {error}", file=sys.stderr)
        return 2

    _print_report(result)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
