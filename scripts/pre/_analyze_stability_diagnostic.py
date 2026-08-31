"""Analyze the completed 12 x 3 NASNet stability diagnostic.

The script reads only completed rows from the frozen diagnostic contract.  It
requires all 36 architecture/seed combinations before writing either output,
so a partial overnight run can never be mistaken for the final diagnostic.
All standard deviations are sample SDs (ddof=1).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence


EXPECTED_ARCHITECTURE_IDS = tuple(f"A{index:02d}" for index in range(1, 13))
EXPECTED_TRAINING_SEEDS = (83001, 83002, 83003)
SUMMARY_FIELDS = (
    "architecture_id",
    "n_seeds",
    "mean_acc_5",
    "sd_acc_5",
    "mean_acc_25",
    "sd_acc_25",
)
REQUIRED_RAW_FIELDS = (
    "architecture_id",
    "training_seed",
    "accuracy_epoch_5",
    "accuracy_epoch_25",
    "status",
)


@dataclass(frozen=True)
class DiagnosticObservation:
    architecture_id: str
    training_seed: int
    accuracy_epoch_5: float
    accuracy_epoch_25: float


@dataclass(frozen=True)
class ArchitectureSummary:
    architecture_id: str
    n_seeds: int
    mean_acc_5: float
    sd_acc_5: float
    mean_acc_25: float
    sd_acc_25: float

    def to_row(self) -> dict[str, str | int]:
        return {
            "architecture_id": self.architecture_id,
            "n_seeds": self.n_seeds,
            "mean_acc_5": _format_float(self.mean_acc_5),
            "sd_acc_5": _format_float(self.sd_acc_5),
            "mean_acc_25": _format_float(self.mean_acc_25),
            "sd_acc_25": _format_float(self.sd_acc_25),
        }


def _format_float(value: float) -> str:
    return f"{value:.12g}"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def read_completed_observations(path: Path) -> list[DiagnosticObservation]:
    """Read and strictly validate the final 36 completed diagnostic rows."""

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = tuple(reader.fieldnames or ())
        missing = [field for field in REQUIRED_RAW_FIELDS if field not in fields]
        if missing:
            raise ValueError(
                f"{path}: missing required columns: {', '.join(missing)}"
            )
        rows = list(reader)

    expected_keys = {
        (architecture_id, training_seed)
        for architecture_id in EXPECTED_ARCHITECTURE_IDS
        for training_seed in EXPECTED_TRAINING_SEEDS
    }
    seen: set[tuple[str, int]] = set()
    observations: list[DiagnosticObservation] = []
    incomplete: list[tuple[str, int, str]] = []
    for row_number, row in enumerate(rows, start=2):
        architecture_id = str(row["architecture_id"]).strip()
        try:
            training_seed = int(row["training_seed"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{path}:{row_number}: invalid training_seed"
            ) from error
        key = (architecture_id, training_seed)
        if key not in expected_keys:
            raise ValueError(f"{path}:{row_number}: unexpected run key {key}")
        if key in seen:
            raise ValueError(f"{path}:{row_number}: duplicate run key {key}")
        seen.add(key)

        status = str(row["status"]).strip().lower()
        if status != "completed":
            incomplete.append((architecture_id, training_seed, status))
            continue
        try:
            accuracy_5 = float(row["accuracy_epoch_5"])
            accuracy_25 = float(row["accuracy_epoch_25"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{path}:{row_number}: completed row has non-numeric accuracy"
            ) from error
        for label, value in (("Acc@5", accuracy_5), ("Acc@25", accuracy_25)):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{path}:{row_number}: {label} must be finite and in [0, 1]"
                )
        observations.append(
            DiagnosticObservation(
                architecture_id=architecture_id,
                training_seed=training_seed,
                accuracy_epoch_5=accuracy_5,
                accuracy_epoch_25=accuracy_25,
            )
        )

    missing_keys = sorted(expected_keys - seen)
    if missing_keys or incomplete:
        details: list[str] = []
        if missing_keys:
            details.append(f"missing={len(missing_keys)}")
        if incomplete:
            details.append(f"non_completed={len(incomplete)}")
        raise RuntimeError(
            "stability diagnostic is incomplete; "
            + ", ".join(details)
            + ". Refusing to write final analysis."
        )
    if len(observations) != 36:
        raise RuntimeError("expected exactly 36 completed observations")
    return observations


def summarize_architectures(
    observations: Iterable[DiagnosticObservation],
) -> list[ArchitectureSummary]:
    grouped: dict[str, list[DiagnosticObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.architecture_id].append(observation)

    if set(grouped) != set(EXPECTED_ARCHITECTURE_IDS):
        raise ValueError("observations must contain exactly A01 through A12")
    summaries: list[ArchitectureSummary] = []
    for architecture_id in EXPECTED_ARCHITECTURE_IDS:
        group = sorted(grouped[architecture_id], key=lambda item: item.training_seed)
        seeds = tuple(item.training_seed for item in group)
        if seeds != EXPECTED_TRAINING_SEEDS:
            raise ValueError(
                f"{architecture_id} must contain seeds {EXPECTED_TRAINING_SEEDS}"
            )
        accuracy_5 = [item.accuracy_epoch_5 for item in group]
        accuracy_25 = [item.accuracy_epoch_25 for item in group]
        summaries.append(
            ArchitectureSummary(
                architecture_id=architecture_id,
                n_seeds=len(group),
                mean_acc_5=statistics.fmean(accuracy_5),
                sd_acc_5=statistics.stdev(accuracy_5),
                mean_acc_25=statistics.fmean(accuracy_25),
                sd_acc_25=statistics.stdev(accuracy_25),
            )
        )
    return summaries


def _average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for offset in range(start, end):
            ranks[indexed[offset][0]] = average_rank
        start = end
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("correlation requires equal sequences with n >= 3")
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    centered_x = [value - mean_x for value in x]
    centered_y = [value - mean_y for value in y]
    sum_x2 = sum(value * value for value in centered_x)
    sum_y2 = sum(value * value for value in centered_y)
    if sum_x2 == 0.0 or sum_y2 == 0.0:
        raise ValueError("Spearman correlation is undefined for a constant input")
    rho = sum(a * b for a, b in zip(centered_x, centered_y)) / math.sqrt(
        sum_x2 * sum_y2
    )
    return max(-1.0, min(1.0, rho))


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    max_iterations = 200
    epsilon = 3.0e-14
    tiny = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d
    for iteration in range(1, max_iterations + 1):
        even = 2 * iteration
        coefficient = (
            iteration * (b - iteration) * x
            / ((qam + even) * (a + even))
        )
        d = 1.0 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c

        coefficient = -(
            (a + iteration)
            * (qab + iteration)
            * x
            / ((a + even) * (qap + even))
        )
        d = 1.0 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < epsilon:
            return result
    raise ArithmeticError("incomplete beta continued fraction did not converge")


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if not 0.0 <= x <= 1.0:
        raise ValueError("x must be in [0, 1]")
    if x in (0.0, 1.0):
        return x
    log_factor = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    factor = math.exp(log_factor)
    if x < (a + 1.0) / (a + b + 2.0):
        return factor * _beta_continued_fraction(a, b, x) / a
    return 1.0 - factor * _beta_continued_fraction(b, a, 1.0 - x) / b


def spearman_with_p_value(
    x: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float]:
    """Return Spearman rho and the usual two-sided t-approximation p-value."""

    rho = _pearson(_average_ranks(x), _average_ranks(y))
    n = len(x)
    if math.isclose(abs(rho), 1.0, rel_tol=0.0, abs_tol=1e-15):
        return rho, 0.0
    degrees_of_freedom = n - 2
    denominator = max(1.0e-300, 1.0 - rho * rho)
    t_squared = rho * rho * degrees_of_freedom / denominator
    p_value = _regularized_incomplete_beta(
        degrees_of_freedom / 2.0,
        0.5,
        degrees_of_freedom / (degrees_of_freedom + t_squared),
    )
    return rho, max(0.0, min(1.0, p_value))


def write_summary(path: Path, summaries: Sequence[ArchitectureSummary]) -> None:
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
    writer.writeheader()
    writer.writerows(summary.to_row() for summary in summaries)
    _atomic_write_text(path, stream.getvalue())


def analyze(
    input_path: Path,
    summary_path: Path,
    correlation_path: Path,
) -> dict[str, object]:
    observations = read_completed_observations(input_path)
    summaries = summarize_architectures(observations)
    sd_5 = [summary.sd_acc_5 for summary in summaries]
    sd_25 = [summary.sd_acc_25 for summary in summaries]
    rho, p_value = spearman_with_p_value(sd_5, sd_25)
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "source": str(input_path),
        "metric": "spearman",
        "x": "sd_acc_5",
        "y": "sd_acc_25",
        "sd_definition": "sample standard deviation (ddof=1)",
        "rho": rho,
        "p_value": p_value,
        "n": len(summaries),
        "direction": "positive" if rho > 0.0 else "negative" if rho < 0.0 else "zero",
        "interpretation": "review_on_2026-08-31_without_fixed_go_threshold",
        "go_threshold": None,
    }

    # Write only after every calculation succeeds.
    write_summary(summary_path, summaries)
    _atomic_write_text(
        correlation_path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("experiments/stability_diagnostic/raw_results.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("experiments/stability_diagnostic/stability_summary.csv"),
    )
    parser.add_argument(
        "--correlation-output",
        type=Path,
        default=Path(
            "experiments/stability_diagnostic/stability_correlation.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = analyze(args.input, args.summary_output, args.correlation_output)
    print(
        "stability diagnostic analysis: PASS "
        f"n={payload['n']} rho={payload['rho']:.6f} "
        f"p_value={payload['p_value']:.6g}"
    )
    print(f"summary: {args.summary_output}")
    print(f"correlation: {args.correlation_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
