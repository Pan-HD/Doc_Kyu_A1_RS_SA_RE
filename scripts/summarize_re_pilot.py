"""Generate a best-so-far CSV from one completed RE pilot run."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate best_so_far.csv for one RE pilot run."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Run directory containing evaluations.csv.",
    )
    return parser.parse_args()


def write_best_so_far(run_dir: Path) -> tuple[Path, int, float]:
    evaluations_path = run_dir / "evaluations.csv"
    output_path = run_dir / "best_so_far.csv"

    with evaluations_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        rows = list(csv.DictReader(stream))

    if not rows:
        raise ValueError("evaluations.csv contains no evaluations")

    required_fields = {"evaluation_index", "final_val_accuracy"}
    missing_fields = required_fields.difference(rows[0])
    if missing_fields:
        raise ValueError(
            "evaluations.csv is missing fields: "
            + ", ".join(sorted(missing_fields))
        )

    rows.sort(key=lambda row: int(row["evaluation_index"]))
    actual_indices = [int(row["evaluation_index"]) for row in rows]
    expected_indices = list(range(1, len(rows) + 1))
    if actual_indices != expected_indices:
        raise ValueError("evaluation indices are not contiguous from 1")

    best = float("-inf")
    output_rows = []
    for row in rows:
        accuracy = float(row["final_val_accuracy"])
        if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
            raise ValueError(
                "final_val_accuracy must be finite and in [0, 1]"
            )
        best = max(best, accuracy)
        output_rows.append(
            {
                "evaluation_index": int(row["evaluation_index"]),
                "best_so_far": best,
            }
        )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("evaluation_index", "best_so_far"),
        )
        writer.writeheader()
        writer.writerows(output_rows)

    return output_path, len(rows), best


def main() -> None:
    args = parse_args()
    output_path, evaluation_count, best = write_best_so_far(args.run_dir)
    print(f"wrote: {output_path}")
    print(f"evaluations: {evaluation_count}")
    print(f"final best-so-far: {best:.6f}")


if __name__ == "__main__":
    main()
