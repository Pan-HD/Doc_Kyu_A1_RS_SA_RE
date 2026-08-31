"""Audit the frozen NASNet stability diagnostic without training a CNN."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import yaml


FIELDS = (
    "architecture_id",
    "training_seed",
    "accuracy_epoch_5",
    "accuracy_epoch_25",
    "training_time",
    "parameter_count",
    "status",
)
EXPECTED_SEEDS = (83001, 83002, 83003)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def audit(
    experiment_dir: Path,
    *,
    allow_incomplete: bool = False,
    smoke: bool = False,
) -> dict[str, int]:
    config = yaml.safe_load(
        (experiment_dir / "config.yaml").read_text(encoding="utf-8")
    )
    diagnostic = config["diagnostic"]
    if int(diagnostic["architecture_seed"]) != 8300:
        raise AssertionError("architecture_seed must be 8300")
    if int(diagnostic["architecture_count"]) != 12:
        raise AssertionError("architecture_count must be 12")
    if tuple(int(value) for value in diagnostic["training_seeds"]) != EXPECTED_SEEDS:
        raise AssertionError("training seeds differ from the frozen design")
    if tuple(int(value) for value in diagnostic["milestone_epochs"]) != (5, 25):
        raise AssertionError("milestones must be epoch 5 and epoch 25")
    if int(config["training"]["epochs"]) != 25:
        raise AssertionError("each run must train for exactly 25 epochs")

    manifest = json.loads(
        (experiment_dir / "architectures.json").read_text(encoding="utf-8")
    )
    if manifest.get("frozen") is not True:
        raise AssertionError("architectures.json is not frozen")
    if int(manifest["architecture_seed"]) != 8300:
        raise AssertionError("manifest architecture seed differs")
    architectures = manifest["architectures"]
    if len(architectures) != 12:
        raise AssertionError("manifest must contain 12 architectures")
    ids = [item["architecture_id"] for item in architectures]
    if ids != [f"A{index:02d}" for index in range(1, 13)]:
        raise AssertionError("architecture IDs must be A01 through A12")
    serialized = [_canonical(item["architecture"]) for item in architectures]
    if len(set(serialized)) != 12:
        raise AssertionError("diagnostic architectures are not unique")

    results_path = experiment_dir / "raw_results.csv"
    with results_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise AssertionError("raw_results.csv columns differ from the contract")
        rows = list(reader)

    allowed = {(architecture_id, seed) for architecture_id in ids for seed in EXPECTED_SEEDS}
    seen: set[tuple[str, int]] = set()
    completed = 0
    failed = 0
    running = 0
    for row in rows:
        key = (row["architecture_id"], int(row["training_seed"]))
        if key not in allowed:
            raise AssertionError(f"unexpected diagnostic combination: {key}")
        if key in seen:
            raise AssertionError(f"duplicate diagnostic combination: {key}")
        seen.add(key)
        status = row["status"]
        if status == "completed":
            completed += 1
            for name in ("accuracy_epoch_5", "accuracy_epoch_25"):
                value = float(row[name])
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise AssertionError(f"{key} has invalid {name}")
            elapsed = float(row["training_time"])
            if not math.isfinite(elapsed) or elapsed < 0.0:
                raise AssertionError(f"{key} has invalid training_time")
            if int(row["parameter_count"]) <= 0:
                raise AssertionError(f"{key} has invalid parameter_count")

            checkpoint_path = (
                experiment_dir
                / "checkpoints"
                / f"{key[0]}_seed{key[1]}.json"
            )
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint.get("same_training_trajectory") is not True:
                raise AssertionError(f"{key} lacks same-trajectory evidence")
            for name in ("accuracy_epoch_5", "accuracy_epoch_25"):
                if not math.isclose(
                    float(checkpoint[name]),
                    float(row[name]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise AssertionError(f"{key} checkpoint differs for {name}")
        elif status == "failed":
            failed += 1
        elif status == "running":
            running += 1
        else:
            raise AssertionError(f"{key} has invalid status {status!r}")

    if smoke:
        if completed != 1 or seen != {("A01", 83001)}:
            raise AssertionError("smoke must contain exactly A01 x seed 83001")
    elif not allow_incomplete:
        if completed != 36 or seen != allowed or failed or running:
            raise AssertionError(
                "complete diagnostic requires exactly 36 completed combinations"
            )

    return {
        "architectures": len(architectures),
        "rows": len(rows),
        "completed": completed,
        "failed": failed,
        "running": running,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("experiments/stability_diagnostic"),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--allow-incomplete", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = audit(
        args.experiment_dir,
        allow_incomplete=args.allow_incomplete,
        smoke=args.smoke,
    )
    print(
        "stability diagnostic audit: PASS "
        + " ".join(f"{key}={value}" for key, value in summary.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
