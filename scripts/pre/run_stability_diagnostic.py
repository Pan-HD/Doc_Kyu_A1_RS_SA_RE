"""Prepare and run the frozen 12 x 3 NASNet stability diagnostic.

Every real run initializes a model once, trains continuously for 25 epochs,
and reads Acc@5 and Acc@25 from that single trainer result's epoch_metrics.
The script deliberately has no implicit full-run mode: choose --prepare-only,
--smoke, or --resume so an accidental command cannot start 36 GPU runs.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import random
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import yaml


RAW_RESULT_FIELDS = (
    "architecture_id",
    "training_seed",
    "accuracy_epoch_5",
    "accuracy_epoch_25",
    "training_time",
    "parameter_count",
    "status",
)
VALID_STATUSES = {"running", "completed", "failed"}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{_timestamp()} {message}\n")
    print(message, flush=True)


def _architecture_to_dict(architecture: Any) -> dict[str, Any]:
    if hasattr(architecture, "to_dict"):
        value = architecture.to_dict()
    elif is_dataclass(architecture):
        value = asdict(architecture)
    elif isinstance(architecture, Mapping):
        value = dict(architecture)
    else:
        raise TypeError(
            "architecture must provide to_dict(), be a dataclass, or be a mapping"
        )
    if not isinstance(value, Mapping):
        raise TypeError("architecture serialization must produce a mapping")
    # This round trip rejects non-JSON values and normalizes nested mappings.
    normalized = json.loads(json.dumps(value, sort_keys=True))
    if not isinstance(normalized, dict):
        raise TypeError("architecture serialization must produce a JSON object")
    return normalized


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    config = yaml.safe_load(text)
    if not isinstance(config, dict):
        raise ValueError("diagnostic config must be a YAML mapping")
    validate_config(config)
    return config, text


def validate_config(config: Mapping[str, Any]) -> None:
    diagnostic = config["diagnostic"]
    dataset = config["dataset"]
    network = config["network"]
    training = config["training"]

    if int(diagnostic["architecture_seed"]) != 8300:
        raise ValueError("diagnostic.architecture_seed must equal 8300")
    if int(diagnostic["architecture_count"]) != 12:
        raise ValueError("diagnostic.architecture_count must equal 12")
    seeds = tuple(int(value) for value in diagnostic["training_seeds"])
    if seeds != (83001, 83002, 83003):
        raise ValueError(
            "diagnostic.training_seeds must equal [83001, 83002, 83003]"
        )
    milestones = tuple(int(value) for value in diagnostic["milestone_epochs"])
    if milestones != (5, 25):
        raise ValueError("diagnostic.milestone_epochs must equal [5, 25]")
    if int(diagnostic["expected_runs"]) != 36:
        raise ValueError("diagnostic.expected_runs must equal 36")
    if str(dataset.get("name", "")).upper() != "CIFAR10":
        raise ValueError("dataset.name must be CIFAR10")
    if int(dataset["split_seed"]) != 20_260_823:
        raise ValueError("dataset.split_seed must equal 20260823")
    if (int(dataset["train_size"]), int(dataset["val_size"])) != (
        45_000,
        5_000,
    ):
        raise ValueError("the diagnostic requires the frozen 45k/5k split")
    if (int(network["N"]), int(network["F"])) != (3, 24):
        raise ValueError("network N/F must equal 3/24")
    if int(training["epochs"]) != 25:
        raise ValueError("training.epochs must equal 25")
    if int(training["batch_size"]) != 128:
        raise ValueError("training.batch_size must equal 128")


def generate_frozen_manifest(
    *,
    architecture_seed: int,
    architecture_count: int,
    random_architecture_fn: Callable[[random.Random], Any],
) -> tuple[dict[str, Any], list[Any]]:
    """Generate the independent set once, rejecting natural duplicates."""

    rng = random.Random(architecture_seed)
    architectures: list[Any] = []
    serialized: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = architecture_count * 1000
    while len(architectures) < architecture_count:
        if attempts >= max_attempts:
            raise RuntimeError("could not sample enough unique architectures")
        attempts += 1
        architecture = random_architecture_fn(rng)
        architecture_dict = _architecture_to_dict(architecture)
        key = _canonical_json(architecture_dict)
        if key in seen:
            continue
        seen.add(key)
        architectures.append(architecture)
        serialized.append(
            {
                "architecture_id": f"A{len(architectures):02d}",
                "architecture": architecture_dict,
            }
        )

    manifest = {
        "schema_version": 1,
        "frozen": True,
        "architecture_seed": architecture_seed,
        "architecture_count": architecture_count,
        "sampling_attempts": attempts,
        "architectures": serialized,
    }
    return manifest, architectures


def _load_or_freeze_manifest(
    path: Path,
    expected: Mapping[str, Any],
) -> None:
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        is_empty_template = (
            isinstance(current, dict)
            and current.get("frozen") is False
            and current.get("architectures") == []
        )
        if not is_empty_template and current != expected:
            raise RuntimeError(
                f"{path} differs from seed-regenerated architectures; "
                "refusing to resample or overwrite the frozen set"
            )
        if current == expected:
            return
    _atomic_write_text(
        path,
        json.dumps(expected, indent=2, sort_keys=True) + "\n",
    )


def _ensure_frozen_config(path: Path, config: Mapping[str, Any], text: str) -> None:
    if path.exists():
        current = yaml.safe_load(path.read_text(encoding="utf-8"))
        if current != config:
            raise RuntimeError(
                f"{path} differs from the requested diagnostic config"
            )
        return
    _atomic_write_text(path, text.rstrip() + "\n")


def _empty_results_text() -> str:
    stream = io.StringIO(newline="")
    csv.DictWriter(
        stream,
        fieldnames=RAW_RESULT_FIELDS,
        lineterminator="\n",
    ).writeheader()
    return stream.getvalue()


def read_result_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != RAW_RESULT_FIELDS:
            raise ValueError(
                f"{path} must have exactly these columns: "
                + ", ".join(RAW_RESULT_FIELDS)
            )
        rows = [dict(row) for row in reader]
    keys: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["architecture_id"], row["training_seed"])
        if key in keys:
            raise ValueError(f"{path} contains duplicate run keys: {key}")
        keys.add(key)
        if row["status"] not in VALID_STATUSES:
            raise ValueError(f"invalid result status: {row['status']}")
    return rows


def upsert_result_row(path: Path, row: Mapping[str, Any]) -> None:
    normalized = {field: str(row.get(field, "")) for field in RAW_RESULT_FIELDS}
    if normalized["status"] not in VALID_STATUSES:
        raise ValueError("result status is invalid")
    rows = read_result_rows(path)
    key = (normalized["architecture_id"], normalized["training_seed"])
    by_key = {
        (item["architecture_id"], item["training_seed"]): item
        for item in rows
    }
    by_key[key] = normalized

    def sort_key(item: dict[str, str]) -> tuple[int, int]:
        return int(item["architecture_id"][1:]), int(item["training_seed"])

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=RAW_RESULT_FIELDS,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(sorted(by_key.values(), key=sort_key))
    _atomic_write_text(path, stream.getvalue())


def prepare_experiment(
    config: Mapping[str, Any],
    config_text: str,
    *,
    random_architecture_fn: Callable[[random.Random], Any] | None = None,
) -> tuple[Path, dict[str, Any], list[Any]]:
    if random_architecture_fn is None:
        from src.nasnet.genotype import random_architecture

        random_architecture_fn = random_architecture

    diagnostic = config["diagnostic"]
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    manifest, architecture_objects = generate_frozen_manifest(
        architecture_seed=int(diagnostic["architecture_seed"]),
        architecture_count=int(diagnostic["architecture_count"]),
        random_architecture_fn=random_architecture_fn,
    )
    _load_or_freeze_manifest(output_dir / "architectures.json", manifest)
    _ensure_frozen_config(output_dir / "config.yaml", config, config_text)

    results_path = output_dir / "raw_results.csv"
    if not results_path.exists():
        _atomic_write_text(results_path, _empty_results_text())
    else:
        read_result_rows(results_path)
    log_path = output_dir / "run.log"
    if not log_path.exists():
        _atomic_write_text(log_path, "")
    _append_log(
        log_path,
        "PREPARED frozen_architectures=12 architecture_seed=8300 "
        "training_seeds=83001,83002,83003 epochs=25 milestones=5,25",
    )
    return output_dir, manifest, architecture_objects


def _select_device(device_config: Mapping[str, Any]):
    import torch

    if bool(device_config.get("use_cuda", True)):
        if not torch.cuda.is_available():
            raise RuntimeError("device.use_cuda=true, but CUDA is unavailable")
        return torch.device(f"cuda:{int(device_config.get('cuda_index', 0))}")
    return torch.device("cpu")


def build_evaluator(config: Mapping[str, Any]):
    from src.data.nasnet_cifar10 import build_cifar10_search_loaders
    from src.search.nasnet_re import NASNetTrainingEvaluator

    dataset = config["dataset"]
    network = config["network"]
    training = dict(config["training"])
    diagnostic = config["diagnostic"]
    device = _select_device(config.get("device", {}))
    training.pop("training_seed", None)
    training.pop("training_seed_base", None)
    batch_size = int(training["batch_size"])

    def loader_factory(training_seed: int):
        loaders = build_cifar10_search_loaders(
            data_root=dataset["data_root"],
            split_dir=dataset["split_dir"],
            batch_size=batch_size,
            num_workers=int(dataset.get("num_workers", 0)),
            pin_memory=bool(dataset.get("pin_memory", True)),
            download=bool(dataset.get("download", False)),
            augment_train=bool(dataset.get("augment_train", True)),
            loader_seed=training_seed,
        )
        return loaders.train_loader, loaders.val_loader

    return NASNetTrainingEvaluator(
        loader_factory=loader_factory,
        training_config_values=training,
        training_seed_base=0,
        device=device,
        N=int(network["N"]),
        F=int(network["F"]),
        num_classes=int(network.get("num_classes", 10)),
        milestone_epochs=tuple(
            int(epoch) for epoch in diagnostic["milestone_epochs"]
        ),
    )


def _validate_completed_values(metadata: Mapping[str, Any]) -> None:
    for key in ("accuracy_epoch_5", "accuracy_epoch_25"):
        value = float(metadata[key])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} must be finite and in [0, 1]")
    if not math.isclose(
        float(metadata["accuracy_epoch_25"]),
        float(metadata["final_val_accuracy"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Acc@25 is not the same-trajectory final accuracy")
    if int(metadata["parameter_count"]) <= 0:
        raise ValueError("parameter_count must be positive")
    elapsed = float(metadata["training_time_seconds"])
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("training_time must be finite and non-negative")


def _write_run_checkpoint(
    path: Path,
    *,
    architecture_id: str,
    training_seed: int,
    metadata: Mapping[str, Any],
) -> None:
    payload = {
        "architecture_id": architecture_id,
        "training_seed": training_seed,
        "epochs": 25,
        "same_training_trajectory": True,
        "accuracy_epoch_5": float(metadata["accuracy_epoch_5"]),
        "accuracy_epoch_25": float(metadata["accuracy_epoch_25"]),
        "training_time": float(metadata["training_time_seconds"]),
        "parameter_count": int(metadata["parameter_count"]),
        "status": "completed",
    }
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_targets(
    *,
    config: Mapping[str, Any],
    output_dir: Path,
    manifest: Mapping[str, Any],
    architecture_objects: Sequence[Any],
    smoke: bool,
    evaluator: Any | None = None,
    continue_on_error: bool = False,
) -> tuple[int, int]:
    results_path = output_dir / "raw_results.csv"
    log_path = output_dir / "run.log"
    seeds = tuple(int(value) for value in config["diagnostic"]["training_seeds"])
    targets = [(0, seeds[0])] if smoke else [
        (architecture_index, training_seed)
        for architecture_index in range(len(architecture_objects))
        for training_seed in seeds
    ]
    evaluator = evaluator or build_evaluator(config)
    completed = 0
    skipped = 0

    for architecture_index, training_seed in targets:
        architecture_id = manifest["architectures"][architecture_index][
            "architecture_id"
        ]
        existing = {
            (row["architecture_id"], int(row["training_seed"])): row
            for row in read_result_rows(results_path)
        }
        key = (architecture_id, training_seed)
        if existing.get(key, {}).get("status") == "completed":
            skipped += 1
            _append_log(
                log_path,
                f"SKIP architecture_id={architecture_id} "
                f"training_seed={training_seed} status=completed",
            )
            continue

        upsert_result_row(
            results_path,
            {
                "architecture_id": architecture_id,
                "training_seed": training_seed,
                "status": "running",
            },
        )
        _append_log(
            log_path,
            f"START architecture_id={architecture_id} "
            f"training_seed={training_seed} epochs=25",
        )
        try:
            outcome = evaluator.evaluate_with_seed(
                architecture_objects[architecture_index],
                training_seed,
            )
            metadata = outcome.metadata
            _validate_completed_values(metadata)
            row = {
                "architecture_id": architecture_id,
                "training_seed": training_seed,
                "accuracy_epoch_5": float(metadata["accuracy_epoch_5"]),
                "accuracy_epoch_25": float(metadata["accuracy_epoch_25"]),
                "training_time": float(metadata["training_time_seconds"]),
                "parameter_count": int(metadata["parameter_count"]),
                "status": "completed",
            }
            upsert_result_row(results_path, row)
            _write_run_checkpoint(
                output_dir
                / "checkpoints"
                / f"{architecture_id}_seed{training_seed}.json",
                architecture_id=architecture_id,
                training_seed=training_seed,
                metadata=metadata,
            )
            completed += 1
            _append_log(
                log_path,
                f"DONE architecture_id={architecture_id} "
                f"training_seed={training_seed} "
                f"acc5={row['accuracy_epoch_5']:.6f} "
                f"acc25={row['accuracy_epoch_25']:.6f} "
                f"seconds={row['training_time']:.3f}",
            )
        except Exception as error:
            upsert_result_row(
                results_path,
                {
                    "architecture_id": architecture_id,
                    "training_seed": training_seed,
                    "status": "failed",
                },
            )
            _append_log(
                log_path,
                f"FAILED architecture_id={architecture_id} "
                f"training_seed={training_seed} "
                f"error={type(error).__name__}: {error}",
            )
            if not continue_on_error:
                raise

    _append_log(
        log_path,
        f"SESSION completed={completed} skipped={skipped} targets={len(targets)}",
    )
    return completed, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/diagnostic/stability_diagnostic.yaml"),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="Run only A01 x seed 83001 for 25 epochs.",
    )
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Run all pending combinations; completed rows are skipped.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Log a failed combination and continue with later runs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, config_text = load_config(args.config)
    output_dir, manifest, architecture_objects = prepare_experiment(
        config,
        config_text,
    )
    if args.prepare_only:
        print(f"architectures frozen: {output_dir / 'architectures.json'}")
        print(f"resolved config: {output_dir / 'config.yaml'}")
        return 0

    run_targets(
        config=config,
        output_dir=output_dir,
        manifest=manifest,
        architecture_objects=architecture_objects,
        smoke=args.smoke,
        continue_on_error=args.continue_on_error,
    )
    print(f"raw results: {output_dir / 'raw_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
