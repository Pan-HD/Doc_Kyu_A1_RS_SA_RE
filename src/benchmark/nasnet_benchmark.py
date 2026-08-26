"""T1/T5 runtime benchmark for fixed NASNet architectures."""

from __future__ import annotations

import csv
import gc
import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

import torch

from src.nasnet.genotype import (
    NASNetArchitecture,
    random_architecture,
    validate_architecture,
)
from src.nasnet.network import build_nasnet
from src.training.nasnet_trainer import (
    TrainingConfig,
    train_nasnet,
)


BENCHMARK_CSV_FIELDS = (
    "architecture_id",
    "parameter_count",
    "T1",
    "T5",
    "peak_vram",
    "epoch1_val_accuracy",
    "epoch5_val_accuracy",
)

IDEAL_T5_SECONDS = 5 * 60
FEASIBLE_T5_SECONDS = 10 * 60
TIGHT_T5_SECONDS = 15 * 60


@dataclass(frozen=True)
class BenchmarkRecord:
    architecture_id: int
    parameter_count: int
    T1: float
    T5: float
    peak_vram: int
    epoch1_val_accuracy: float
    epoch5_val_accuracy: float

    def to_csv_row(self) -> dict[str, int | float]:
        return {
            "architecture_id": self.architecture_id,
            "parameter_count": self.parameter_count,
            "T1": self.T1,
            "T5": self.T5,
            "peak_vram": self.peak_vram,
            "epoch1_val_accuracy": self.epoch1_val_accuracy,
            "epoch5_val_accuracy": self.epoch5_val_accuracy,
        }

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "BenchmarkRecord":
        return cls(
            architecture_id=int(row["architecture_id"]),
            parameter_count=int(row["parameter_count"]),
            T1=float(row["T1"]),
            T5=float(row["T5"]),
            peak_vram=int(row["peak_vram"]),
            epoch1_val_accuracy=float(row["epoch1_val_accuracy"]),
            epoch5_val_accuracy=float(row["epoch5_val_accuracy"]),
        )


@dataclass(frozen=True)
class BenchmarkSummary:
    records: tuple[BenchmarkRecord, ...]
    mean_T5: float
    decision: str
    stopped_early: bool
    architectures_path: Path
    benchmark_csv_path: Path


def _write_text_atomically(path: Path, text: str) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(path)


def create_or_load_benchmark_architectures(
    output_dir: str | Path,
    seed: int = 20_260_826,
    count: int = 5,
) -> tuple[NASNetArchitecture, ...]:
    """Create the fixed architecture file once, then always load it."""

    if count <= 0:
        raise ValueError("architecture count must be positive")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    architectures_path = output_dir / "architectures.json"

    if architectures_path.is_file():
        payload = json.loads(architectures_path.read_text(encoding="utf-8"))

        if isinstance(payload, list):
            stored_architectures = payload
        else:
            if payload.get("seed") != seed:
                raise ValueError(
                    "stored benchmark seed differs from requested seed; "
                    "the fixed architecture file will not be regenerated"
                )
            stored_architectures = payload.get("architectures", [])

        if len(stored_architectures) != count:
            raise ValueError(
                f"stored architecture count is {len(stored_architectures)}; "
                f"expected {count}"
            )

        architectures = tuple(
            NASNetArchitecture.from_dict(item)
            for item in stored_architectures
        )
    else:
        rng = random.Random(seed)
        architectures = tuple(
            random_architecture(rng)
            for _ in range(count)
        )
        payload = {
            "format_version": 1,
            "seed": seed,
            "count": count,
            "architectures": [
                architecture.to_dict()
                for architecture in architectures
            ],
        }
        _write_text_atomically(
            architectures_path,
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )

    if any(not validate_architecture(item) for item in architectures):
        raise ValueError("benchmark architecture file contains an invalid item")
    return architectures


def write_benchmark_csv(
    records: list[BenchmarkRecord] | tuple[BenchmarkRecord, ...],
    csv_path: str | Path,
) -> None:
    """Atomically write all completed records after each architecture."""

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = csv_path.with_name(csv_path.name + ".tmp")

    with temporary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=BENCHMARK_CSV_FIELDS)
        writer.writeheader()
        for record in sorted(records, key=lambda item: item.architecture_id):
            writer.writerow(record.to_csv_row())

    temporary_path.replace(csv_path)


def read_benchmark_csv(csv_path: str | Path) -> list[BenchmarkRecord]:
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        return []

    with csv_path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != BENCHMARK_CSV_FIELDS:
            raise ValueError("benchmark CSV has unexpected columns")
        records = [BenchmarkRecord.from_csv_row(row) for row in reader]

    ids = [record.architecture_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("benchmark CSV contains duplicate architecture IDs")
    return sorted(records, key=lambda item: item.architecture_id)


def classify_t5_runtime(mean_t5_seconds: float) -> str:
    if mean_t5_seconds < 0:
        raise ValueError("mean T5 must be non-negative")
    if mean_t5_seconds <= IDEAL_T5_SECONDS:
        return "ideal"
    if mean_t5_seconds <= FEASIBLE_T5_SECONDS:
        return "b60_feasible"
    if mean_t5_seconds <= TIGHT_T5_SECONDS:
        return "b60_tight"
    return "pause_and_optimize"


def should_stop_after_slow_first_three(
    records: list[BenchmarkRecord] | tuple[BenchmarkRecord, ...],
    early_stop_after: int = 3,
    threshold_seconds: float = TIGHT_T5_SECONDS,
) -> bool:
    if early_stop_after <= 0:
        raise ValueError("early_stop_after must be positive")

    ordered = sorted(records, key=lambda item: item.architecture_id)
    if len(ordered) < early_stop_after:
        return False

    first_records = ordered[:early_stop_after]
    return all(record.T5 > threshold_seconds for record in first_records)


def benchmark_architecture(
    architecture_id: int,
    architecture: NASNetArchitecture,
    train_loader,
    val_loader,
    training_config: TrainingConfig,
    device: str | torch.device,
    N: int = 3,
    F: int = 24,
    num_classes: int = 10,
) -> BenchmarkRecord:
    """Train once for five epochs and extract T1/T5 from that same run."""

    if training_config.epochs != 5:
        raise ValueError("Part D benchmark requires exactly five epochs")

    device = torch.device(device)
    model = build_nasnet(
        architecture,
        N=N,
        F=F,
        num_classes=num_classes,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    result = train_nasnet(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device,
    )

    if len(result.epoch_metrics) < 5:
        raise RuntimeError("trainer did not return five epoch metrics")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_vram = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_vram = 0

    epoch_1 = result.epoch_metrics[0]
    epoch_5 = result.epoch_metrics[4]
    if epoch_1.cumulative_time_seconds > epoch_5.cumulative_time_seconds:
        raise RuntimeError("T1 exceeds T5; cumulative timing is invalid")

    record = BenchmarkRecord(
        architecture_id=architecture_id,
        parameter_count=result.parameter_count,
        T1=epoch_1.cumulative_time_seconds,
        T5=epoch_5.cumulative_time_seconds,
        peak_vram=peak_vram,
        epoch1_val_accuracy=epoch_1.val_accuracy,
        epoch5_val_accuracy=epoch_5.val_accuracy,
    )

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return record


def run_benchmark(
    architectures: tuple[NASNetArchitecture, ...],
    train_loader,
    val_loader,
    training_config: TrainingConfig,
    device: str | torch.device,
    output_dir: str | Path,
    N: int = 3,
    F: int = 24,
    num_classes: int = 10,
    early_stop_after: int = 3,
    early_stop_t5_seconds: float = TIGHT_T5_SECONDS,
) -> BenchmarkSummary:
    """Run or resume the benchmark, saving CSV after every completed model."""

    if not architectures:
        raise ValueError("at least one benchmark architecture is required")
    if training_config.epochs != 5:
        raise ValueError("Part D benchmark requires exactly five epochs")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    architectures_path = output_dir / "architectures.json"
    benchmark_csv_path = output_dir / "benchmark.csv"
    records = read_benchmark_csv(benchmark_csv_path)

    if any(
        record.architecture_id < 0
        or record.architecture_id >= len(architectures)
        for record in records
    ):
        raise ValueError("benchmark CSV contains an out-of-range architecture ID")

    stopped_early = should_stop_after_slow_first_three(
        records,
        early_stop_after=early_stop_after,
        threshold_seconds=early_stop_t5_seconds,
    )
    completed_ids = {record.architecture_id for record in records}

    for architecture_id, architecture in enumerate(architectures):
        if stopped_early:
            break
        if architecture_id in completed_ids:
            continue

        record = benchmark_architecture(
            architecture_id=architecture_id,
            architecture=architecture,
            train_loader=train_loader,
            val_loader=val_loader,
            training_config=training_config,
            device=device,
            N=N,
            F=F,
            num_classes=num_classes,
        )
        records.append(record)
        records.sort(key=lambda item: item.architecture_id)
        completed_ids.add(architecture_id)
        write_benchmark_csv(records, benchmark_csv_path)

        stopped_early = should_stop_after_slow_first_three(
            records,
            early_stop_after=early_stop_after,
            threshold_seconds=early_stop_t5_seconds,
        )

    if not records:
        raise RuntimeError("benchmark produced no completed records")

    mean_t5 = fmean(record.T5 for record in records)
    return BenchmarkSummary(
        records=tuple(records),
        mean_T5=mean_t5,
        decision=classify_t5_runtime(mean_t5),
        stopped_early=stopped_early,
        architectures_path=architectures_path,
        benchmark_csv_path=benchmark_csv_path,
    )


__all__ = [
    "BENCHMARK_CSV_FIELDS",
    "BenchmarkRecord",
    "BenchmarkSummary",
    "classify_t5_runtime",
    "create_or_load_benchmark_architectures",
    "benchmark_architecture",
    "read_benchmark_csv",
    "run_benchmark",
    "should_stop_after_slow_first_three",
    "write_benchmark_csv",
]
