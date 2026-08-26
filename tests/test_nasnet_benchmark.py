"""Tests for fixed architectures, T1/T5, VRAM, CSV, and stop policy."""

from __future__ import annotations

import csv
import random

import pytest
import torch

import src.benchmark.nasnet_benchmark as benchmark_module
from src.benchmark.nasnet_benchmark import (
    BENCHMARK_CSV_FIELDS,
    BenchmarkRecord,
    benchmark_architecture,
    classify_t5_runtime,
    create_or_load_benchmark_architectures,
    read_benchmark_csv,
    run_benchmark,
    should_stop_after_slow_first_three,
    write_benchmark_csv,
)
from src.nasnet.genotype import random_architecture
from src.training.nasnet_trainer import (
    EpochMetrics,
    TrainingConfig,
    TrainingResult,
)


def _record(architecture_id: int, t5: float) -> BenchmarkRecord:
    return BenchmarkRecord(
        architecture_id=architecture_id,
        parameter_count=1000 + architecture_id,
        T1=t5 / 5.0,
        T5=t5,
        peak_vram=1024,
        epoch1_val_accuracy=0.10,
        epoch5_val_accuracy=0.20,
    )


def _training_result() -> TrainingResult:
    epoch_metrics = tuple(
        EpochMetrics(
            epoch=epoch,
            learning_rate=0.025,
            train_loss=1.0,
            train_accuracy=0.1,
            val_loss=1.0,
            val_accuracy=epoch / 10.0,
            epoch_time_seconds=10.0,
            cumulative_time_seconds=epoch * 10.0,
        )
        for epoch in range(1, 6)
    )
    return TrainingResult(
        final_val_accuracy=0.5,
        best_val_accuracy=0.5,
        final_val_loss=1.0,
        best_epoch=5,
        epoch_metrics=epoch_metrics,
        training_time_seconds=50.0,
        parameter_count=12345,
        amp_enabled=True,
    )


def test_fixed_architectures_are_saved_then_loaded(tmp_path, monkeypatch) -> None:
    first = create_or_load_benchmark_architectures(
        tmp_path,
        seed=20260826,
        count=5,
    )

    assert len(first) == 5
    assert (tmp_path / "architectures.json").is_file()

    def fail_if_regenerated(rng):
        del rng
        raise AssertionError("existing architectures must not be regenerated")

    monkeypatch.setattr(
        benchmark_module,
        "random_architecture",
        fail_if_regenerated,
    )
    second = create_or_load_benchmark_architectures(
        tmp_path,
        seed=20260826,
        count=5,
    )

    assert second == first


@pytest.mark.parametrize(
    ("seconds", "expected"),
    (
        (300.0, "ideal"),
        (300.001, "b60_feasible"),
        (600.0, "b60_feasible"),
        (600.001, "b60_tight"),
        (900.0, "b60_tight"),
        (900.001, "pause_and_optimize"),
    ),
)
def test_t5_decision_boundaries(seconds: float, expected: str) -> None:
    assert classify_t5_runtime(seconds) == expected


def test_csv_contains_required_columns_and_round_trips(tmp_path) -> None:
    records = [_record(0, 100.0), _record(1, 200.0)]
    csv_path = tmp_path / "benchmark.csv"
    write_benchmark_csv(records, csv_path)

    with csv_path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames or ()) == BENCHMARK_CSV_FIELDS

    assert read_benchmark_csv(csv_path) == records


def test_slow_first_three_trigger_early_stop() -> None:
    assert should_stop_after_slow_first_three(
        [_record(0, 901.0), _record(1, 902.0)],
    ) is False
    assert should_stop_after_slow_first_three(
        [_record(0, 901.0), _record(1, 902.0), _record(2, 903.0)],
    ) is True
    assert should_stop_after_slow_first_three(
        [_record(0, 901.0), _record(1, 800.0), _record(2, 903.0)],
    ) is False


def test_architecture_benchmark_uses_same_five_epoch_result(
    monkeypatch,
) -> None:
    architecture = random_architecture(random.Random(20260826))
    cuda_calls = []

    monkeypatch.setattr(
        benchmark_module,
        "build_nasnet",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        benchmark_module,
        "train_nasnet",
        lambda **kwargs: _training_result(),
    )
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device=None: cuda_calls.append(("sync", device)),
    )
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda device=None: cuda_calls.append(("reset", device)),
    )
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_allocated",
        lambda device=None: 987654321,
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    record = benchmark_architecture(
        architecture_id=0,
        architecture=architecture,
        train_loader=object(),
        val_loader=object(),
        training_config=TrainingConfig(epochs=5),
        device=torch.device("cuda"),
    )

    assert record.T1 == 10.0
    assert record.T5 == 50.0
    assert record.T1 <= record.T5
    assert record.peak_vram == 987654321
    assert record.parameter_count == 12345
    assert record.epoch1_val_accuracy == 0.1
    assert record.epoch5_val_accuracy == 0.5
    assert [name for name, _ in cuda_calls].count("sync") == 2
    assert [name for name, _ in cuda_calls].count("reset") == 1


def test_run_stops_after_three_slow_architectures(
    tmp_path,
    monkeypatch,
) -> None:
    architectures = tuple(object() for _ in range(5))
    calls = []

    def fake_benchmark_architecture(architecture_id, **kwargs):
        del kwargs
        calls.append(architecture_id)
        return _record(architecture_id, 901.0 + architecture_id)

    monkeypatch.setattr(
        benchmark_module,
        "benchmark_architecture",
        fake_benchmark_architecture,
    )

    summary = run_benchmark(
        architectures=architectures,
        train_loader=object(),
        val_loader=object(),
        training_config=TrainingConfig(epochs=5),
        device="cpu",
        output_dir=tmp_path,
        early_stop_after=3,
        early_stop_t5_seconds=900.0,
    )

    assert calls == [0, 1, 2]
    assert len(summary.records) == 3
    assert summary.stopped_early is True
    assert summary.decision == "pause_and_optimize"
    assert len(read_benchmark_csv(tmp_path / "benchmark.csv")) == 3
