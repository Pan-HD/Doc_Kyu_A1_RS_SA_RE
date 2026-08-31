from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts.check_stability_diagnostic import audit
from scripts.run_stability_diagnostic import (
    RAW_RESULT_FIELDS,
    generate_frozen_manifest,
    prepare_experiment,
    read_result_rows,
    run_targets,
    upsert_result_row,
    validate_config,
)
from src.search.nasnet_re import NASNetTrainingEvaluator


def _config(output_dir: Path) -> dict:
    return {
        "experiment": {
            "name": "test_stability_diagnostic",
            "output_dir": str(output_dir),
        },
        "diagnostic": {
            "architecture_seed": 8300,
            "architecture_count": 12,
            "training_seeds": [83001, 83002, 83003],
            "milestone_epochs": [5, 25],
            "expected_runs": 36,
        },
        "dataset": {
            "name": "CIFAR10",
            "data_root": "data/cifar10",
            "split_dir": "data/splits/nasnet_v041",
            "split_seed": 20260823,
            "train_size": 45000,
            "val_size": 5000,
            "official_test_size": 10000,
            "download": False,
            "augment_train": True,
            "num_workers": 0,
            "pin_memory": False,
        },
        "network": {"N": 3, "F": 24, "num_classes": 10},
        "training": {
            "epochs": 25,
            "batch_size": 128,
            "optimizer": "SGD",
            "learning_rate": 0.025,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "scheduler": "cosine",
            "gradient_clip_norm": 5.0,
            "amp": False,
            "drop_path": False,
            "auxiliary_head": False,
            "torch_compile": False,
        },
        "device": {"use_cuda": False, "cuda_index": 0},
    }


def _fake_random_architecture(rng: random.Random) -> dict[str, int]:
    return {"token": rng.randrange(1_000_000_000)}


def test_manifest_is_deterministic_unique_and_independent() -> None:
    first, _ = generate_frozen_manifest(
        architecture_seed=8300,
        architecture_count=12,
        random_architecture_fn=_fake_random_architecture,
    )
    second, _ = generate_frozen_manifest(
        architecture_seed=8300,
        architecture_count=12,
        random_architecture_fn=_fake_random_architecture,
    )
    assert first == second
    assert first["frozen"] is True
    assert [item["architecture_id"] for item in first["architectures"]] == [
        f"A{index:02d}" for index in range(1, 13)
    ]
    assert len(
        {
            json.dumps(item["architecture"], sort_keys=True)
            for item in first["architectures"]
        }
    ) == 12


def test_prepare_is_idempotent_and_refuses_architecture_drift(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "diagnostic")
    text = yaml.safe_dump(config, sort_keys=False)
    first = prepare_experiment(
        config,
        text,
        random_architecture_fn=_fake_random_architecture,
    )
    second = prepare_experiment(
        config,
        text,
        random_architecture_fn=_fake_random_architecture,
    )
    assert first[1] == second[1]

    def drifted(rng: random.Random) -> dict[str, int]:
        return {"token": rng.randrange(1_000_000_000) + 1}

    with pytest.raises(RuntimeError, match="refusing to resample"):
        prepare_experiment(
            config,
            text,
            random_architecture_fn=drifted,
        )


def _training_result() -> SimpleNamespace:
    metrics = tuple(
        SimpleNamespace(val_accuracy=0.50 + epoch / 1000.0)
        for epoch in range(1, 26)
    )
    return SimpleNamespace(
        final_val_accuracy=metrics[-1].val_accuracy,
        best_val_accuracy=metrics[-1].val_accuracy,
        parameter_count=12345,
        training_time_seconds=12.5,
        epoch_metrics=metrics,
    )


def _evaluator(*, milestones=()) -> tuple[NASNetTrainingEvaluator, list[int]]:
    calls: list[int] = []

    def trainer(model, train_loader, val_loader, config, device):
        calls.append(1)
        return _training_result()

    evaluator = NASNetTrainingEvaluator(
        loader_factory=lambda seed: ("train", "val"),
        training_config_values={"epochs": 25},
        training_seed_base=100,
        device="cpu",
        model_builder=lambda architecture, N, F, classes: object(),
        training_config_factory=lambda **values: values,
        trainer_fn=trainer,
        milestone_epochs=milestones,
        seed_fn=lambda seed: None,
        cleanup_fn=lambda model, device: None,
    )
    return evaluator, calls


def test_milestones_come_from_one_25_epoch_training_trajectory() -> None:
    evaluator, calls = _evaluator(milestones=(5, 25))
    outcome = evaluator.evaluate_with_seed({"token": 1}, 83001)
    assert calls == [1]
    assert outcome.metadata["accuracy_epoch_5"] == pytest.approx(0.505)
    assert outcome.metadata["accuracy_epoch_25"] == pytest.approx(0.525)
    assert outcome.metadata["accuracy_epoch_25"] == outcome.metadata[
        "final_val_accuracy"
    ]
    assert outcome.metadata["validation_accuracy_by_epoch"] == {
        "5": pytest.approx(0.505),
        "25": pytest.approx(0.525),
    }


def test_default_evaluator_behavior_has_no_diagnostic_metadata() -> None:
    evaluator, calls = _evaluator()
    outcome = evaluator({"token": 1})
    assert calls == [1]
    assert "accuracy_epoch_5" not in outcome.metadata
    assert "validation_accuracy_by_epoch" not in outcome.metadata


def test_raw_results_upsert_never_duplicates_a_run(tmp_path: Path) -> None:
    path = tmp_path / "raw_results.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=RAW_RESULT_FIELDS).writeheader()
    base = {
        "architecture_id": "A01",
        "training_seed": 83001,
        "status": "running",
    }
    upsert_result_row(path, base)
    upsert_result_row(
        path,
        {
            **base,
            "accuracy_epoch_5": 0.7,
            "accuracy_epoch_25": 0.8,
            "training_time": 1.0,
            "parameter_count": 100,
            "status": "completed",
        },
    )
    rows = read_result_rows(path)
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"


def test_smoke_runs_exactly_one_combination_and_passes_audit(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "diagnostic")
    validate_config(config)
    text = yaml.safe_dump(config, sort_keys=False)
    output_dir, manifest, architectures = prepare_experiment(
        config,
        text,
        random_architecture_fn=_fake_random_architecture,
    )
    evaluator, calls = _evaluator(milestones=(5, 25))
    completed, skipped = run_targets(
        config=config,
        output_dir=output_dir,
        manifest=manifest,
        architecture_objects=architectures,
        smoke=True,
        evaluator=evaluator,
    )
    assert (completed, skipped) == (1, 0)
    assert calls == [1]
    assert audit(output_dir, smoke=True)["completed"] == 1
