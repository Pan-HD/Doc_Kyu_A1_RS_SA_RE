"""Run the fixed five-architecture NASNet T1/T5 benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.benchmark.nasnet_benchmark import (
    create_or_load_benchmark_architectures,
    run_benchmark,
)
from src.data.nasnet_cifar10 import build_cifar10_search_loaders
from src.training.nasnet_trainer import TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed NASNet T1/T5 benchmark."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/nasnet_benchmark.yaml"),
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("benchmark configuration must be a YAML mapping")
    return config


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    experiment = config["experiment"]
    dataset_config = config["dataset"]
    network_config = config["network"]
    training_config = TrainingConfig(**config["training"])
    benchmark_config = config["benchmark"]
    device = torch.device(config.get("device", "cuda"))

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested, but CUDA is unavailable")

    output_dir = Path(experiment["output_dir"])
    architectures = create_or_load_benchmark_architectures(
        output_dir=output_dir,
        seed=int(experiment["architecture_seed"]),
        count=int(experiment["architecture_count"]),
    )

    loaders = build_cifar10_search_loaders(
        data_root=dataset_config["data_root"],
        split_dir=dataset_config["split_dir"],
        batch_size=training_config.batch_size,
        num_workers=int(dataset_config.get("num_workers", 0)),
        pin_memory=bool(dataset_config.get("pin_memory", True)),
        download=bool(dataset_config.get("download", False)),
        augment_train=bool(dataset_config.get("augment_train", True)),
        loader_seed=training_config.training_seed,
    )

    summary = run_benchmark(
        architectures=architectures,
        train_loader=loaders.train_loader,
        val_loader=loaders.val_loader,
        training_config=training_config,
        device=device,
        output_dir=output_dir,
        N=int(network_config["N"]),
        F=int(network_config["F"]),
        num_classes=int(network_config.get("num_classes", 10)),
        early_stop_after=int(benchmark_config["early_stop_after"]),
        early_stop_t5_seconds=(
            float(benchmark_config["early_stop_t5_minutes"])
            * 60.0
        ),
    )

    print(f"architectures: {summary.architectures_path}")
    print(f"benchmark CSV: {summary.benchmark_csv_path}")
    print(f"completed architectures: {len(summary.records)}")
    print(f"mean T5: {summary.mean_T5 / 60.0:.3f} minutes")
    print(f"decision: {summary.decision}")
    print(f"stopped early: {summary.stopped_early}")


if __name__ == "__main__":
    main()
