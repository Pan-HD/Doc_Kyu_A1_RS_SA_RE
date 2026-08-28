"""Run a NASNet surrogate-assisted regularized-evolution experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import torch
import yaml

from src.data.nasnet_cifar10 import build_cifar10_search_loaders
from src.search.nasnet_re import NASNetTrainingEvaluator
from src.search.nasnet_sa_re import run_nasnet_sa_re


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NASNet surrogate-assisted regularized evolution."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/debug/sa_re_debug.yaml"),
    )
    parser.add_argument(
        "--search-seed",
        type=int,
        default=None,
        help="Override experiment.search_seed from the YAML config.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("SA-RE configuration must be a YAML mapping")
    return config


def _select_device(device_config: dict) -> torch.device:
    use_cuda = bool(device_config.get("use_cuda", True))
    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("device.use_cuda=true, but CUDA is unavailable")
        cuda_index = int(device_config.get("cuda_index", 0))
        return torch.device(f"cuda:{cuda_index}")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    experiment = config["experiment"]
    search_seed = (
        int(experiment["search_seed"])
        if args.search_seed is None
        else int(args.search_seed)
    )
    if search_seed < 0:
        raise ValueError("search_seed must be non-negative")
    output_dir = str(experiment["output_dir"]).replace(
        "{search_seed}",
        str(search_seed),
    )
    experiment["search_seed"] = search_seed
    experiment["output_dir"] = output_dir
    resolved_config_text = yaml.safe_dump(
        config,
        sort_keys=False,
        allow_unicode=True,
    )

    dataset = config["dataset"]
    network = config["network"]
    training = dict(config["training"])
    evolution = config["evolution"]
    surrogate = config["surrogate"]
    device = _select_device(config.get("device", {}))

    if str(dataset.get("name", "")).upper() != "CIFAR10":
        raise ValueError("SA-RE supports only dataset.name=CIFAR10")
    if int(dataset["split_seed"]) != 20_260_823:
        raise ValueError("SA-RE requires split_seed=20260823")
    if int(dataset["train_size"]) != 45_000:
        raise ValueError("SA-RE requires train_size=45000")
    if int(dataset["val_size"]) != 5_000:
        raise ValueError("SA-RE requires val_size=5000")

    training_seed_base = int(training.pop("training_seed_base"))
    training.pop("training_seed", None)
    batch_size = int(training.get("batch_size", 128))

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
        # The official test loader is never used during architecture search.
        return loaders.train_loader, loaders.val_loader

    evaluator = NASNetTrainingEvaluator(
        loader_factory=loader_factory,
        training_config_values=training,
        training_seed_base=training_seed_base,
        device=device,
        N=int(network["N"]),
        F=int(network["F"]),
        num_classes=int(network.get("num_classes", 10)),
    )

    result = run_nasnet_sa_re(
        evaluator=evaluator,
        output_dir=output_dir,
        config_text=resolved_config_text,
        method=experiment["method"],
        search_seed=search_seed,
        population_size=int(evolution["population_size"]),
        tournament_size=int(evolution["tournament_size"]),
        budget=int(evolution["budget"]),
        candidate_count=int(evolution["candidate_count"]),
        surrogate_config_values=surrogate,
        overwrite=bool(experiment.get("overwrite", False)),
    )

    print(f"output directory: {result.output_dir}")
    print(f"real evaluations: {result.real_training_runs}")
    print(f"best fitness: {result.best_fitness:.6f}")
    print(f"evaluations CSV: {result.evaluations_csv_path}")
    print(
        "candidate predictions CSV: "
        f"{result.candidate_predictions_csv_path}"
    )
    print(f"history JSON: {result.history_json_path}")


if __name__ == "__main__":
    main()
