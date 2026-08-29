"""Run the one-epoch real NASNet RS-SA-RE debug smoke."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import torch
import yaml

from scripts.check_rs_sa_re_smoke import audit_rs_sa_re_smoke
from src.data.nasnet_cifar10 import build_cifar10_search_loaders
from src.evolution.repeat_policy import RepeatPolicyConfig
from src.search.nasnet_re import NASNetTrainingEvaluator
from src.search.nasnet_rs_sa_re import run_nasnet_rs_sa_re


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the real one-epoch NASNet RS-SA-RE smoke."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/debug/rs_sa_re_debug.yaml"),
    )
    parser.add_argument(
        "--search-seed",
        type=int,
        default=None,
        help="Override experiment.search_seed from the YAML config.",
    )
    parser.add_argument(
        "--repeat-seed",
        type=int,
        default=None,
        help="Override stability.repeat_seed from the YAML config.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("RS-SA-RE configuration must be a YAML mapping")
    return config


def validate_debug_config(config: dict) -> None:
    """Reject accidental drift from the frozen August 29 smoke design."""

    experiment = config["experiment"]
    dataset = config["dataset"]
    network = config["network"]
    training = config["training"]
    evolution = config["evolution"]
    surrogate = config["surrogate"]
    stability = config["stability"]

    if str(experiment["method"]).upper() != "RS-SA-RE":
        raise ValueError("experiment.method must be RS-SA-RE")
    if str(experiment["mode"]).lower() != "debug":
        raise ValueError("the Part F smoke requires experiment.mode=debug")
    if str(dataset.get("name", "")).upper() != "CIFAR10":
        raise ValueError("RS-SA-RE smoke supports only dataset.name=CIFAR10")
    if int(dataset["split_seed"]) != 20_260_823:
        raise ValueError("RS-SA-RE smoke requires split_seed=20260823")
    if int(dataset["train_size"]) != 45_000:
        raise ValueError("RS-SA-RE smoke requires train_size=45000")
    if int(dataset["val_size"]) != 5_000:
        raise ValueError("RS-SA-RE smoke requires val_size=5000")
    if (int(network["N"]), int(network["F"])) != (3, 24):
        raise ValueError("RS-SA-RE smoke requires N=3 and F=24")
    if int(training["epochs"]) != 1:
        raise ValueError("RS-SA-RE smoke must use exactly one epoch")
    if int(training["batch_size"]) != 128:
        raise ValueError("RS-SA-RE smoke requires batch_size=128")

    expected_evolution = {
        "population_size": 20,
        "tournament_size": 5,
        "budget": 30,
        "candidate_count": 5,
    }
    for name, expected in expected_evolution.items():
        if int(evolution[name]) != expected:
            raise ValueError(f"evolution.{name} must equal {expected}")

    if int(surrogate["input_dim"]) != 280:
        raise ValueError("surrogate.input_dim must equal 280")
    if tuple(int(value) for value in surrogate["hidden_dims"]) != (32, 16):
        raise ValueError("surrogate.hidden_dims must equal [32, 16]")
    if str(surrogate["optimizer"]).lower() != "adam":
        raise ValueError("surrogate.optimizer must be Adam")

    policy = RepeatPolicyConfig(
        initial_population_size=int(evolution["population_size"]),
        warmup_pairs=int(stability["warmup_pairs"]),
        repeat_interval=int(stability["repeat_interval"]),
        repeat_rate_beta=float(stability["repeat_rate_beta"]),
    )
    if policy != RepeatPolicyConfig():
        raise ValueError("stability repeat policy differs from the frozen policy")
    penalty_lambda = float(stability["lambda"])
    if not math.isclose(penalty_lambda, 1.0, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("Part F requires debug lambda=1.0")
    if int(stability["repeat_seed"]) < 0:
        raise ValueError("stability.repeat_seed must be non-negative")


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
    stability = config["stability"]
    search_seed = (
        int(experiment["search_seed"])
        if args.search_seed is None
        else int(args.search_seed)
    )
    repeat_seed = (
        int(stability["repeat_seed"])
        if args.repeat_seed is None
        else int(args.repeat_seed)
    )
    if search_seed < 0 or repeat_seed < 0:
        raise ValueError("search and repeat seeds must be non-negative")

    output_dir = str(experiment["output_dir"]).replace(
        "{search_seed}",
        str(search_seed),
    )
    experiment["search_seed"] = search_seed
    experiment["output_dir"] = output_dir
    stability["repeat_seed"] = repeat_seed
    validate_debug_config(config)
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
    repeat_policy = RepeatPolicyConfig(
        initial_population_size=int(evolution["population_size"]),
        warmup_pairs=int(stability["warmup_pairs"]),
        repeat_interval=int(stability["repeat_interval"]),
        repeat_rate_beta=float(stability["repeat_rate_beta"]),
    )

    result = run_nasnet_rs_sa_re(
        evaluator=evaluator,
        output_dir=output_dir,
        config_text=resolved_config_text,
        method=experiment["method"],
        search_seed=search_seed,
        repeat_seed=repeat_seed,
        training_seed_base=training_seed_base,
        population_size=int(evolution["population_size"]),
        tournament_size=int(evolution["tournament_size"]),
        budget=int(evolution["budget"]),
        candidate_count=int(evolution["candidate_count"]),
        stability_penalty_lambda=float(stability["lambda"]),
        surrogate_config_values=surrogate,
        repeat_policy_config=repeat_policy,
        overwrite=bool(experiment.get("overwrite", False)),
    )

    audit = audit_rs_sa_re_smoke(result.output_dir)
    print(f"output directory: {result.output_dir}")
    print(f"real training runs: {audit.real_training_runs}")
    print(f"first evaluations: {audit.first_evaluations}")
    print(f"warm-up repeats: {audit.warmup_repeats}")
    print(f"periodic repeats: {audit.periodic_repeats}")
    print(f"final population: {audit.final_population_size}")
    print(f"candidate rows: {audit.candidate_rows}")
    print(f"best fitness: {result.best_fitness:.6f}")
    print(f"evaluations CSV: {result.evaluations_csv_path}")
    print(
        "candidate predictions CSV: "
        f"{result.candidate_predictions_csv_path}"
    )
    print(f"repeat evaluations CSV: {result.repeat_evaluations_csv_path}")
    print(f"history JSON: {result.history_json_path}")
    print("RS-SA-RE smoke audit: PASS")


if __name__ == "__main__":
    main()
