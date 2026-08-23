from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.cifar10 import build_cifar10_loaders
from src.evolution.regularized_evolution import run_regularized_evolution
from src.training.trainer import train_and_evaluate
from src.utils.io import load_yaml
from src.utils.logger import save_config_copy, save_search_history


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)

    use_cuda = config.get("device", {}).get("use_cuda", True)
    device = torch.device(
        "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    ds_cfg = config["dataset"]
    tr_cfg = config["training"]
    ev_cfg = config["evolution"]
    exp_cfg = config["experiment"]

    train_loader, val_loader, _ = build_cifar10_loaders(
        data_root=ROOT / "data" / "raw",
        split_dir=ROOT / "data" / "splits",
        train_size=ds_cfg["train_size"],
        val_size=ds_cfg["val_size"],
        split_seed=ds_cfg["split_seed"],
        batch_size=tr_cfg["batch_size"],
        num_workers=ds_cfg.get("num_workers", 2),
        pin_memory=(device.type == "cuda"),
    )

    def evaluate_fn(architecture, training_seed):
        return train_and_evaluate(
            architecture=architecture,
            training_seed=training_seed,
            epochs=tr_cfg["epochs"],
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=tr_cfg["learning_rate"],
            momentum=tr_cfg["momentum"],
            weight_decay=tr_cfg["weight_decay"],
            deterministic=tr_cfg.get("deterministic", True),
        )

    _, history = run_regularized_evolution(
        evaluate_fn=evaluate_fn,
        population_size=ev_cfg["population_size"],
        tournament_size=ev_cfg["tournament_size"],
        budget=ev_cfg["budget"],
        search_seed=exp_cfg["search_seed"],
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "experiments" / exp_cfg["mode"] / f"re_{timestamp}"

    save_config_copy(config, out_dir)
    save_search_history(
        history,
        out_dir,
        metadata={
            "method": "RE",
            "search_seed": exp_cfg["search_seed"],
        },
    )

    best = max(history, key=lambda r: r.validation_accuracy)
    print(f"RE search completed. Budget = {len(history)}")
    print(f"Best validation accuracy = {best.validation_accuracy:.3f}%")
    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
