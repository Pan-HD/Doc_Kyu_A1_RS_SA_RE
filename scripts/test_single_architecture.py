from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.cifar10 import build_cifar10_loaders
from src.search_space.architecture import Architecture
from src.training.trainer import train_and_evaluate


def main():
    arch = Architecture(
        num_conv_blocks=2,
        initial_channels=16,
        channel_multiplier=1,
        kernel_size=3,
        dropout=0.0,
        use_batchnorm=False,
        activation="relu",
        pooling="max",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader, _ = build_cifar10_loaders(
        data_root=ROOT / "data" / "raw",
        split_dir=ROOT / "data" / "splits",
        train_size=20000,
        val_size=5000,
        split_seed=20260823,
        batch_size=128,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    result = train_and_evaluate(
        architecture=arch,
        training_seed=12345,
        epochs=1,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    print(result)


if __name__ == "__main__":
    main()
