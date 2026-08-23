from dataclasses import dataclass
import time

import torch
import torch.nn as nn

from src.models.cnn_builder import build_cnn, count_parameters
from src.utils.seed import set_seed


@dataclass
class EvaluationResult:
    architecture: object
    training_seed: int
    epochs: int
    best_val_accuracy: float
    final_val_accuracy: float
    training_time: float
    parameter_count: int


@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(inputs)
        predictions = logits.argmax(dim=1)

        correct += (predictions == targets).sum().item()
        total += targets.numel()

    return 100.0 * correct / max(total, 1)


def train_and_evaluate(
    architecture,
    training_seed,
    epochs,
    train_loader,
    val_loader,
    device,
    learning_rate=0.05,
    momentum=0.9,
    weight_decay=5e-4,
    deterministic=True,
):
    set_seed(training_seed, deterministic=deterministic)

    model = build_cnn(architecture).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    best_val = float("-inf")
    final_val = float("nan")
    start = time.perf_counter()

    for _ in range(epochs):
        model.train()

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        final_val = evaluate_accuracy(model, val_loader, device)
        best_val = max(best_val, final_val)

    elapsed = time.perf_counter() - start

    return EvaluationResult(
        architecture=architecture,
        training_seed=training_seed,
        epochs=epochs,
        best_val_accuracy=best_val,
        final_val_accuracy=final_val,
        training_time=elapsed,
        parameter_count=count_parameters(model),
    )
