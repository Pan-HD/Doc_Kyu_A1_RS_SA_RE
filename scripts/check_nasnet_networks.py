"""Build and forward 20, then cumulatively 100, random full NASNets."""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import torch

from src.nasnet.genotype import random_architecture, validate_architecture
from src.nasnet.network import NASNetCIFAR, build_nasnet


DEFAULT_SEED = 20_260_826
DEFAULT_COUNT = 100
DEFAULT_CHECKPOINT = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate random full NASNet build/forward behavior. The default "
            "run reports a 20-network Gate, then continues to 100 total."
        )
    )
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--N", type=int, default=3)
    parser.add_argument("--F", type=int, default=24)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is unavailable")
    return torch.device(requested)


def parameter_count(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def expected_cell_kinds(N: int) -> tuple[str, ...]:
    kinds = []
    for stack_index in range(3):
        kinds.extend(["normal"] * N)
        if stack_index < 2:
            kinds.append("reduction")
    return tuple(kinds)


def assert_structure(
    model: NASNetCIFAR,
    *,
    N: int,
    F: int,
    num_classes: int,
) -> int:
    expected_normal_cells = 3 * N
    expected_total_cells = expected_normal_cells + 2

    if model.stack_channels != (F, 2 * F, 4 * F):
        raise AssertionError(
            f"filter schedule is {model.stack_channels}, expected "
            f"{(F, 2 * F, 4 * F)}"
        )
    if model.stem_channels != 3 * F:
        raise AssertionError("CIFAR stem multiplier is not 3")
    if model.normal_cell_count != expected_normal_cells:
        raise AssertionError(
            f"normal Cell count is {model.normal_cell_count}, expected "
            f"{expected_normal_cells}"
        )
    if model.reduction_cell_count != 2:
        raise AssertionError(
            f"reduction Cell count is {model.reduction_cell_count}, expected 2"
        )
    if model.total_cell_count != expected_total_cells:
        raise AssertionError(
            f"total Cell count is {model.total_cell_count}, expected "
            f"{expected_total_cells}"
        )
    if model.cell_kinds != expected_cell_kinds(N):
        raise AssertionError(f"unexpected Cell order: {model.cell_kinds}")
    if model.classifier.in_features != model.final_feature_channels:
        raise AssertionError(
            "classifier input does not match dynamically tracked final channels"
        )
    if model.classifier.out_features != num_classes:
        raise AssertionError("classifier output class count is incorrect")

    count = parameter_count(model)
    if count <= 0:
        raise AssertionError("model has no trainable parameters")
    return count


def forward_and_check(
    model: NASNetCIFAR,
    *,
    device: torch.device,
    batch_size: int,
    num_classes: int,
) -> None:
    captured = {}

    def capture_last_cell(_module, _inputs, output):
        captured["shape"] = tuple(output.shape)
        captured["finite"] = bool(torch.isfinite(output).all().item())

    handle = model.cells[-1].register_forward_hook(capture_last_cell)
    inputs = torch.randn(
        batch_size,
        3,
        32,
        32,
        device=device,
    )
    try:
        model.eval()
        with torch.inference_mode():
            logits = model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    finally:
        handle.remove()

    if logits.shape != (batch_size, num_classes):
        raise AssertionError(
            f"logits shape is {tuple(logits.shape)}, expected "
            f"{(batch_size, num_classes)}"
        )
    if not bool(torch.isfinite(logits).all().item()):
        raise AssertionError("logits contain NaN or infinity")
    if "shape" not in captured:
        raise AssertionError("last Cell forward hook did not execute")
    if captured["shape"][-2:] != (8, 8):
        raise AssertionError(
            f"final feature resolution is {captured['shape'][-2:]}, "
            "expected (8, 8)"
        )
    if captured["shape"][1] != model.final_feature_channels:
        raise AssertionError(
            "runtime final channels do not match construction metadata"
        )
    if not captured["finite"]:
        raise AssertionError("final feature map contains NaN or infinity")


def architecture_json(architecture) -> str:
    return json.dumps(
        architecture.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    )


def release_device_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def check_random_networks(args: argparse.Namespace) -> None:
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.checkpoint <= 0:
        raise ValueError("--checkpoint must be positive")
    if args.checkpoint > args.count:
        raise ValueError("--checkpoint cannot exceed --count")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.N <= 0 or args.F <= 0 or args.num_classes <= 0:
        raise ValueError("--N, --F, and --num-classes must be positive")

    device = select_device(args.device)
    architecture_rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(
        f"Configuration: N={args.N}, F={args.F}, "
        f"classes={args.num_classes}, batch_size={args.batch_size}"
    )

    start = time.perf_counter()
    parameter_counts = []

    for architecture_index in range(1, args.count + 1):
        architecture = random_architecture(architecture_rng)
        if not validate_architecture(architecture):
            raise AssertionError(
                f"architecture {architecture_index} failed genotype validation"
            )

        model = None
        try:
            model = build_nasnet(
                architecture,
                N=args.N,
                F=args.F,
                num_classes=args.num_classes,
            ).to(device)
            parameter_counts.append(
                assert_structure(
                    model,
                    N=args.N,
                    F=args.F,
                    num_classes=args.num_classes,
                )
            )
            forward_and_check(
                model,
                device=device,
                batch_size=args.batch_size,
                num_classes=args.num_classes,
            )
        except Exception as error:
            raise AssertionError(
                f"random full network {architecture_index}/{args.count} "
                f"failed; architecture={architecture_json(architecture)}"
            ) from error
        finally:
            if model is not None:
                del model
            release_device_memory(device)

        if architecture_index == args.checkpoint:
            elapsed = time.perf_counter() - start
            print(
                f"{args.checkpoint}/{args.checkpoint} random full networks: "
                f"PASSED ({elapsed:.2f}s)"
            )

    elapsed = time.perf_counter() - start
    if args.count != args.checkpoint:
        print(
            f"{args.count}/{args.count} random full networks: "
            f"PASSED ({elapsed:.2f}s)"
        )
    print(
        "Parameter-count range: "
        f"{min(parameter_counts):,} .. {max(parameter_counts):,}"
    )
    print(
        f"{args.count} random NASNet full-network build/forward checks: PASSED"
    )


def main() -> None:
    check_random_networks(parse_args())


if __name__ == "__main__":
    main()
