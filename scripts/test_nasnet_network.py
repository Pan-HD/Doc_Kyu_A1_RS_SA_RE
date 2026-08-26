"""Full NASNet CIFAR network tests, including the 20-network Gate."""

from __future__ import annotations

import gc
import json
import random

import torch

from src.nasnet.genotype import random_architecture, validate_architecture
from src.nasnet.network import NASNetCIFAR, build_nasnet


SEED = 20_260_826
N = 3
F = 24
NUM_CLASSES = 10
EXPECTED_STACK_CHANNELS = (24, 48, 96)
EXPECTED_CELL_KINDS = (
    "normal",
    "normal",
    "normal",
    "reduction",
    "normal",
    "normal",
    "normal",
    "reduction",
    "normal",
    "normal",
    "normal",
)


def _test_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def _assert_structure(model: NASNetCIFAR) -> None:
    assert model.N == N
    assert model.F == F
    assert model.stack_channels == EXPECTED_STACK_CHANNELS
    assert model.stem_channels == 3 * F

    assert model.normal_cell_count == 9
    assert model.reduction_cell_count == 2
    assert model.total_cell_count == 11
    assert model.num_normal_cells == 9
    assert model.num_reduction_cells == 2
    assert model.num_cells == 11
    assert len(model.cells) == 11
    assert model.cell_kinds == EXPECTED_CELL_KINDS

    # The classifier must follow the dynamically tracked concatenated Cell
    # output channels; it must not assume that the output width equals F.
    assert model.final_feature_channels > 0
    assert model.classifier.in_features == model.final_feature_channels
    assert model.classifier.out_features == NUM_CLASSES
    assert _parameter_count(model) > 0


def _forward_once(
    model: NASNetCIFAR,
    *,
    device: torch.device,
    batch_size: int,
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

    assert logits.shape == (batch_size, NUM_CLASSES)
    assert torch.isfinite(logits).all()

    assert "shape" in captured
    assert captured["shape"][0] == batch_size
    assert captured["shape"][1] == model.final_feature_channels
    assert captured["shape"][-2:] == (8, 8)
    assert captured["finite"]


def _release_device_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _architecture_json(architecture) -> str:
    return json.dumps(
        architecture.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    )


def test_default_full_network_layout_and_filter_schedule():
    architecture = random_architecture(random.Random(SEED))
    model = build_nasnet(
        architecture,
        N=N,
        F=F,
        num_classes=NUM_CLASSES,
    )

    assert isinstance(model, NASNetCIFAR)
    _assert_structure(model)


def test_full_network_maps_cifar_batch_to_ten_finite_logits():
    device = _test_device()
    architecture = random_architecture(random.Random(SEED))
    model = build_nasnet(
        architecture,
        N=N,
        F=F,
        num_classes=NUM_CLASSES,
    ).to(device)

    try:
        _assert_structure(model)
        _forward_once(model, device=device, batch_size=2)
    finally:
        del model
        _release_device_memory(device)


def test_twenty_random_full_networks_build_and_forward():
    """Required first Gate before the separate 100-network check."""

    device = _test_device()
    architecture_rng = random.Random(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    for architecture_index in range(1, 21):
        architecture = random_architecture(architecture_rng)
        assert validate_architecture(architecture)

        model = None
        try:
            model = build_nasnet(
                architecture,
                N=N,
                F=F,
                num_classes=NUM_CLASSES,
            ).to(device)
            _assert_structure(model)
            _forward_once(model, device=device, batch_size=1)
        except Exception as error:
            raise AssertionError(
                f"random full network {architecture_index}/20 failed; "
                f"architecture={_architecture_json(architecture)}"
            ) from error
        finally:
            if model is not None:
                del model
            _release_device_memory(device)
