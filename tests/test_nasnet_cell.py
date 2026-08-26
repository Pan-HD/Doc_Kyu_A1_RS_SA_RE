"""Structural tests for the generic PyTorch NASNet cell.

These tests cover the 2026-08-26 definition of done:

* current/previous input projection;
* factorized reduction for a spatially mismatched previous input;
* Normal Cell shape and variable output-channel count;
* Reduction Cell one-time spatial reduction;
* alignment of unused original states 0/1 before output concatenation;
* cell output metadata; and
* 100 deterministic random Normal/Reduction architecture forwards.

Drop-path and training behavior are intentionally outside this test module.
"""

from __future__ import annotations

import random
from dataclasses import replace

import pytest
import torch

from src.nasnet.cell import FactorizedReduction, NASNetCell, Projection
from src.nasnet.genotype import get_unused_states, random_architecture


SEED = 20260825
BATCH_SIZE = 4
INPUT_SIZE = 32
BASE_CHANNELS = 24
REDUCTION_CHANNELS = 48
STRESS_ARCHITECTURES = 100


def _assert_finite(tensor: torch.Tensor) -> None:
    """Fail with a concise message if a structural forward produced NaN/Inf."""

    assert torch.isfinite(tensor).all(), "cell output contains NaN or Inf"


def _expected_output_channels(gene, cell_channels: int) -> int:
    """Derive concat channels from every unused hidden state in 0..6."""

    unused_indices = get_unused_states(gene)
    assert len(unused_indices) >= 1
    assert all(index in range(7) for index in unused_indices)
    return len(unused_indices) * cell_channels


def _replace_input_state(gene, old_state: int, new_state: int):
    """Return a frozen CellGene with one input-state value replaced."""

    pairs = []

    for pair in gene.pairs:
        branch_1 = pair.branch_1
        branch_2 = pair.branch_2

        if branch_1.input_state == old_state:
            branch_1 = replace(branch_1, input_state=new_state)
        if branch_2.input_state == old_state:
            branch_2 = replace(branch_2, input_state=new_state)

        pairs.append(
            replace(
                pair,
                branch_1=branch_1,
                branch_2=branch_2,
            )
        )

    return replace(gene, pairs=tuple(pairs))


def test_projection_changes_channels_without_resizing() -> None:
    projection = Projection(
        in_channels=48,
        out_channels=24,
    )
    projection.eval()

    x = torch.randn(BATCH_SIZE, 48, INPUT_SIZE, INPUT_SIZE)

    with torch.no_grad():
        y = projection(x)

    assert y.shape == (BATCH_SIZE, 24, INPUT_SIZE, INPUT_SIZE)
    _assert_finite(y)


def test_factorized_reduction_halves_spatial_size() -> None:
    reduction = FactorizedReduction(
        in_channels=48,
        out_channels=24,
    )
    reduction.eval()

    x = torch.randn(BATCH_SIZE, 48, INPUT_SIZE, INPUT_SIZE)

    with torch.no_grad():
        y = reduction(x)

    assert y.shape == (
        BATCH_SIZE,
        24,
        INPUT_SIZE // 2,
        INPUT_SIZE // 2,
    )
    _assert_finite(y)


def test_normal_cell_preserves_spatial_size_and_concatenates_unused_states(
) -> None:
    rng = random.Random(SEED)
    arch = random_architecture(rng)

    cell = NASNetCell(
        gene=arch.normal,
        prev_channels=BASE_CHANNELS,
        curr_channels=BASE_CHANNELS,
        cell_channels=BASE_CHANNELS,
        reduction=False,
    )
    cell.eval()

    s0 = torch.randn(
        BATCH_SIZE,
        BASE_CHANNELS,
        INPUT_SIZE,
        INPUT_SIZE,
    )
    s1 = torch.randn_like(s0)

    with torch.no_grad():
        y = cell(s0, s1)

    expected_channels = _expected_output_channels(
        arch.normal,
        BASE_CHANNELS,
    )

    assert y.shape == (
        BATCH_SIZE,
        expected_channels,
        INPUT_SIZE,
        INPUT_SIZE,
    )
    assert cell.output_multiplier == len(get_unused_states(arch.normal))
    assert cell.output_channels == expected_channels
    _assert_finite(y)


@pytest.mark.parametrize(
    ("unused_state", "replacement_state"),
    ((0, 1), (1, 0)),
)
def test_normal_cell_concatenates_unused_original_state(
    unused_state: int,
    replacement_state: int,
) -> None:
    """A Normal Cell can directly concatenate unused state 0 or state 1."""

    rng = random.Random(SEED)
    gene = _replace_input_state(
        random_architecture(rng).normal,
        old_state=unused_state,
        new_state=replacement_state,
    )
    unused_indices = get_unused_states(gene)

    assert unused_state in unused_indices

    cell = NASNetCell(
        gene=gene,
        prev_channels=BASE_CHANNELS,
        curr_channels=BASE_CHANNELS,
        cell_channels=BASE_CHANNELS,
        reduction=False,
    )
    cell.eval()

    s0 = torch.randn(
        BATCH_SIZE,
        BASE_CHANNELS,
        INPUT_SIZE,
        INPUT_SIZE,
    )
    s1 = torch.randn_like(s0)

    with torch.no_grad():
        y = cell(s0, s1)

    expected_channels = len(unused_indices) * BASE_CHANNELS

    assert y.shape == (
        BATCH_SIZE,
        expected_channels,
        INPUT_SIZE,
        INPUT_SIZE,
    )
    assert cell.output_multiplier == len(unused_indices)
    assert cell.output_channels == expected_channels

    alignment_index = unused_indices.index(unused_state)
    assert isinstance(
        cell.output_alignments[alignment_index],
        torch.nn.Identity,
    )
    _assert_finite(y)


def test_reduction_cell_halves_spatial_size_once() -> None:
    rng = random.Random(SEED)
    arch = random_architecture(rng)

    cell = NASNetCell(
        gene=arch.reduction,
        prev_channels=BASE_CHANNELS,
        curr_channels=BASE_CHANNELS,
        cell_channels=REDUCTION_CHANNELS,
        reduction=True,
    )
    cell.eval()

    s0 = torch.randn(
        BATCH_SIZE,
        BASE_CHANNELS,
        INPUT_SIZE,
        INPUT_SIZE,
    )
    s1 = torch.randn_like(s0)

    with torch.no_grad():
        y = cell(s0, s1)

    expected_channels = _expected_output_channels(
        arch.reduction,
        REDUCTION_CHANNELS,
    )

    assert y.shape == (
        BATCH_SIZE,
        expected_channels,
        INPUT_SIZE // 2,
        INPUT_SIZE // 2,
    )
    assert cell.output_multiplier == len(get_unused_states(arch.reduction))
    assert cell.output_channels == expected_channels
    _assert_finite(y)


@pytest.mark.parametrize(
    ("unused_state", "replacement_state"),
    ((0, 1), (1, 0)),
)
def test_reduction_cell_aligns_unused_original_state(
    unused_state: int,
    replacement_state: int,
) -> None:
    """An unused original state is reduced before Reduction Cell concat."""

    rng = random.Random(SEED)
    gene = _replace_input_state(
        random_architecture(rng).reduction,
        old_state=unused_state,
        new_state=replacement_state,
    )
    unused_indices = get_unused_states(gene)

    assert unused_state in unused_indices

    cell = NASNetCell(
        gene=gene,
        prev_channels=BASE_CHANNELS,
        curr_channels=BASE_CHANNELS,
        cell_channels=REDUCTION_CHANNELS,
        reduction=True,
    )
    cell.eval()

    s0 = torch.randn(
        BATCH_SIZE,
        BASE_CHANNELS,
        INPUT_SIZE,
        INPUT_SIZE,
    )
    s1 = torch.randn_like(s0)

    with torch.no_grad():
        y = cell(s0, s1)

    expected_channels = len(unused_indices) * REDUCTION_CHANNELS

    assert y.shape == (
        BATCH_SIZE,
        expected_channels,
        INPUT_SIZE // 2,
        INPUT_SIZE // 2,
    )
    assert cell.output_multiplier == len(unused_indices)
    assert cell.output_channels == expected_channels

    alignment_index = unused_indices.index(unused_state)
    assert isinstance(
        cell.output_alignments[alignment_index],
        FactorizedReduction,
    )
    _assert_finite(y)


def test_normal_cell_after_reduction_preprocesses_both_inputs() -> None:
    """Model s0=32x32 and s1=16x16 entering a post-reduction Normal Cell."""

    rng = random.Random(SEED)
    arch = random_architecture(rng)

    cell = NASNetCell(
        gene=arch.normal,
        prev_channels=48,
        curr_channels=96,
        cell_channels=48,
        reduction=False,
        prev_reduction=True,
    )
    cell.eval()

    s0 = torch.randn(BATCH_SIZE, 48, 32, 32)
    s1 = torch.randn(BATCH_SIZE, 96, 16, 16)

    with torch.no_grad():
        y = cell(s0, s1)

    expected_channels = _expected_output_channels(arch.normal, 48)

    assert y.shape == (BATCH_SIZE, expected_channels, 16, 16)
    assert cell.output_channels == expected_channels
    _assert_finite(y)


def test_cell_rejects_a_gene_without_exactly_five_pairs() -> None:
    """The generic NASNet search space is frozen to exactly five pairs."""

    rng = random.Random(SEED)
    arch = random_architecture(rng)

    # Rebuild the same genotype type with one pair removed. This works for
    # the frozen dataclass-style CellGene without coupling the test to its
    # concrete class name or import path.
    invalid_gene = type(arch.normal)(pairs=arch.normal.pairs[:-1])

    with pytest.raises(ValueError, match="exactly 5 pairs"):
        NASNetCell(
            gene=invalid_gene,
            prev_channels=BASE_CHANNELS,
            curr_channels=BASE_CHANNELS,
            cell_channels=BASE_CHANNELS,
            reduction=False,
        )


def test_one_hundred_random_normal_and_reduction_cells() -> None:
    """Run 100 deterministic architecture pairs (200 total cell forwards)."""

    rng = random.Random(SEED)
    torch.manual_seed(SEED)

    # Batch size 2 matches the recommended stress-check script while keeping
    # this structural test lighter than a training workload.
    s0 = torch.randn(2, BASE_CHANNELS, INPUT_SIZE, INPUT_SIZE)
    s1 = torch.randn_like(s0)

    for case_index in range(STRESS_ARCHITECTURES):
        arch = random_architecture(rng)

        normal = NASNetCell(
            gene=arch.normal,
            prev_channels=BASE_CHANNELS,
            curr_channels=BASE_CHANNELS,
            cell_channels=BASE_CHANNELS,
            reduction=False,
        )
        normal.eval()

        reduction = NASNetCell(
            gene=arch.reduction,
            prev_channels=BASE_CHANNELS,
            curr_channels=BASE_CHANNELS,
            cell_channels=REDUCTION_CHANNELS,
            reduction=True,
        )
        reduction.eval()

        with torch.no_grad():
            normal_output = normal(s0, s1)
            reduction_output = reduction(s0, s1)

        expected_normal_channels = _expected_output_channels(
            arch.normal,
            BASE_CHANNELS,
        )
        expected_reduction_channels = _expected_output_channels(
            arch.reduction,
            REDUCTION_CHANNELS,
        )

        assert normal_output.shape == (
            2,
            expected_normal_channels,
            INPUT_SIZE,
            INPUT_SIZE,
        ), f"normal cell failed for random case {case_index}"

        assert reduction_output.shape == (
            2,
            expected_reduction_channels,
            INPUT_SIZE // 2,
            INPUT_SIZE // 2,
        ), f"reduction cell failed for random case {case_index}"

        assert normal.output_channels == expected_normal_channels
        assert reduction.output_channels == expected_reduction_channels
        _assert_finite(normal_output)
        _assert_finite(reduction_output)
