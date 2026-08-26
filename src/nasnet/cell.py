"""Generic PyTorch NASNet cell implementation.

This module implements the tensor-level cell mechanics used by the NASNet
baseline:

* preprocessing for the two original cell inputs;
* five pairwise combinations with elementwise addition;
* reduction-cell stride handling for original inputs only; and
* alignment and concatenation of every unused hidden state along the NCHW
  channel axis.

The eight searchable branch operations are intentionally implemented in
``operations.py`` and are constructed here through ``build_operation``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .genotype import CellGene, get_unused_states
from .operations import build_operation, make_bn


class FactorizedReduction(nn.Module):
    """Reduce spatial resolution by two while projecting channel count.

    Two offset paths sample complementary spatial grids. Each path produces
    part of ``out_channels``; their outputs are concatenated and normalized.

    Expected NASNet usage has even spatial dimensions (for example,
    32 x 32 -> 16 x 16). The small crop in ``forward`` also keeps both paths
    compatible if an odd spatial size is encountered.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if out_channels < 2:
            raise ValueError(
                "FactorizedReduction requires out_channels >= 2 "
                "so both offset paths are non-empty"
            )

        c1 = out_channels // 2
        c2 = out_channels - c1

        self.relu = nn.ReLU(inplace=False)
        self.path1_conv = nn.Conv2d(
            in_channels,
            c1,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.path2_conv = nn.Conv2d(
            in_channels,
            c2,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn = make_bn(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(x)

        # Path 1 samples pixels (0, 0), (0, 2), (2, 0), ...
        path1 = self.path1_conv(x[:, :, ::2, ::2])

        # Path 2 first shifts one pixel in both spatial dimensions and then
        # samples every second pixel.
        path2 = self.path2_conv(x[:, :, 1::2, 1::2])

        # NASNet's intended input sizes are even. Cropping makes the module
        # well-defined for an accidental odd input without interpolation.
        if path1.shape[2:] != path2.shape[2:]:
            height = min(path1.shape[2], path2.shape[2])
            width = min(path1.shape[3], path2.shape[3])
            path1 = path1[:, :, :height, :width]
            path2 = path2[:, :, :height, :width]

        out = torch.cat((path1, path2), dim=1)
        return self.bn(out)


class Projection(nn.Module):
    """Apply ReLU -> 1x1 convolution -> BatchNorm without resizing."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")

        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            make_bn(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class NASNetCell(nn.Module):
    """Build and execute one normal or reduction NASNet cell.

    Args:
        gene: Frozen five-pair cell genotype.
        prev_channels: Channel count of ``s0`` (previous-previous output).
        curr_channels: Channel count of ``s1`` (previous output).
        cell_channels: Channel count produced by every branch.
        reduction: Whether this cell reduces spatial resolution by two.
        prev_reduction: Whether ``s0`` has a larger spatial resolution than
            ``s1`` and therefore needs factorized reduction before entering
            this cell.
    """

    NUM_PAIRS = 5

    def __init__(
        self,
        gene: CellGene,
        prev_channels: int,
        curr_channels: int,
        cell_channels: int,
        reduction: bool,
        prev_reduction: bool = False,
    ) -> None:
        super().__init__()

        if prev_channels <= 0 or curr_channels <= 0 or cell_channels <= 0:
            raise ValueError("all channel counts must be positive")
        if len(gene.pairs) != self.NUM_PAIRS:
            raise ValueError(
                f"NASNetCell requires exactly {self.NUM_PAIRS} pairs, "
                f"got {len(gene.pairs)}"
            )

        self.gene = gene
        self.cell_channels = cell_channels
        self.reduction = reduction

        # s0: reduce spatial size when it comes from before a reduction cell;
        # otherwise project only when its channel count differs.
        if prev_reduction:
            self.preprocess_prev: nn.Module = FactorizedReduction(
                prev_channels,
                cell_channels,
            )
        elif prev_channels != cell_channels:
            self.preprocess_prev = Projection(
                prev_channels,
                cell_channels,
            )
        else:
            self.preprocess_prev = nn.Identity()

        # s1 always follows the reference ReLU -> 1x1 Conv -> BN path,
        # including when curr_channels already equals cell_channels.
        self.preprocess_curr = Projection(
            curr_channels,
            cell_channels,
        )

        # Each of the five pairs owns two independent branch operations.
        # In a reduction cell, stride=2 applies only to branches reading one
        # of the two original inputs (state 0 or state 1). Generated states
        # have already been reduced and therefore always use stride=1.
        self.branch_ops = nn.ModuleList()
        for pair in gene.pairs:
            for branch in (pair.branch_1, pair.branch_2):
                input_state = branch.input_state
                stride = 2 if reduction and input_state < 2 else 1
                self.branch_ops.append(
                    build_operation(
                        branch.op,
                        channels=cell_channels,
                        stride=stride,
                    )
                )

        # NASNet concatenates every hidden state that is not consumed by a
        # later pair. This includes original states 0 and 1 when they remain
        # unused; it is not restricted to generated states 2..6.
        self.unused_indices = tuple(get_unused_states(gene))
        if not self.unused_indices:
            raise ValueError("a NASNet cell must have at least one unused state")

        # State 6 (the output of the fifth pair) is always unused because no
        # later pair can consume it. It therefore defines the target spatial
        # resolution and channel count for cell-output concatenation.
        #
        # All states in this implementation already have ``cell_channels``.
        # In a reduction cell, however, an unused original input (state 0 or
        # state 1) is still at the pre-reduction resolution and must be reduced
        # before concatenation. Register these trainable alignment modules in
        # __init__ so their parameters are visible to state_dict/optimizers.
        self.output_alignments = nn.ModuleList()
        for state_index in self.unused_indices:
            needs_spatial_reduction = reduction and state_index < 2
            if needs_spatial_reduction:
                alignment: nn.Module = FactorizedReduction(
                    cell_channels,
                    cell_channels,
                )
            else:
                alignment = nn.Identity()
            self.output_alignments.append(alignment)

        self.output_multiplier = len(self.unused_indices)
        self.output_channels = self.output_multiplier * cell_channels

    def forward(self, s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        """Execute preprocessing, five pair additions, and aligned concat."""

        s0 = self.preprocess_prev(s0)
        s1 = self.preprocess_curr(s1)

        if s0.shape != s1.shape:
            raise RuntimeError(
                "NASNet cell inputs differ after preprocessing: "
                f"s0={tuple(s0.shape)}, s1={tuple(s1.shape)}"
            )

        states = [s0, s1]

        for pair_index, pair in enumerate(self.gene.pairs):
            branch_1 = pair.branch_1
            branch_2 = pair.branch_2

            max_available_state = len(states) - 1
            if not 0 <= branch_1.input_state <= max_available_state:
                raise IndexError(
                    f"pair {pair_index} branch_1 references unavailable "
                    f"state {branch_1.input_state}; available range is "
                    f"0..{max_available_state}"
                )
            if not 0 <= branch_2.input_state <= max_available_state:
                raise IndexError(
                    f"pair {pair_index} branch_2 references unavailable "
                    f"state {branch_2.input_state}; available range is "
                    f"0..{max_available_state}"
                )

            h1 = states[branch_1.input_state]
            h2 = states[branch_2.input_state]

            h1 = self.branch_ops[2 * pair_index](h1)
            h2 = self.branch_ops[2 * pair_index + 1](h2)

            if h1.shape != h2.shape:
                raise RuntimeError(
                    f"pair {pair_index} branch outputs have different shapes: "
                    f"branch_1={tuple(h1.shape)}, branch_2={tuple(h2.shape)}"
                )

            states.append(h1 + h2)

        assert len(states) == 2 + self.NUM_PAIRS

        max_state_index = len(states) - 1
        for index in self.unused_indices:
            if not 0 <= index <= max_state_index:
                raise IndexError(
                    f"unused state index {index} is outside 0..{max_state_index}"
                )

        target = states[-1]
        outputs = []
        for state_index, alignment in zip(
            self.unused_indices,
            self.output_alignments,
        ):
            aligned = alignment(states[state_index])

            if aligned.shape[0] != target.shape[0]:
                raise RuntimeError(
                    f"unused state {state_index} has batch size "
                    f"{aligned.shape[0]}, expected {target.shape[0]}"
                )
            if aligned.shape[1] != target.shape[1]:
                raise RuntimeError(
                    f"unused state {state_index} has {aligned.shape[1]} "
                    f"channels after alignment, expected {target.shape[1]}"
                )
            if aligned.shape[2:] != target.shape[2:]:
                raise RuntimeError(
                    f"unused state {state_index} has spatial shape "
                    f"{tuple(aligned.shape[2:])} after alignment, expected "
                    f"{tuple(target.shape[2:])}"
                )

            outputs.append(aligned)

        out = torch.cat(outputs, dim=1)

        if out.shape[1] != self.output_channels:
            raise RuntimeError(
                f"cell produced {out.shape[1]} output channels, expected "
                f"{self.output_channels}"
            )
        return out


__all__ = [
    "FactorizedReduction",
    "Projection",
    "NASNetCell",
]
