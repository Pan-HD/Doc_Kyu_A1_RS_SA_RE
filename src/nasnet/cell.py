"""Generic PyTorch NASNet cell implementation.

This module implements the tensor-level cell mechanics frozen in
``Tasks_0825.md``:

* preprocessing for the two original cell inputs;
* five pairwise combinations with elementwise addition;
* reduction-cell stride handling for original inputs only; and
* concatenation of unused generated states along the NCHW channel axis.

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

    Two offset paths sample complementary spatial grids.  Each path produces
    part of ``out_channels``; their outputs are concatenated and normalized.

    Expected NASNet usage has even spatial dimensions (for example,
    32 x 32 -> 16 x 16).  The small crop in ``forward`` also keeps both paths
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

        # NASNet's intended input sizes are even.  Cropping makes the module
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
        # of the two original inputs (state 0 or state 1).  Generated states
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

        self.unused_indices = tuple(get_unused_states(gene))
        if not self.unused_indices:
            raise ValueError("a NASNet cell must have at least one unused state")

        self.output_multiplier = len(self.unused_indices)
        self.output_channels = self.output_multiplier * cell_channels

    def forward(self, s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        """Execute preprocessing, five pair additions, and output concat."""

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

        outputs = [states[index] for index in self.unused_indices]
        output_shapes = {tuple(tensor.shape[2:]) for tensor in outputs}
        if len(output_shapes) != 1:
            raise RuntimeError(
                "unused states cannot be concatenated because their spatial "
                f"shapes differ: {sorted(output_shapes)}"
            )

        out = torch.cat(outputs, dim=1)

        assert out.shape[1] == self.output_channels
        return out


__all__ = [
    "FactorizedReduction",
    "Projection",
    "NASNetCell",
]
