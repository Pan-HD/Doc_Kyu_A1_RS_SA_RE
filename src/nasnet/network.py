"""Full CIFAR-10 NASNet network assembled from searchable NASNet cells.

The default search model follows the reduced-budget experiment specification:

* three stacks with ``N=3`` Normal Cells per stack;
* two Reduction Cells between stacks;
* branch filters ``F``, ``2F``, and ``4F`` across the three stacks;
* a CIFAR stem with ``stem_multiplier=3``; and
* ReLU, global average pooling, and a linear logits classifier.

Cell output channels are never assumed to equal the branch-filter count.
Instead, construction tracks each ``NASNetCell.output_channels`` value because
the cell output concatenates every unused hidden state.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .cell import NASNetCell
from .genotype import NASNetArchitecture


class NASNetCIFAR(nn.Module):
    """Build the full three-stack NASNet search model for CIFAR images.

    Args:
        architecture: Normal and Reduction Cell genotypes.
        N: Number of Normal Cells in each of the three stacks.
        F: Branch-filter count in the first stack. It doubles after each
            Reduction Cell.
        num_classes: Number of classifier logits.

    The default ``N=3`` configuration contains nine Normal Cells and two
    Reduction Cells. The first cell follows the TensorFlow NASNet behavior for
    ``prev_layer=None`` by reusing the stem output as both ``s0`` and ``s1``.
    """

    NUM_STACKS = 3
    NUM_REDUCTION_CELLS = 2
    STEM_MULTIPLIER = 3

    def __init__(
        self,
        architecture: NASNetArchitecture,
        N: int = 3,
        F: int = 24,
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        if not isinstance(N, int) or isinstance(N, bool) or N <= 0:
            raise ValueError("N must be a positive integer")
        if not isinstance(F, int) or isinstance(F, bool) or F <= 0:
            raise ValueError("F must be a positive integer")
        if (
            not isinstance(num_classes, int)
            or isinstance(num_classes, bool)
            or num_classes <= 0
        ):
            raise ValueError("num_classes must be a positive integer")

        self.architecture = architecture
        self.N = N
        self.F = F
        self.num_classes = num_classes

        self.stem_channels = self.STEM_MULTIPLIER * F
        self.stack_channels = tuple(
            F * (2**stack_index)
            for stack_index in range(self.NUM_STACKS)
        )

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                self.stem_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                self.stem_channels,
                eps=1e-5,
                momentum=0.1,
            ),
        )

        self.cells = nn.ModuleList()
        cell_kinds: list[str] = []

        # Before the first cell, TensorFlow's prev_layer is None. Forward will
        # use the stem output for both inputs, so both channel counts start at
        # stem_channels here as well.
        s0_channels = self.stem_channels
        s1_channels = self.stem_channels
        previous_cell_was_reduction = False

        for stack_index, stack_channels in enumerate(self.stack_channels):
            for _ in range(N):
                normal_cell = NASNetCell(
                    gene=architecture.normal,
                    prev_channels=s0_channels,
                    curr_channels=s1_channels,
                    cell_channels=stack_channels,
                    reduction=False,
                    prev_reduction=previous_cell_was_reduction,
                )
                self.cells.append(normal_cell)
                cell_kinds.append("normal")

                # Runtime update is s0, s1 = s1, normal_cell(s0, s1).
                # Mirror that update exactly for the construction-time channel
                # metadata instead of assuming that output channels equal F.
                old_s1_channels = s1_channels
                s0_channels = old_s1_channels
                s1_channels = normal_cell.output_channels
                previous_cell_was_reduction = False

            if stack_index < self.NUM_STACKS - 1:
                reduction_channels = self.stack_channels[stack_index + 1]
                reduction_cell = NASNetCell(
                    gene=architecture.reduction,
                    prev_channels=s0_channels,
                    curr_channels=s1_channels,
                    cell_channels=reduction_channels,
                    reduction=True,
                    prev_reduction=previous_cell_was_reduction,
                )
                self.cells.append(reduction_cell)
                cell_kinds.append("reduction")

                old_s1_channels = s1_channels
                s0_channels = old_s1_channels
                s1_channels = reduction_cell.output_channels
                previous_cell_was_reduction = True

        self.cell_kinds = tuple(cell_kinds)
        self.normal_cell_count = self.cell_kinds.count("normal")
        self.reduction_cell_count = self.cell_kinds.count("reduction")
        self.total_cell_count = len(self.cell_kinds)

        # Convenient aliases for tests and experiment metadata.
        self.num_normal_cells = self.normal_cell_count
        self.num_reduction_cells = self.reduction_cell_count
        self.num_cells = self.total_cell_count

        expected_normal_cells = self.NUM_STACKS * N
        if self.normal_cell_count != expected_normal_cells:
            raise RuntimeError(
                f"constructed {self.normal_cell_count} Normal Cells, "
                f"expected {expected_normal_cells}"
            )
        if self.reduction_cell_count != self.NUM_REDUCTION_CELLS:
            raise RuntimeError(
                f"constructed {self.reduction_cell_count} Reduction Cells, "
                f"expected {self.NUM_REDUCTION_CELLS}"
            )

        self.final_feature_channels = s1_channels
        self.classifier = nn.Linear(
            self.final_feature_channels,
            num_classes,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the activated feature map immediately before global pooling."""

        if x.ndim != 4:
            raise ValueError(
                "NASNetCIFAR expects a 4D NCHW tensor; "
                f"got shape {tuple(x.shape)}"
            )
        if x.shape[1] != 3:
            raise ValueError(
                "NASNetCIFAR expects three input channels; "
                f"got {x.shape[1]}"
            )

        s1 = self.stem(x)
        s0: torch.Tensor | None = None

        for cell in self.cells:
            # TensorFlow NASNet uses the current layer when prev_layer is None.
            if s0 is None:
                s0 = s1

            y = cell(s0, s1)
            s0, s1 = s1, y

        return functional.relu(s1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return unnormalized class logits with shape ``[batch, classes]``."""

        x = self.forward_features(x)
        x = functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_nasnet(
    architecture: NASNetArchitecture,
    N: int = 3,
    F: int = 24,
    num_classes: int = 10,
) -> NASNetCIFAR:
    """Build the shared NASNet model used by training and search algorithms."""

    return NASNetCIFAR(
        architecture=architecture,
        N=N,
        F=F,
        num_classes=num_classes,
    )


__all__ = [
    "NASNetCIFAR",
    "build_nasnet",
]
