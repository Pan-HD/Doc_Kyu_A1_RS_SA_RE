import random
import torch

from src.nasnet.genotype import (
    random_architecture,
)

from src.nasnet.cell import (
    NASNetCell,
)


rng = random.Random(
    20260825
)

arch = random_architecture(
    rng
)

s0 = torch.randn(
    4, 24, 32, 32
)

s1 = torch.randn(
    4, 24, 32, 32
)


normal = NASNetCell(
    gene=arch.normal,
    prev_channels=24,
    curr_channels=24,
    cell_channels=24,
    reduction=False,
)

normal_out = normal(
    s0,
    s1
)

print(
    "Normal:",
    normal_out.shape
)


reduction = NASNetCell(
    gene=arch.reduction,
    prev_channels=24,
    curr_channels=24,
    cell_channels=48,
    reduction=True,
)

reduction_out = reduction(
    s0,
    s1
)

print(
    "Reduction:",
    reduction_out.shape
)