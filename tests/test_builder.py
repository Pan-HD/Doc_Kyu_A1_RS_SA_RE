import torch

from src.models.cnn_builder import build_cnn, count_parameters
from src.search_space.architecture import Architecture


ARCHITECTURES = [
    Architecture(2, 16, 1, 3, 0.0, False, "relu", "max"),
    Architecture(3, 24, 2, 3, 0.25, True, "gelu", "avg"),
    Architecture(4, 32, 2, 5, 0.50, True, "gelu", "max"),
]


def test_builder_output_shape():
    x = torch.randn(4, 3, 32, 32)

    for arch in ARCHITECTURES:
        model = build_cnn(arch)
        y = model(x)
        assert y.shape == (4, 10)
        assert count_parameters(model) > 0
