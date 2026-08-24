from src.search_space.architecture import Architecture


def make_architecture():
    return Architecture(
        num_conv_blocks=3,
        initial_channels=24,
        channel_multiplier=2,
        kernel_size=3,
        dropout=0.25,
        use_batchnorm=True,
        activation="relu",
        pooling="max",
    )


def test_architecture_equality():
    arch1 = make_architecture()
    arch2 = make_architecture()

    assert arch1 == arch2


def test_architecture_hashable():
    arch = make_architecture()

    architectures = {arch}

    assert arch in architectures


def test_architecture_to_dict():
    arch = make_architecture()

    d = arch.to_dict()

    assert d["num_conv_blocks"] == 3
    assert d["initial_channels"] == 24
    assert d["activation"] == "relu"

import pytest
from dataclasses import FrozenInstanceError

def test_architecture_is_frozen():
    arch = make_architecture()

    with pytest.raises(FrozenInstanceError):
        arch.kernel_size = 5