import random

from src.nasnet.genotype import (
    random_architecture,
    validate_architecture,
)


def test_10000_random_architectures_are_valid():
    rng = random.Random(20260824)

    for i in range(10000):
        arch = random_architecture(rng)

        assert validate_architecture(arch), (
            f"Invalid architecture generated at iteration {i}: {arch}"
        )


def test_generate_1000_architectures():
    rng = random.Random(20260824)

    architectures = [
        random_architecture(rng)
        for _ in range(1000)
    ]

    assert len(architectures) == 1000