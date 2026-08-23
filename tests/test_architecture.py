import random

from src.search_space.encoding import encode_architecture
from src.search_space.space import (
    enumerate_architectures,
    sample_random_architecture,
    validate_architecture,
)


def test_formal_space_has_1728_unique_architectures():
    arches = list(enumerate_architectures())
    assert len(arches) == 1728
    assert len(set(arches)) == 1728


def test_random_architecture_is_valid():
    rng = random.Random(123)
    for _ in range(100):
        assert validate_architecture(sample_random_architecture(rng))


def test_all_encodings_are_unique_and_fixed_length():
    arches = list(enumerate_architectures())
    encodings = [encode_architecture(a) for a in arches]
    dims = {e.shape for e in encodings}
    assert len(dims) == 1
    assert len({tuple(e.tolist()) for e in encodings}) == 1728
