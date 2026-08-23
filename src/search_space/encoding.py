import numpy as np

from .architecture import Architecture
from .space import SEARCH_SPACE


def encode_architecture(arch: Architecture) -> np.ndarray:
    values = arch.to_dict()
    encoded = []

    for gene, legal_values in SEARCH_SPACE.items():
        value = values[gene]
        encoded.extend([1.0 if value == candidate else 0.0 for candidate in legal_values])

    return np.asarray(encoded, dtype=np.float32)


def encoding_dimension() -> int:
    return sum(len(v) for v in SEARCH_SPACE.values())
