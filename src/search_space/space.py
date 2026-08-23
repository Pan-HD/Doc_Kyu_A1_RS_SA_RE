from itertools import product
import random

from .architecture import Architecture


SEARCH_SPACE = {
    "num_conv_blocks": [2, 3, 4],
    "initial_channels": [16, 24, 32],
    "channel_multiplier": [1, 2],
    "kernel_size": [3, 5],
    "dropout": [0.0, 0.25, 0.5],
    "use_batchnorm": [False, True],
    "activation": ["relu", "gelu"],
    "pooling": ["max", "avg"],
}


def validate_architecture(arch: Architecture) -> bool:
    d = arch.to_dict()
    return all(d[k] in SEARCH_SPACE[k] for k in SEARCH_SPACE)


def enumerate_architectures():
    keys = list(SEARCH_SPACE)
    for values in product(*(SEARCH_SPACE[k] for k in keys)):
        yield Architecture(**dict(zip(keys, values)))


def sample_random_architecture(rng: random.Random) -> Architecture:
    return Architecture(
        **{k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
    )
