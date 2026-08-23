from dataclasses import replace
import random

from .architecture import Architecture
from .space import SEARCH_SPACE, validate_architecture


def count_changed_genes(a: Architecture, b: Architecture) -> int:
    da, db = a.to_dict(), b.to_dict()
    return sum(da[k] != db[k] for k in da)


def mutate(parent: Architecture, rng: random.Random) -> Architecture:
    gene = rng.choice(list(SEARCH_SPACE.keys()))
    current = getattr(parent, gene)
    candidates = [x for x in SEARCH_SPACE[gene] if x != current]
    new_value = rng.choice(candidates)
    child = replace(parent, **{gene: new_value})

    assert child != parent
    assert count_changed_genes(parent, child) == 1
    assert validate_architecture(child)
    return child
