import random

from src.search_space.mutation import count_changed_genes, mutate
from src.search_space.space import sample_random_architecture, validate_architecture


def test_single_gene_mutation_1000_times():
    rng = random.Random(20260823)
    parent = sample_random_architecture(rng)

    for _ in range(1000):
        child = mutate(parent, rng)
        assert child != parent
        assert count_changed_genes(parent, child) == 1
        assert validate_architecture(child)
        parent = child
