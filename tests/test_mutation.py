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

def test_mutation_is_reproducible():

    rng1 = random.Random(12345)
    rng2 = random.Random(12345)

    parent1 = sample_random_architecture(rng1)
    parent2 = sample_random_architecture(rng2)

    assert parent1 == parent2

    for _ in range(100):

        child1 = mutate(parent1, rng1)
        child2 = mutate(parent2, rng2)

        assert child1 == child2

        parent1 = child1
        parent2 = child2

def test_different_seeds_produce_different_trajectories():

    rng1 = random.Random(1001)
    rng2 = random.Random(1002)

    parent = sample_random_architecture(
        random.Random(999)
    )

    seq1 = []
    seq2 = []

    p1 = parent
    p2 = parent

    for _ in range(20):

        p1 = mutate(p1, rng1)
        p2 = mutate(p2, rng2)

        seq1.append(p1)
        seq2.append(p2)

    assert seq1 != seq2

def test_parent_is_not_modified():

    rng = random.Random(20260823)

    parent = sample_random_architecture(rng)

    original = parent

    child = mutate(parent, rng)

    assert parent == original
    assert child != parent

def test_all_genes_are_mutated():

    rng = random.Random(20260823)

    parent = sample_random_architecture(rng)

    mutation_count = {
        gene: 0
        for gene in parent.to_dict()
    }

    for _ in range(1000):

        child = mutate(parent, rng)

        before = parent.to_dict()
        after = child.to_dict()

        changed = [
            key
            for key in before
            if before[key] != after[key]
        ]

        assert len(changed) == 1

        mutation_count[
            changed[0]
        ] += 1

        parent = child

    print(mutation_count)

    assert all(
        count > 0
        for count in mutation_count.values()
    )