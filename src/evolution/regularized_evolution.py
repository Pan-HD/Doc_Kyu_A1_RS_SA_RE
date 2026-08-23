from dataclasses import dataclass
import random
from typing import Callable, List, Optional

from src.search_space.architecture import Architecture
from src.search_space.mutation import mutate
from src.search_space.space import sample_random_architecture


@dataclass
class Individual:
    architecture: Architecture
    fitness: float
    birth_order: int
    training_seed: int


@dataclass
class SearchRecord:
    training_run_index: int
    training_seed: int
    architecture: Architecture
    parent_architecture: Optional[Architecture]
    validation_accuracy: float
    parameter_count: int
    training_time: float
    birth_order: int


def run_regularized_evolution(
    evaluate_fn: Callable,
    population_size: int,
    tournament_size: int,
    budget: int,
    search_seed: int,
):
    if budget < population_size:
        raise ValueError("budget must be >= population_size")
    if tournament_size > population_size:
        raise ValueError("tournament_size must be <= population_size")

    rng = random.Random(search_seed)
    population: List[Individual] = []
    history: List[SearchRecord] = []
    real_training_runs = 0
    birth_order = 0

    def next_training_seed(run_index: int) -> int:
        # Explicit and reproducible mapping for the baseline.
        return search_seed * 100000 + run_index

    # Initial population
    while len(population) < population_size:
        arch = sample_random_architecture(rng)
        training_seed = next_training_seed(real_training_runs)
        result = evaluate_fn(arch, training_seed)

        individual = Individual(
            architecture=arch,
            fitness=result.best_val_accuracy,
            birth_order=birth_order,
            training_seed=training_seed,
        )
        population.append(individual)

        history.append(
            SearchRecord(
                training_run_index=real_training_runs,
                training_seed=training_seed,
                architecture=arch,
                parent_architecture=None,
                validation_accuracy=result.best_val_accuracy,
                parameter_count=result.parameter_count,
                training_time=result.training_time,
                birth_order=birth_order,
            )
        )

        birth_order += 1
        real_training_runs += 1

    assert len(population) == population_size

    # Evolution loop
    while real_training_runs < budget:
        tournament = rng.sample(population, tournament_size)
        parent = max(tournament, key=lambda x: x.fitness)

        child_arch = mutate(parent.architecture, rng)
        assert child_arch != parent.architecture

        training_seed = next_training_seed(real_training_runs)
        result = evaluate_fn(child_arch, training_seed)

        child = Individual(
            architecture=child_arch,
            fitness=result.best_val_accuracy,
            birth_order=birth_order,
            training_seed=training_seed,
        )

        population.append(child)
        oldest = min(population, key=lambda x: x.birth_order)
        population.remove(oldest)

        history.append(
            SearchRecord(
                training_run_index=real_training_runs,
                training_seed=training_seed,
                architecture=child_arch,
                parent_architecture=parent.architecture,
                validation_accuracy=result.best_val_accuracy,
                parameter_count=result.parameter_count,
                training_time=result.training_time,
                birth_order=birth_order,
            )
        )

        birth_order += 1
        real_training_runs += 1

        assert len(population) == population_size
        assert real_training_runs <= budget

    return population, history
