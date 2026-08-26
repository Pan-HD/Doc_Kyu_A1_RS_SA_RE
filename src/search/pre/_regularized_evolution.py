"""Official regularized-evolution control flow from Real et al.

The implementation in this module is intentionally independent of PyTorch.
Architectures, mutations, and evaluations are supplied as callables, which
makes the algorithm easy to test before it is connected to NASNet training.

Important fidelity properties:

* the live population is a FIFO :class:`collections.deque`;
* tournament members are sampled uniformly *with replacement*;
* exactly one mutation is performed for every post-initialization evaluation;
* duplicate architectures, including identity mutations, are evaluated again;
* a child is appended on the right and the oldest member is removed on the
  left; and
* the evaluation budget counts both random initialization and children.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Callable, Generic, Protocol, Sequence, TypeVar


ArchitectureT = TypeVar("ArchitectureT")
RANDOM_INITIALIZATION = "random_initialization"


class MutationResultLike(Protocol[ArchitectureT]):
    """Structural interface expected from a mutation function.

    ``src.nasnet.mutation.MutationResult`` already satisfies this protocol.
    """

    architecture: ArchitectureT
    mutation_type: str


@dataclass(frozen=True)
class EvaluatedIndividual(Generic[ArchitectureT]):
    """One independently evaluated architecture in the search history."""

    architecture: ArchitectureT
    fitness: float
    evaluation_id: int
    mutation_type: str
    parent_evaluation_id: int | None = None


@dataclass(frozen=True)
class EvolutionResult(Generic[ArchitectureT]):
    """Final live population together with every evaluation event."""

    population: deque[EvaluatedIndividual[ArchitectureT]]
    history: tuple[EvaluatedIndividual[ArchitectureT], ...]

    @property
    def evaluation_count(self) -> int:
        """Number of evaluations consumed from the search budget."""

        return len(self.history)

    @property
    def best_individual(self) -> EvaluatedIndividual[ArchitectureT]:
        """Highest-fitness individual observed during the entire search."""

        if not self.history:
            raise RuntimeError("evolution result has an empty history")
        return max(self.history, key=lambda individual: individual.fitness)


def sample_tournament(
    population: Sequence[EvaluatedIndividual[ArchitectureT]]
    | deque[EvaluatedIndividual[ArchitectureT]],
    sample_size: int,
    rng: random.Random,
) -> tuple[EvaluatedIndividual[ArchitectureT], ...]:
    """Uniformly sample a tournament with replacement.

    ``random.choice`` is called once per tournament slot. Consequently the
    same population member may appear multiple times and ``sample_size`` may
    be larger than the population itself.
    """

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    members = list(population)
    if not members:
        raise ValueError("cannot sample from an empty population")

    return tuple(
        rng.choice(members)
        for _ in range(sample_size)
    )


def tournament_select(
    population: Sequence[EvaluatedIndividual[ArchitectureT]]
    | deque[EvaluatedIndividual[ArchitectureT]],
    sample_size: int,
    rng: random.Random,
) -> EvaluatedIndividual[ArchitectureT]:
    """Return the highest-fitness member of a with-replacement tournament."""

    sample = sample_tournament(
        population=population,
        sample_size=sample_size,
        rng=rng,
    )
    return max(sample, key=lambda individual: individual.fitness)


def _evaluate_fitness(
    architecture: ArchitectureT,
    evaluate_fn: Callable[[ArchitectureT], float],
) -> float:
    """Run one real evaluation event and validate its scalar fitness."""

    # Deliberately no architecture cache or duplicate check is used here.
    # Equal genotypes are separate training/evaluation events in official RE.
    raw_fitness = evaluate_fn(architecture)
    try:
        fitness = float(raw_fitness)
    except (TypeError, ValueError) as error:
        raise TypeError("evaluate_fn must return a scalar fitness") from error

    if not math.isfinite(fitness):
        raise ValueError("evaluate_fn returned a non-finite fitness")
    return fitness


def regularized_evolution(
    *,
    random_architecture_fn: Callable[[random.Random], ArchitectureT],
    mutate_fn: Callable[
        [ArchitectureT, random.Random],
        MutationResultLike[ArchitectureT],
    ],
    evaluate_fn: Callable[[ArchitectureT], float],
    population_size: int,
    tournament_size: int,
    budget: int,
    rng: random.Random,
) -> EvolutionResult[ArchitectureT]:
    """Run aging/regularized evolution for an exact evaluation budget.

    Parameters
    ----------
    random_architecture_fn:
        Called once per initial individual as ``fn(rng)``.
    mutate_fn:
        Called exactly once per evolution cycle as ``fn(parent, rng)``. Its
        return value must provide ``architecture`` and ``mutation_type``.
    evaluate_fn:
        Trains/evaluates one architecture and returns validation fitness.
        It is called even when the genotype has appeared before.
    population_size:
        Number of live individuals in the FIFO queue.
    tournament_size:
        Number of samples drawn with replacement for parent selection.
    budget:
        Exact total number of evaluations, including initialization.
    rng:
        Explicit random generator used by initialization, selection, and
        mutation so a run can be reproduced from its seed.
    """

    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if tournament_size <= 0:
        raise ValueError("tournament_size must be positive")
    if budget < population_size:
        raise ValueError("budget must be at least population_size")

    population: deque[EvaluatedIndividual[ArchitectureT]] = deque()
    history: list[EvaluatedIndividual[ArchitectureT]] = []

    # Random initialization consumes the first population_size evaluations.
    for _ in range(population_size):
        architecture = random_architecture_fn(rng)
        individual = EvaluatedIndividual(
            architecture=architecture,
            fitness=_evaluate_fitness(architecture, evaluate_fn),
            evaluation_id=len(history),
            mutation_type=RANDOM_INITIALIZATION,
            parent_evaluation_id=None,
        )
        population.append(individual)
        history.append(individual)

    while len(history) < budget:
        parent = tournament_select(
            population=population,
            sample_size=tournament_size,
            rng=rng,
        )

        # One and only one mutation event occurs in each cycle. Identity
        # mutation is valid and must continue through evaluation unchanged.
        mutation_result = mutate_fn(parent.architecture, rng)
        try:
            child_architecture = mutation_result.architecture
            mutation_type = mutation_result.mutation_type
        except AttributeError as error:
            raise TypeError(
                "mutate_fn must return an object with architecture and "
                "mutation_type attributes"
            ) from error

        if not isinstance(mutation_type, str) or not mutation_type:
            raise ValueError("mutation_type must be a non-empty string")

        child = EvaluatedIndividual(
            architecture=child_architecture,
            fitness=_evaluate_fitness(child_architecture, evaluate_fn),
            evaluation_id=len(history),
            mutation_type=mutation_type,
            parent_evaluation_id=parent.evaluation_id,
        )

        # These operations deliberately mirror Algorithm 1's FIFO aging.
        population.append(child)
        history.append(child)
        population.popleft()

    if len(population) != population_size:
        raise RuntimeError("population size changed during evolution")
    if len(history) != budget:
        raise RuntimeError("evaluation budget accounting is inconsistent")

    return EvolutionResult(
        population=deque(population),
        history=tuple(history),
    )


__all__ = [
    "EvaluatedIndividual",
    "EvolutionResult",
    "MutationResultLike",
    "RANDOM_INITIALIZATION",
    "regularized_evolution",
    "sample_tournament",
    "tournament_select",
]
