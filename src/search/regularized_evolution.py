"""Official regularized-evolution control flow from Real et al.

The algorithm is domain-independent: architecture creation, mutation, and
training/evaluation are injected as callables. This Part F version preserves
all Part E behavior while adding optional evaluation metadata and progress
callbacks for real NASNet experiment logging.

Fidelity properties:

* the live population is a FIFO :class:`collections.deque`;
* tournament members are sampled uniformly *with replacement*;
* exactly one mutation is performed per post-initialization evaluation;
* duplicate architectures, including identity mutations, are evaluated again;
* a child is appended on the right and the oldest member is removed on the
  left; and
* the budget counts initialization and child evaluations exactly once.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Mapping, Protocol, Sequence, TypeVar


ArchitectureT = TypeVar("ArchitectureT")
RANDOM_INITIALIZATION = "random_initialization"
INITIALIZATION_PHASE = "initialization"
EVOLUTION_PHASE = "evolution"


class MutationResultLike(Protocol[ArchitectureT]):
    """Structural interface satisfied by NASNet ``MutationResult``."""

    architecture: ArchitectureT
    mutation_type: str


@dataclass(frozen=True)
class EvaluationOutcome:
    """Scalar search fitness plus JSON-compatible diagnostic metadata."""

    fitness: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, Mapping):
            raise TypeError("evaluation metadata must be a mapping")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class EvaluatedIndividual(Generic[ArchitectureT]):
    """One independently evaluated architecture in search history."""

    architecture: ArchitectureT
    fitness: float
    evaluation_id: int
    mutation_type: str
    parent_evaluation_id: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvolutionProgress(Generic[ArchitectureT]):
    """Immutable state snapshot emitted after each successful evaluation."""

    phase: str
    individual: EvaluatedIndividual[ArchitectureT]
    population: tuple[EvaluatedIndividual[ArchitectureT], ...]
    history_length: int


@dataclass(frozen=True)
class EvolutionResult(Generic[ArchitectureT]):
    """Final FIFO population and every evaluation event."""

    population: deque[EvaluatedIndividual[ArchitectureT]]
    history: tuple[EvaluatedIndividual[ArchitectureT], ...]

    @property
    def evaluation_count(self) -> int:
        return len(self.history)

    @property
    def best_individual(self) -> EvaluatedIndividual[ArchitectureT]:
        if not self.history:
            raise RuntimeError("evolution result has an empty history")
        return max(self.history, key=lambda individual: individual.fitness)


def sample_tournament(
    population: Sequence[EvaluatedIndividual[ArchitectureT]]
    | deque[EvaluatedIndividual[ArchitectureT]],
    sample_size: int,
    rng: random.Random,
) -> tuple[EvaluatedIndividual[ArchitectureT], ...]:
    """Uniformly sample a tournament with replacement."""

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
    """Return the highest-fitness sampled individual."""

    sample = sample_tournament(
        population=population,
        sample_size=sample_size,
        rng=rng,
    )
    return max(sample, key=lambda individual: individual.fitness)


def _normalize_evaluation(
    architecture: ArchitectureT,
    evaluate_fn: Callable[
        [ArchitectureT],
        float | EvaluationOutcome,
    ],
) -> EvaluationOutcome:
    # There is deliberately no cache or duplicate-genotype check here.
    raw_outcome = evaluate_fn(architecture)
    if isinstance(raw_outcome, EvaluationOutcome):
        raw_fitness = raw_outcome.fitness
        metadata = dict(raw_outcome.metadata)
    else:
        raw_fitness = raw_outcome
        metadata = {}

    try:
        fitness = float(raw_fitness)
    except (TypeError, ValueError) as error:
        raise TypeError("evaluate_fn must return a scalar fitness") from error

    if not math.isfinite(fitness):
        raise ValueError("evaluate_fn returned a non-finite fitness")
    return EvaluationOutcome(fitness=fitness, metadata=metadata)


def _emit_progress(
    *,
    phase: str,
    individual: EvaluatedIndividual[ArchitectureT],
    population: deque[EvaluatedIndividual[ArchitectureT]],
    history_length: int,
    progress_fn: Callable[[EvolutionProgress[ArchitectureT]], None] | None,
) -> None:
    if progress_fn is None:
        return
    progress_fn(
        EvolutionProgress(
            phase=phase,
            individual=individual,
            population=tuple(population),
            history_length=history_length,
        )
    )


def regularized_evolution(
    *,
    random_architecture_fn: Callable[[random.Random], ArchitectureT],
    mutate_fn: Callable[
        [ArchitectureT, random.Random],
        MutationResultLike[ArchitectureT],
    ],
    evaluate_fn: Callable[
        [ArchitectureT],
        float | EvaluationOutcome,
    ],
    population_size: int,
    tournament_size: int,
    budget: int,
    rng: random.Random,
    progress_fn: Callable[[EvolutionProgress[ArchitectureT]], None]
    | None = None,
) -> EvolutionResult[ArchitectureT]:
    """Run aging evolution for an exact number of real evaluations.

    ``progress_fn`` is called after every completed evaluation. During random
    initialization the population grows from one member to ``population_size``.
    During evolution every snapshot has exactly ``population_size`` members.
    """

    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if tournament_size <= 0:
        raise ValueError("tournament_size must be positive")
    if budget < population_size:
        raise ValueError("budget must be at least population_size")

    population: deque[EvaluatedIndividual[ArchitectureT]] = deque()
    history: list[EvaluatedIndividual[ArchitectureT]] = []

    for _ in range(population_size):
        architecture = random_architecture_fn(rng)
        outcome = _normalize_evaluation(architecture, evaluate_fn)
        individual = EvaluatedIndividual(
            architecture=architecture,
            fitness=outcome.fitness,
            evaluation_id=len(history),
            mutation_type=RANDOM_INITIALIZATION,
            parent_evaluation_id=None,
            metadata=outcome.metadata,
        )
        population.append(individual)
        history.append(individual)

        if len(population) != len(history):
            raise RuntimeError("initialization queue/history mismatch")
        _emit_progress(
            phase=INITIALIZATION_PHASE,
            individual=individual,
            population=population,
            history_length=len(history),
            progress_fn=progress_fn,
        )

    while len(history) < budget:
        parent = tournament_select(
            population=population,
            sample_size=tournament_size,
            rng=rng,
        )

        # Exactly one mutation call. Identity mutation proceeds unchanged.
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

        outcome = _normalize_evaluation(child_architecture, evaluate_fn)
        child = EvaluatedIndividual(
            architecture=child_architecture,
            fitness=outcome.fitness,
            evaluation_id=len(history),
            mutation_type=mutation_type,
            parent_evaluation_id=parent.evaluation_id,
            metadata=outcome.metadata,
        )

        # FIFO aging order from Algorithm 1.
        population.append(child)
        history.append(child)
        population.popleft()

        if len(population) != population_size:
            raise RuntimeError("population size changed during evolution")
        if len(history) > budget:
            raise RuntimeError("evaluation budget was exceeded")
        _emit_progress(
            phase=EVOLUTION_PHASE,
            individual=child,
            population=population,
            history_length=len(history),
            progress_fn=progress_fn,
        )

    if len(population) != population_size:
        raise RuntimeError("final population size is inconsistent")
    if len(history) != budget:
        raise RuntimeError("evaluation budget accounting is inconsistent")

    return EvolutionResult(
        population=deque(population),
        history=tuple(history),
    )


__all__ = [
    "EVOLUTION_PHASE",
    "INITIALIZATION_PHASE",
    "RANDOM_INITIALIZATION",
    "EvaluatedIndividual",
    "EvaluationOutcome",
    "EvolutionProgress",
    "EvolutionResult",
    "MutationResultLike",
    "regularized_evolution",
    "sample_tournament",
    "tournament_select",
]
