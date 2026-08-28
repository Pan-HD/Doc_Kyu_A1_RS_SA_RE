"""Domain-independent surrogate-assisted regularized evolution.

Parent selection, true fitness, FIFO aging, and the real-evaluation budget
match the official RE baseline. The only algorithmic change is offspring
screening: K independent baseline mutations are ranked by a surrogate and
exactly one selected child is evaluated for real.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Generic, Mapping, Protocol, Sequence, TypeVar

from src.surrogate.dataset import SurrogateDataset

from .regularized_evolution import (
    EVOLUTION_PHASE,
    INITIALIZATION_PHASE,
    RANDOM_INITIALIZATION,
    EvaluatedIndividual,
    EvaluationOutcome,
    tournament_select,
)


ArchitectureT = TypeVar("ArchitectureT")
EncodingT = TypeVar("EncodingT")


class MutationResultLike(Protocol[ArchitectureT]):
    architecture: ArchitectureT
    mutation_type: str


@dataclass(frozen=True)
class SurrogateScoreResult:
    scores: tuple[float, ...]
    training_mse: float

    def __post_init__(self) -> None:
        normalized_scores = tuple(float(value) for value in self.scores)
        if not normalized_scores:
            raise ValueError("surrogate scores must not be empty")
        if any(not math.isfinite(value) for value in normalized_scores):
            raise ValueError("surrogate scores must be finite")
        training_mse = float(self.training_mse)
        if not math.isfinite(training_mse) or training_mse < 0.0:
            raise ValueError("surrogate training_mse must be finite and non-negative")
        object.__setattr__(self, "scores", normalized_scores)
        object.__setattr__(self, "training_mse", training_mse)


@dataclass(frozen=True)
class CandidatePrediction(Generic[ArchitectureT]):
    candidate_index: int
    architecture: ArchitectureT
    mutation_type: str
    predicted_mu: float
    selected: bool

    def __post_init__(self) -> None:
        if self.candidate_index < 0:
            raise ValueError("candidate_index must be non-negative")
        if not isinstance(self.mutation_type, str) or not self.mutation_type:
            raise ValueError("mutation_type must be a non-empty string")
        predicted_mu = float(self.predicted_mu)
        if not math.isfinite(predicted_mu):
            raise ValueError("predicted_mu must be finite")
        object.__setattr__(self, "predicted_mu", predicted_mu)
        object.__setattr__(self, "selected", bool(self.selected))


@dataclass(frozen=True)
class CandidateBatch(Generic[ArchitectureT]):
    evaluation_index: int
    parent_evaluation_id: int
    surrogate_training_size: int
    surrogate_training_mse: float
    candidates: tuple[CandidatePrediction[ArchitectureT], ...]

    def __post_init__(self) -> None:
        if self.evaluation_index <= 0:
            raise ValueError("evaluation_index must be positive")
        if self.parent_evaluation_id < 0:
            raise ValueError("parent_evaluation_id must be non-negative")
        if self.surrogate_training_size <= 0:
            raise ValueError("surrogate_training_size must be positive")
        if (
            not math.isfinite(float(self.surrogate_training_mse))
            or self.surrogate_training_mse < 0.0
        ):
            raise ValueError(
                "surrogate_training_mse must be finite and non-negative"
            )
        if not self.candidates:
            raise ValueError("candidate batch must not be empty")
        indices = tuple(candidate.candidate_index for candidate in self.candidates)
        if indices != tuple(range(len(self.candidates))):
            raise ValueError("candidate indices must be contiguous from zero")
        if sum(candidate.selected for candidate in self.candidates) != 1:
            raise ValueError("candidate batch must contain one selected candidate")

    @property
    def selected_candidate_index(self) -> int:
        selected = [
            candidate.candidate_index
            for candidate in self.candidates
            if candidate.selected
        ]
        if len(selected) != 1:
            raise RuntimeError("candidate batch must contain one selected candidate")
        return selected[0]


@dataclass(frozen=True)
class SAEvolutionProgress(Generic[ArchitectureT]):
    phase: str
    individual: EvaluatedIndividual[ArchitectureT]
    population: tuple[EvaluatedIndividual[ArchitectureT], ...]
    history_length: int
    candidate_batch: CandidateBatch[ArchitectureT] | None


@dataclass(frozen=True)
class SAEvolutionResult(Generic[ArchitectureT]):
    population: deque[EvaluatedIndividual[ArchitectureT]]
    history: tuple[EvaluatedIndividual[ArchitectureT], ...]
    candidate_batches: tuple[CandidateBatch[ArchitectureT], ...]
    surrogate_dataset: SurrogateDataset

    @property
    def evaluation_count(self) -> int:
        return len(self.history)

    @property
    def best_individual(self) -> EvaluatedIndividual[ArchitectureT]:
        if not self.history:
            raise RuntimeError("SA-RE result has an empty history")
        return max(self.history, key=lambda individual: individual.fitness)


def _normalize_evaluation(
    architecture: ArchitectureT,
    evaluate_fn: Callable[[ArchitectureT], float | EvaluationOutcome],
) -> EvaluationOutcome:
    # Duplicate and identity architectures deliberately reach evaluate_fn.
    raw_outcome = evaluate_fn(architecture)
    if isinstance(raw_outcome, EvaluationOutcome):
        raw_fitness = raw_outcome.fitness
        metadata: Mapping[str, Any] = raw_outcome.metadata
    else:
        raw_fitness = raw_outcome
        metadata = {}
    try:
        fitness = float(raw_fitness)
    except (TypeError, ValueError) as error:
        raise TypeError("evaluate_fn must return scalar fitness") from error
    if not math.isfinite(fitness):
        raise ValueError("evaluate_fn returned non-finite fitness")
    return EvaluationOutcome(fitness=fitness, metadata=dict(metadata))


def surrogate_assisted_evolution(
    *,
    random_architecture_fn: Callable[[random.Random], ArchitectureT],
    mutate_fn: Callable[
        [ArchitectureT, random.Random],
        MutationResultLike[ArchitectureT],
    ],
    evaluate_fn: Callable[[ArchitectureT], float | EvaluationOutcome],
    encode_fn: Callable[[ArchitectureT], EncodingT],
    surrogate_score_fn: Callable[
        [SurrogateDataset, Sequence[EncodingT]],
        SurrogateScoreResult,
    ],
    surrogate_dataset: SurrogateDataset,
    population_size: int,
    tournament_size: int,
    budget: int,
    candidate_count: int,
    rng: random.Random,
    progress_fn: Callable[[SAEvolutionProgress[ArchitectureT]], None]
    | None = None,
) -> SAEvolutionResult[ArchitectureT]:
    """Run SA-RE for an exact number of completed real evaluations."""

    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if tournament_size <= 0:
        raise ValueError("tournament_size must be positive")
    if budget < population_size:
        raise ValueError("budget must be at least population_size")
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if len(surrogate_dataset) != 0:
        raise ValueError("surrogate_dataset must be empty at search start")

    population: deque[EvaluatedIndividual[ArchitectureT]] = deque()
    history: list[EvaluatedIndividual[ArchitectureT]] = []
    candidate_batches: list[CandidateBatch[ArchitectureT]] = []

    def emit_progress(
        phase: str,
        individual: EvaluatedIndividual[ArchitectureT],
        candidate_batch: CandidateBatch[ArchitectureT] | None,
    ) -> None:
        if progress_fn is None:
            return
        progress_fn(
            SAEvolutionProgress(
                phase=phase,
                individual=individual,
                population=tuple(population),
                history_length=len(history),
                candidate_batch=candidate_batch,
            )
        )

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
        surrogate_dataset.add(
            architecture=architecture,
            encoding=encode_fn(architecture),
            target_accuracy=outcome.fitness,
            evaluation_index=len(history),
        )
        if len(surrogate_dataset) != len(history):
            raise RuntimeError("surrogate dataset/history mismatch")
        emit_progress(INITIALIZATION_PHASE, individual, None)

    while len(history) < budget:
        parent = tournament_select(
            population=population,
            sample_size=tournament_size,
            rng=rng,
        )

        mutation_results = []
        candidate_encodings = []
        for _ in range(candidate_count):
            mutation_result = mutate_fn(parent.architecture, rng)
            try:
                candidate_architecture = mutation_result.architecture
                mutation_type = mutation_result.mutation_type
            except AttributeError as error:
                raise TypeError(
                    "mutate_fn must return architecture and mutation_type"
                ) from error
            if not isinstance(mutation_type, str) or not mutation_type:
                raise ValueError("mutation_type must be a non-empty string")
            mutation_results.append(mutation_result)
            candidate_encodings.append(encode_fn(candidate_architecture))

        surrogate_training_size = len(surrogate_dataset)
        # Surrogate fitting/inference is isolated from the search RNG. The
        # scorer does not receive ``rng``; this state check also catches an
        # accidental closure over the same random.Random instance.
        search_rng_state_before_scoring = rng.getstate()
        score_result = surrogate_score_fn(
            surrogate_dataset,
            tuple(candidate_encodings),
        )
        if rng.getstate() != search_rng_state_before_scoring:
            raise RuntimeError("surrogate scoring consumed the search RNG")
        if len(score_result.scores) != candidate_count:
            raise ValueError("surrogate returned the wrong number of scores")

        # Python max returns the first index on ties: deterministic tie-break.
        selected_index = max(
            range(candidate_count),
            key=lambda index: score_result.scores[index],
        )
        selected_mutation = mutation_results[selected_index]
        selected_architecture = selected_mutation.architecture
        outcome = _normalize_evaluation(selected_architecture, evaluate_fn)

        child_metadata = dict(outcome.metadata)
        child_metadata.update(
            {
                "predicted_mu_before_training": score_result.scores[
                    selected_index
                ],
                "surrogate_training_size": surrogate_training_size,
                "selected_candidate_index": selected_index,
                "surrogate_training_mse": score_result.training_mse,
            }
        )
        child = EvaluatedIndividual(
            architecture=selected_architecture,
            fitness=outcome.fitness,
            evaluation_id=len(history),
            mutation_type=selected_mutation.mutation_type,
            parent_evaluation_id=parent.evaluation_id,
            metadata=child_metadata,
        )

        candidate_batch = CandidateBatch(
            evaluation_index=child.evaluation_id + 1,
            parent_evaluation_id=parent.evaluation_id,
            surrogate_training_size=surrogate_training_size,
            surrogate_training_mse=score_result.training_mse,
            candidates=tuple(
                CandidatePrediction(
                    candidate_index=index,
                    architecture=mutation_result.architecture,
                    mutation_type=mutation_result.mutation_type,
                    predicted_mu=score_result.scores[index],
                    selected=index == selected_index,
                )
                for index, mutation_result in enumerate(mutation_results)
            ),
        )

        population.append(child)
        history.append(child)
        population.popleft()
        candidate_batches.append(candidate_batch)
        surrogate_dataset.add(
            architecture=selected_architecture,
            encoding=candidate_encodings[selected_index],
            target_accuracy=outcome.fitness,
            evaluation_index=len(history),
        )

        if len(population) != population_size:
            raise RuntimeError("population size changed during SA-RE")
        if len(history) > budget:
            raise RuntimeError("real-evaluation budget was exceeded")
        if len(surrogate_dataset) != len(history):
            raise RuntimeError("surrogate dataset/history mismatch")
        emit_progress(EVOLUTION_PHASE, child, candidate_batch)

    expected_batches = budget - population_size
    if len(population) != population_size:
        raise RuntimeError("final population size is inconsistent")
    if len(history) != budget or len(surrogate_dataset) != budget:
        raise RuntimeError("final real-evaluation accounting is inconsistent")
    if len(candidate_batches) != expected_batches:
        raise RuntimeError("candidate-batch accounting is inconsistent")

    return SAEvolutionResult(
        population=deque(population),
        history=tuple(history),
        candidate_batches=tuple(candidate_batches),
        surrogate_dataset=surrogate_dataset,
    )


__all__ = [
    "CandidateBatch",
    "CandidatePrediction",
    "SAEvolutionProgress",
    "SAEvolutionResult",
    "SurrogateScoreResult",
    "surrogate_assisted_evolution",
]
