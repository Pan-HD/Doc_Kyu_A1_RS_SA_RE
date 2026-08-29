"""RS-SA-RE state transitions for real-training budget accounting.

Only calls routed through ``_run_real_training`` consume the real-training
budget. Candidate generation, surrogate training, and surrogate prediction are
deliberately budget-neutral. A repeat evaluation updates its paired-label
record but never appends to the population, changes FIFO order, creates a birth,
or replaces the first-seed population fitness.

This module isolates the invariants required by RS-SA-RE so it can be composed
with the project's existing Regularized Evolution implementation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any, Callable, Generic, Iterable, Literal, Sequence, TypeVar

from ..surrogate.multitask_dataset import (
    PairedEvaluationRecord,
    PairedEvaluationStore,
)


ArchitectureT = TypeVar("ArchitectureT")
SurrogateT = TypeVar("SurrogateT")
PredictionT = TypeVar("PredictionT")
EventType = Literal["first_evaluation", "repeat_evaluation"]


class BudgetExhausted(RuntimeError):
    """Raised before a real CNN training would exceed the hard budget."""


@dataclass(frozen=True, slots=True)
class RealTrainingEvent(Generic[ArchitectureT]):
    budget_index: int
    event_type: EventType
    base_evaluation_index: int
    architecture: ArchitectureT
    training_seed: int
    accuracy: float


class RealTrainingBudget(Generic[ArchitectureT]):
    """Hard counter containing only completed real CNN training runs."""

    def __init__(self, limit: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("budget limit must be an integer")
        if limit <= 0:
            raise ValueError("budget limit must be positive")
        self.limit = limit
        self._events: list[RealTrainingEvent[ArchitectureT]] = []

    @property
    def used(self) -> int:
        return len(self._events)

    @property
    def remaining(self) -> int:
        return self.limit - self.used

    @property
    def exhausted(self) -> bool:
        return self.used >= self.limit

    @property
    def events(self) -> tuple[RealTrainingEvent[ArchitectureT], ...]:
        return tuple(self._events)

    def ensure_available(self) -> None:
        if self.exhausted:
            raise BudgetExhausted(
                f"real-training budget exhausted: {self.used}/{self.limit}"
            )

    def record(
        self,
        *,
        event_type: EventType,
        base_evaluation_index: int,
        architecture: ArchitectureT,
        training_seed: int,
        accuracy: float,
    ) -> RealTrainingEvent[ArchitectureT]:
        self.ensure_available()
        event = RealTrainingEvent(
            budget_index=self.used + 1,
            event_type=event_type,
            base_evaluation_index=base_evaluation_index,
            architecture=architecture,
            training_seed=training_seed,
            accuracy=accuracy,
        )
        self._events.append(event)
        return event


@dataclass(frozen=True, slots=True)
class PopulationIndividual(Generic[ArchitectureT]):
    architecture: ArchitectureT
    fitness: float
    birth_order: int
    base_evaluation_index: int
    training_seed: int


class RSSARECore(Generic[ArchitectureT]):
    """Minimal RS-SA-RE engine surface that enforces budget/population rules."""

    def __init__(
        self,
        *,
        population_size: int,
        budget: int,
        evaluator: Callable[[ArchitectureT, int], float],
    ) -> None:
        if isinstance(population_size, bool) or not isinstance(population_size, int):
            raise TypeError("population_size must be an integer")
        if population_size <= 0:
            raise ValueError("population_size must be positive")

        self.population_size = population_size
        self.population: deque[PopulationIndividual[ArchitectureT]] = deque()
        self.records: PairedEvaluationStore[ArchitectureT] = PairedEvaluationStore()
        self.budget: RealTrainingBudget[ArchitectureT] = RealTrainingBudget(budget)
        self._evaluator = evaluator
        self._next_birth_order = 0

    @property
    def births_created(self) -> int:
        return self._next_birth_order

    def population_snapshot(self) -> tuple[PopulationIndividual[ArchitectureT], ...]:
        return tuple(self.population)

    def _run_real_training(
        self,
        *,
        event_type: EventType,
        base_evaluation_index: int,
        architecture: ArchitectureT,
        training_seed: int,
    ) -> float:
        self.budget.ensure_available()
        accuracy = self._evaluator(architecture, training_seed)
        if isinstance(accuracy, bool) or not isinstance(accuracy, Real):
            raise TypeError("evaluator accuracy must be a real number")
        accuracy = float(accuracy)
        if not isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
            raise ValueError("evaluator accuracy must be finite and in [0, 1]")

        self.budget.record(
            event_type=event_type,
            base_evaluation_index=base_evaluation_index,
            architecture=architecture,
            training_seed=training_seed,
            accuracy=accuracy,
        )
        return accuracy

    def run_first_evaluation(
        self,
        *,
        base_evaluation_index: int,
        architecture: ArchitectureT,
        training_seed: int,
    ) -> PopulationIndividual[ArchitectureT]:
        """Train a new individual once, add it to the population, and apply FIFO."""

        if base_evaluation_index in self.records:
            raise ValueError(
                f"duplicate base_evaluation_index: {base_evaluation_index}"
            )

        accuracy = self._run_real_training(
            event_type="first_evaluation",
            base_evaluation_index=base_evaluation_index,
            architecture=architecture,
            training_seed=training_seed,
        )
        record = PairedEvaluationRecord(
            base_evaluation_index=base_evaluation_index,
            architecture=architecture,
            seed_1=training_seed,
            accuracy_1=accuracy,
        )
        self.records.add(record)

        individual = PopulationIndividual(
            architecture=architecture,
            fitness=accuracy,
            birth_order=self._next_birth_order,
            base_evaluation_index=base_evaluation_index,
            training_seed=training_seed,
        )
        self._next_birth_order += 1

        if len(self.population) >= self.population_size:
            self.population.popleft()
        self.population.append(individual)
        return individual

    def run_repeat_evaluation(
        self,
        *,
        base_evaluation_index: int,
        training_seed: int,
    ) -> PairedEvaluationRecord[ArchitectureT]:
        """Train one second seed without touching population state or fitness."""

        record = self.records.get(base_evaluation_index)
        if record.has_pair:
            raise ValueError("this evaluation record already has a repeat")
        if training_seed == record.seed_1:
            raise ValueError("repeat training seed must differ from first seed")

        accuracy = self._run_real_training(
            event_type="repeat_evaluation",
            base_evaluation_index=base_evaluation_index,
            architecture=record.architecture,
            training_seed=training_seed,
        )
        record.add_repeat(seed_2=training_seed, accuracy_2=accuracy)
        return record

    @staticmethod
    def generate_candidates(
        *,
        parent: ArchitectureT,
        candidate_count: int,
        mutate: Callable[[ArchitectureT], ArchitectureT],
    ) -> tuple[ArchitectureT, ...]:
        """Generate mutations without consuming real-training budget."""

        if candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        return tuple(mutate(parent) for _ in range(candidate_count))

    @staticmethod
    def train_surrogate(
        trainer: Callable[..., SurrogateT],
        *args: Any,
        **kwargs: Any,
    ) -> SurrogateT:
        """Train a surrogate without consuming real-training budget."""

        return trainer(*args, **kwargs)

    @staticmethod
    def predict_with_surrogate(
        predictor: Callable[[Sequence[ArchitectureT]], PredictionT],
        candidates: Sequence[ArchitectureT],
    ) -> PredictionT:
        """Run surrogate inference without consuming real-training budget."""

        return predictor(candidates)

