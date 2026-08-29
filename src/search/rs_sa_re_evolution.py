"""Domain-independent repeat-stability-aware regularized evolution.

The engine composes the project's existing Regularized Evolution primitives:
population individuals, tournament selection, mutation, and FIFO aging.  It
adds only paired evaluation records, the independent repeat scheduler, a
multi-task surrogate controller, and ``mu_hat - lambda * d_hat`` screening.

``history`` contains first evaluations only.  Scheduled repeats are separate
real-training events and never become population individuals.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Generic, Mapping, Protocol, Sequence, TypeVar

from src.evolution.candidate_scoring import (
    CandidateScoringConfig,
    score_candidates,
)
from src.evolution.repeat_policy import (
    RepeatPolicyConfig,
    RepeatScheduler,
    derive_training_seed,
)
from src.evolution.rs_sa_re import RealTrainingBudget, RealTrainingEvent
from src.surrogate.multitask_dataset import (
    PairedEvaluationRecord,
    PairedEvaluationStore,
)
from src.surrogate.multitask_model import MultiTaskPrediction

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
FitResultT = TypeVar("FitResultT", bound="SurrogateFitResultLike")

WARMUP_REPEAT_PHASE = "warmup_repeat"
PERIODIC_REPEAT_PHASE = "periodic_repeat"


class MutationResultLike(Protocol[ArchitectureT]):
    architecture: ArchitectureT
    mutation_type: str


class SurrogateFitResultLike(Protocol):
    model: Any
    training_loss: float
    mean_training_mse: float
    instability_training_mse: float
    observation_count: int
    paired_count: int


@dataclass(frozen=True)
class RSCandidatePrediction(Generic[ArchitectureT]):
    candidate_index: int
    architecture: ArchitectureT
    mutation_type: str
    predicted_mu: float
    predicted_d: float
    stability_penalty_lambda: float
    score: float
    selected: bool

    def __post_init__(self) -> None:
        if self.candidate_index < 0:
            raise ValueError("candidate_index must be non-negative")
        if not isinstance(self.mutation_type, str) or not self.mutation_type:
            raise ValueError("mutation_type must be a non-empty string")
        for name in (
            "predicted_mu",
            "predicted_d",
            "stability_penalty_lambda",
            "score",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.predicted_d < 0.0:
            raise ValueError("predicted_d must be non-negative")
        if self.stability_penalty_lambda < 0.0:
            raise ValueError("stability_penalty_lambda must be non-negative")
        expected = self.predicted_mu - self.stability_penalty_lambda * self.predicted_d
        # Predictions and scores originate from float32 tensors.  Validate the
        # formula at float32 precision instead of rejecting normal rounding
        # from two separately converted tensor values.
        if not math.isclose(self.score, expected, rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError("candidate score does not equal mu_hat - lambda*d_hat")
        object.__setattr__(self, "selected", bool(self.selected))


@dataclass(frozen=True)
class RSCandidateBatch(Generic[ArchitectureT]):
    evaluation_index: int
    parent_evaluation_id: int
    surrogate_training_size: int
    paired_label_count: int
    surrogate_training_loss: float
    mean_training_mse: float
    instability_training_mse: float
    candidates: tuple[RSCandidatePrediction[ArchitectureT], ...]

    def __post_init__(self) -> None:
        if self.evaluation_index <= 0:
            raise ValueError("evaluation_index must be positive")
        if self.parent_evaluation_id < 0:
            raise ValueError("parent_evaluation_id must be non-negative")
        if self.surrogate_training_size <= 0:
            raise ValueError("surrogate_training_size must be positive")
        if not 0 <= self.paired_label_count <= self.surrogate_training_size:
            raise ValueError("paired_label_count is inconsistent")
        for name in (
            "surrogate_training_loss",
            "mean_training_mse",
            "instability_training_mse",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not self.candidates:
            raise ValueError("candidate batch must not be empty")
        indices = tuple(item.candidate_index for item in self.candidates)
        if indices != tuple(range(len(self.candidates))):
            raise ValueError("candidate indices must be contiguous from zero")
        if sum(item.selected for item in self.candidates) != 1:
            raise ValueError("candidate batch must select exactly one candidate")

    @property
    def selected_candidate_index(self) -> int:
        return next(
            item.candidate_index for item in self.candidates if item.selected
        )


@dataclass(frozen=True)
class RSRepeatEvaluation(Generic[ArchitectureT]):
    event: RealTrainingEvent[ArchitectureT]
    record: PairedEvaluationRecord
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class RSSAREvolutionProgress(Generic[ArchitectureT]):
    phase: str
    event_type: str
    budget_event: RealTrainingEvent[ArchitectureT]
    population: tuple[EvaluatedIndividual[ArchitectureT], ...]
    history_length: int
    real_training_runs: int
    individual: EvaluatedIndividual[ArchitectureT] | None
    candidate_batch: RSCandidateBatch[ArchitectureT] | None
    repeat_evaluation: RSRepeatEvaluation[ArchitectureT] | None


@dataclass(frozen=True)
class RSSAREvolutionResult(Generic[ArchitectureT, FitResultT]):
    population: deque[EvaluatedIndividual[ArchitectureT]]
    history: tuple[EvaluatedIndividual[ArchitectureT], ...]
    candidate_batches: tuple[RSCandidateBatch[ArchitectureT], ...]
    repeat_evaluations: tuple[RSRepeatEvaluation[ArchitectureT], ...]
    paired_records: tuple[PairedEvaluationRecord, ...]
    budget_events: tuple[RealTrainingEvent[ArchitectureT], ...]
    surrogate_fit: FitResultT

    @property
    def evaluation_count(self) -> int:
        """Number of first evaluations (population births), excluding repeats."""

        return len(self.history)

    @property
    def real_training_runs(self) -> int:
        return len(self.budget_events)

    @property
    def best_individual(self) -> EvaluatedIndividual[ArchitectureT]:
        if not self.history:
            raise RuntimeError("RS-SA-RE result has an empty history")
        return max(self.history, key=lambda item: item.fitness)


def _normalize_evaluation(
    *,
    architecture: ArchitectureT,
    training_seed: int,
    evaluate_fn: Callable[
        [ArchitectureT, int],
        float | EvaluationOutcome,
    ],
) -> EvaluationOutcome:
    raw_outcome = evaluate_fn(architecture, training_seed)
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
    if not math.isfinite(fitness) or not 0.0 <= fitness <= 1.0:
        raise ValueError("evaluate_fn fitness must be finite and in [0, 1]")

    normalized_metadata = dict(metadata)
    recorded_seed = normalized_metadata.get("training_seed")
    if recorded_seed is not None and int(recorded_seed) != training_seed:
        raise RuntimeError("evaluator metadata contains the wrong training seed")
    normalized_metadata["training_seed"] = training_seed
    return EvaluationOutcome(fitness=fitness, metadata=normalized_metadata)


def _validate_fit_result(
    fit_result: SurrogateFitResultLike,
    records: Sequence[PairedEvaluationRecord],
) -> None:
    expected_observations = len(records)
    expected_pairs = sum(record.has_pair for record in records)
    if int(fit_result.observation_count) != expected_observations:
        raise RuntimeError("surrogate observation count is inconsistent")
    if int(fit_result.paired_count) != expected_pairs:
        raise RuntimeError("surrogate paired-label count is inconsistent")
    for name in (
        "training_loss",
        "mean_training_mse",
        "instability_training_mse",
    ):
        value = float(getattr(fit_result, name))
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError(f"surrogate {name} is invalid")


def repeat_stability_assisted_evolution(
    *,
    random_architecture_fn: Callable[[random.Random], ArchitectureT],
    mutate_fn: Callable[
        [ArchitectureT, random.Random],
        MutationResultLike[ArchitectureT],
    ],
    evaluate_fn: Callable[
        [ArchitectureT, int],
        float | EvaluationOutcome,
    ],
    encode_fn: Callable[[ArchitectureT], EncodingT],
    fit_surrogate_fn: Callable[
        [Sequence[PairedEvaluationRecord], Sequence[EncodingT]],
        FitResultT,
    ],
    predict_surrogate_fn: Callable[
        [Any, Sequence[EncodingT]],
        MultiTaskPrediction,
    ],
    population_size: int,
    tournament_size: int,
    budget: int,
    candidate_count: int,
    training_seed_base: int,
    stability_penalty_lambda: float,
    repeat_seed: int,
    search_rng: random.Random,
    repeat_policy_config: RepeatPolicyConfig | None = None,
    progress_fn: Callable[[RSSAREvolutionProgress[ArchitectureT]], None]
    | None = None,
) -> RSSAREvolutionResult[ArchitectureT, FitResultT]:
    """Run RS-SA-RE under a hard real-CNN-training budget."""

    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if tournament_size <= 0:
        raise ValueError("tournament_size must be positive")
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if not isinstance(search_rng, random.Random):
        raise TypeError("search_rng must be random.Random")

    policy = repeat_policy_config or RepeatPolicyConfig(
        initial_population_size=population_size
    )
    if policy.initial_population_size != population_size:
        raise ValueError("repeat policy population size does not match search")
    minimum_budget = population_size + policy.warmup_pairs
    if budget < minimum_budget:
        raise ValueError(
            "budget must cover initial population and all warm-up repeats"
        )

    scoring_config = CandidateScoringConfig(
        stability_penalty_lambda=stability_penalty_lambda
    )
    scheduler = RepeatScheduler(repeat_seed=repeat_seed, config=policy)
    budget_counter: RealTrainingBudget[ArchitectureT] = RealTrainingBudget(budget)
    records = PairedEvaluationStore()
    population: deque[EvaluatedIndividual[ArchitectureT]] = deque()
    history: list[EvaluatedIndividual[ArchitectureT]] = []
    candidate_batches: list[RSCandidateBatch[ArchitectureT]] = []
    repeat_evaluations: list[RSRepeatEvaluation[ArchitectureT]] = []

    def emit(
        *,
        phase: str,
        event: RealTrainingEvent[ArchitectureT],
        individual: EvaluatedIndividual[ArchitectureT] | None = None,
        candidate_batch: RSCandidateBatch[ArchitectureT] | None = None,
        repeat_evaluation: RSRepeatEvaluation[ArchitectureT] | None = None,
    ) -> None:
        if progress_fn is None:
            return
        progress_fn(
            RSSAREvolutionProgress(
                phase=phase,
                event_type=event.event_type,
                budget_event=event,
                population=tuple(population),
                history_length=len(history),
                real_training_runs=budget_counter.used,
                individual=individual,
                candidate_batch=candidate_batch,
                repeat_evaluation=repeat_evaluation,
            )
        )

    def run_first(
        *,
        architecture: ArchitectureT,
        mutation_type: str,
        parent_evaluation_id: int | None,
        metadata_updates: Mapping[str, Any] | None = None,
    ) -> tuple[EvaluatedIndividual[ArchitectureT], RealTrainingEvent[ArchitectureT]]:
        base_index = len(history) + 1
        training_seed = derive_training_seed(
            training_seed_base=training_seed_base,
            base_evaluation_index=base_index,
            replicate_id=1,
        )
        budget_counter.ensure_available()
        outcome = _normalize_evaluation(
            architecture=architecture,
            training_seed=training_seed,
            evaluate_fn=evaluate_fn,
        )
        event = budget_counter.record(
            event_type="first_evaluation",
            base_evaluation_index=base_index,
            architecture=architecture,
            training_seed=training_seed,
            accuracy=outcome.fitness,
        )
        records.add(
            PairedEvaluationRecord(
                base_evaluation_index=base_index,
                architecture=architecture,
                seed_1=training_seed,
                accuracy_1=outcome.fitness,
            )
        )
        metadata = dict(outcome.metadata)
        metadata["base_evaluation_index"] = base_index
        if metadata_updates is not None:
            metadata.update(metadata_updates)
        individual = EvaluatedIndividual(
            architecture=architecture,
            fitness=outcome.fitness,
            evaluation_id=base_index - 1,
            mutation_type=mutation_type,
            parent_evaluation_id=parent_evaluation_id,
            metadata=metadata,
        )
        population.append(individual)
        history.append(individual)
        if len(population) > population_size:
            population.popleft()
        return individual, event

    def run_repeat(
        *,
        record: PairedEvaluationRecord,
        phase: str,
    ) -> RSRepeatEvaluation[ArchitectureT]:
        population_before = tuple(population)
        history_length_before = len(history)
        seed_2 = derive_training_seed(
            training_seed_base=training_seed_base,
            base_evaluation_index=record.base_evaluation_index,
            replicate_id=2,
        )
        budget_counter.ensure_available()
        outcome = _normalize_evaluation(
            architecture=record.architecture,
            training_seed=seed_2,
            evaluate_fn=evaluate_fn,
        )
        event = budget_counter.record(
            event_type="repeat_evaluation",
            base_evaluation_index=record.base_evaluation_index,
            architecture=record.architecture,
            training_seed=seed_2,
            accuracy=outcome.fitness,
        )
        record.add_repeat(seed_2=seed_2, accuracy_2=outcome.fitness)
        repeat_result = RSRepeatEvaluation(
            event=event,
            record=record,
            metadata=dict(outcome.metadata),
        )
        repeat_evaluations.append(repeat_result)
        if tuple(population) != population_before or len(history) != history_length_before:
            raise RuntimeError("repeat evaluation modified population or births")
        emit(
            phase=phase,
            event=event,
            repeat_evaluation=repeat_result,
        )
        return repeat_result

    def refit() -> FitResultT:
        materialized_records = tuple(records)
        encodings = tuple(encode_fn(record.architecture) for record in materialized_records)
        search_state = search_rng.getstate()
        fit_result = fit_surrogate_fn(materialized_records, encodings)
        if search_rng.getstate() != search_state:
            raise RuntimeError("surrogate training consumed the search RNG")
        _validate_fit_result(fit_result, materialized_records)
        return fit_result

    for _ in range(population_size):
        architecture = random_architecture_fn(search_rng)
        individual, event = run_first(
            architecture=architecture,
            mutation_type=RANDOM_INITIALIZATION,
            parent_evaluation_id=None,
        )
        emit(phase=INITIALIZATION_PHASE, event=event, individual=individual)

    search_state = search_rng.getstate()
    warmup_records = scheduler.select_warmup(tuple(records))
    if search_rng.getstate() != search_state:
        raise RuntimeError("repeat warm-up selection consumed the search RNG")
    for record in warmup_records:
        run_repeat(record=record, phase=WARMUP_REPEAT_PHASE)

    current_fit = refit()
    first_evaluations_after_warmup = 0

    while not budget_counter.exhausted:
        parent = tournament_select(
            population=population,
            sample_size=tournament_size,
            rng=search_rng,
        )

        mutations = []
        candidate_encodings = []
        for _ in range(candidate_count):
            mutation = mutate_fn(parent.architecture, search_rng)
            try:
                architecture = mutation.architecture
                mutation_type = mutation.mutation_type
            except AttributeError as error:
                raise TypeError(
                    "mutate_fn must return architecture and mutation_type"
                ) from error
            if not isinstance(mutation_type, str) or not mutation_type:
                raise ValueError("mutation_type must be a non-empty string")
            mutations.append(mutation)
            candidate_encodings.append(encode_fn(architecture))

        search_state = search_rng.getstate()
        prediction = predict_surrogate_fn(
            current_fit.model,
            tuple(candidate_encodings),
        )
        if search_rng.getstate() != search_state:
            raise RuntimeError("surrogate prediction consumed the search RNG")
        if not isinstance(prediction, MultiTaskPrediction):
            raise TypeError("predict_surrogate_fn must return MultiTaskPrediction")
        if prediction.predicted_mean.numel() != candidate_count:
            raise ValueError("surrogate returned the wrong number of predictions")
        scored = score_candidates(prediction, scoring_config)
        selected_index = scored.selected_index
        selected_mutation = mutations[selected_index]

        candidate_batch = RSCandidateBatch(
            evaluation_index=len(history) + 1,
            parent_evaluation_id=parent.evaluation_id,
            surrogate_training_size=int(current_fit.observation_count),
            paired_label_count=int(current_fit.paired_count),
            surrogate_training_loss=float(current_fit.training_loss),
            mean_training_mse=float(current_fit.mean_training_mse),
            instability_training_mse=float(
                current_fit.instability_training_mse
            ),
            candidates=tuple(
                RSCandidatePrediction(
                    candidate_index=index,
                    architecture=mutation.architecture,
                    mutation_type=mutation.mutation_type,
                    predicted_mu=float(scored.predicted_mean[index].item()),
                    predicted_d=float(
                        scored.predicted_instability[index].item()
                    ),
                    stability_penalty_lambda=(
                        scored.stability_penalty_lambda
                    ),
                    score=float(scored.scores[index].item()),
                    selected=index == selected_index,
                )
                for index, mutation in enumerate(mutations)
            ),
        )
        selected_prediction = candidate_batch.candidates[selected_index]
        child, event = run_first(
            architecture=selected_mutation.architecture,
            mutation_type=selected_mutation.mutation_type,
            parent_evaluation_id=parent.evaluation_id,
            metadata_updates={
                "predicted_mu_before_training": selected_prediction.predicted_mu,
                "predicted_d_before_training": selected_prediction.predicted_d,
                "stability_penalty_lambda": selected_prediction.stability_penalty_lambda,
                "candidate_score_before_training": selected_prediction.score,
                "selected_candidate_index": selected_index,
                "surrogate_training_size": candidate_batch.surrogate_training_size,
                "paired_label_count": candidate_batch.paired_label_count,
                "surrogate_training_loss": candidate_batch.surrogate_training_loss,
                "mean_training_mse": candidate_batch.mean_training_mse,
                "instability_training_mse": candidate_batch.instability_training_mse,
            },
        )
        candidate_batches.append(candidate_batch)
        if len(population) != population_size:
            raise RuntimeError("population size changed during RS-SA-RE")
        emit(
            phase=EVOLUTION_PHASE,
            event=event,
            individual=child,
            candidate_batch=candidate_batch,
        )
        first_evaluations_after_warmup += 1

        if (
            not budget_counter.exhausted
            and scheduler.should_schedule(
                completed_first_evaluations_after_warmup=(
                    first_evaluations_after_warmup
                )
            )
        ):
            search_state = search_rng.getstate()
            repeat_record = scheduler.select(tuple(records))
            if search_rng.getstate() != search_state:
                raise RuntimeError("repeat selection consumed the search RNG")
            run_repeat(record=repeat_record, phase=PERIODIC_REPEAT_PHASE)

        # Training is budget-neutral. Refit once after the cycle so the next
        # candidate batch sees the new first label and any scheduled pair.
        current_fit = refit()

    if budget_counter.used != budget:
        raise RuntimeError("RS-SA-RE did not consume the exact budget")
    if len(population) != population_size:
        raise RuntimeError("final population size is inconsistent")
    if len(records) != len(history):
        raise RuntimeError("paired-record and first-evaluation counts differ")
    if int(current_fit.observation_count) != len(records):
        raise RuntimeError("final surrogate fit is stale")

    return RSSAREvolutionResult(
        population=deque(population),
        history=tuple(history),
        candidate_batches=tuple(candidate_batches),
        repeat_evaluations=tuple(repeat_evaluations),
        paired_records=tuple(records),
        budget_events=budget_counter.events,
        surrogate_fit=current_fit,
    )


__all__ = [
    "PERIODIC_REPEAT_PHASE",
    "RSCandidateBatch",
    "RSCandidatePrediction",
    "RSRepeatEvaluation",
    "RSSAREvolutionProgress",
    "RSSAREvolutionResult",
    "SurrogateFitResultLike",
    "WARMUP_REPEAT_PHASE",
    "repeat_stability_assisted_evolution",
]
