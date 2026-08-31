"""Formal B=60 dry run against the production RS-SA-RE scheduler.

This test injects a CPU-only fake evaluator into the real
``repeat_stability_assisted_evolution`` entry point. It performs no CNN
training, dataset access, checkpoint writing, or experiment logging.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import pytest
import torch

from src.evolution.repeat_policy import RepeatPolicyConfig, derive_training_seed
from src.search.rs_sa_re_evolution import (
    PERIODIC_REPEAT_PHASE,
    WARMUP_REPEAT_PHASE,
    repeat_stability_assisted_evolution,
)
from src.surrogate.multitask_model import MultiTaskPrediction


FORMAL_POPULATION_SIZE = 20
FORMAL_TOURNAMENT_SIZE = 5
FORMAL_CANDIDATE_COUNT = 5
FORMAL_BUDGET = 60
FORMAL_WARMUP_PAIRS = 4
FORMAL_REPEAT_INTERVAL = 4
FORMAL_REPEAT_RATE_BETA = 0.25
FORMAL_LAMBDA = 1.0


@dataclass(frozen=True)
class FakeMutation:
    architecture: int
    mutation_type: str


@dataclass(frozen=True)
class FakeFitResult:
    model: tuple[int, int]
    training_loss: float
    mean_training_mse: float
    instability_training_mse: float
    observation_count: int
    paired_count: int


class FakeEvaluator:
    """Deterministic scalar evaluator with the production callable contract."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def __call__(self, architecture: int, training_seed: int) -> float:
        self.calls.append((architecture, training_seed))
        # Remains in [0, 1] and varies slightly across architecture and seed.
        return (
            0.60
            + (architecture % 100) / 1_000.0
            + (training_seed % 17) / 100_000.0
        )


@dataclass
class FormalRunArtifacts:
    result: Any
    evaluator: FakeEvaluator
    progress: list[Any]
    mutation_calls: list[tuple[int, int]]
    prediction_batch_sizes: list[int]


def _fit_surrogate(records, encodings) -> FakeFitResult:
    assert len(records) == len(encodings)
    paired_count = sum(record.has_pair for record in records)
    return FakeFitResult(
        model=(len(records), paired_count),
        training_loss=0.1,
        mean_training_mse=0.06,
        instability_training_mse=0.04,
        observation_count=len(records),
        paired_count=paired_count,
    )


@pytest.fixture(scope="module")
def formal_run() -> FormalRunArtifacts:
    initialization = iter(range(1, FORMAL_POPULATION_SIZE + 1))
    next_mutation_architecture = iter(range(10_000, 100_000))
    evaluator = FakeEvaluator()
    progress: list[Any] = []
    mutation_calls: list[tuple[int, int]] = []
    prediction_batch_sizes: list[int] = []

    def mutate(parent: int, rng: random.Random) -> FakeMutation:
        del rng
        architecture = next(next_mutation_architecture)
        mutation_calls.append((parent, architecture))
        return FakeMutation(
            architecture=architecture,
            mutation_type="formal_budget_fake_mutation",
        )

    def predict(model, encodings) -> MultiTaskPrediction:
        del model
        count = len(encodings)
        prediction_batch_sizes.append(count)
        # Candidate 0 has the highest mu and the lowest d. The exact ranking is
        # unimportant here; deterministic predictions make the dry run stable.
        predicted_mean = torch.tensor(
            [0.80 - 0.01 * index for index in range(count)],
            dtype=torch.float32,
        )
        predicted_instability = torch.tensor(
            [0.01 + 0.005 * index for index in range(count)],
            dtype=torch.float32,
        )
        return MultiTaskPrediction(predicted_mean, predicted_instability)

    result = repeat_stability_assisted_evolution(
        random_architecture_fn=lambda rng: next(initialization),
        mutate_fn=mutate,
        evaluate_fn=evaluator,
        encode_fn=lambda architecture: (float(architecture),),
        fit_surrogate_fn=_fit_surrogate,
        predict_surrogate_fn=predict,
        population_size=FORMAL_POPULATION_SIZE,
        tournament_size=FORMAL_TOURNAMENT_SIZE,
        budget=FORMAL_BUDGET,
        candidate_count=FORMAL_CANDIDATE_COUNT,
        training_seed_base=20_260_830,
        stability_penalty_lambda=FORMAL_LAMBDA,
        repeat_seed=903_001,
        search_rng=random.Random(3_001),
        repeat_policy_config=RepeatPolicyConfig(
            initial_population_size=FORMAL_POPULATION_SIZE,
            warmup_pairs=FORMAL_WARMUP_PAIRS,
            repeat_interval=FORMAL_REPEAT_INTERVAL,
            repeat_rate_beta=FORMAL_REPEAT_RATE_BETA,
        ),
        progress_fn=progress.append,
    )
    return FormalRunArtifacts(
        result=result,
        evaluator=evaluator,
        progress=progress,
        mutation_calls=mutation_calls,
        prediction_batch_sizes=prediction_batch_sizes,
    )


def test_formal_budget_exact_counts(formal_run: FormalRunArtifacts) -> None:
    result = formal_run.result
    first_events = [
        event for event in result.budget_events
        if event.event_type == "first_evaluation"
    ]
    repeat_events = [
        event for event in result.budget_events
        if event.event_type == "repeat_evaluation"
    ]

    assert result.real_training_runs == FORMAL_BUDGET == 60
    assert len(formal_run.evaluator.calls) == 60
    assert len(first_events) == 49
    assert len(repeat_events) == 11
    assert len(result.history) == 49
    assert len(result.repeat_evaluations) == 11
    assert len(result.population) == 20


def test_formal_budget_candidate_and_selection_counts(
    formal_run: FormalRunArtifacts,
) -> None:
    result = formal_run.result

    assert len(result.candidate_batches) == 29
    assert len(formal_run.mutation_calls) == 145
    assert formal_run.prediction_batch_sizes == [5] * 29
    assert sum(len(batch.candidates) for batch in result.candidate_batches) == 145
    assert sum(
        candidate.selected
        for batch in result.candidate_batches
        for candidate in batch.candidates
    ) == 29
    assert all(len(batch.candidates) == 5 for batch in result.candidate_batches)
    assert all(
        sum(candidate.selected for candidate in batch.candidates) == 1
        for batch in result.candidate_batches
    )


def test_formal_budget_event_sequence_has_no_b61(
    formal_run: FormalRunArtifacts,
) -> None:
    result = formal_run.result
    budget_indices = [event.budget_index for event in result.budget_events]
    repeat_budgets = [
        event.budget_index
        for event in result.budget_events
        if event.event_type == "repeat_evaluation"
    ]

    assert budget_indices == list(range(1, 61))
    assert max(budget_indices) == 60
    assert repeat_budgets[:4] == [21, 22, 23, 24]
    assert repeat_budgets[4:] == [29, 34, 39, 44, 49, 54, 59]
    assert result.budget_events[-1].budget_index == 60
    assert result.budget_events[-1].event_type == "first_evaluation"


def test_repeats_do_not_modify_population_births_or_candidates(
    formal_run: FormalRunArtifacts,
) -> None:
    repeat_progress = [
        item for item in formal_run.progress
        if item.event_type == "repeat_evaluation"
    ]
    assert len(repeat_progress) == 11

    for index, item in enumerate(formal_run.progress):
        if item.event_type != "repeat_evaluation":
            continue
        assert index > 0
        previous = formal_run.progress[index - 1]
        assert item.population == previous.population
        assert item.history_length == previous.history_length
        assert item.individual is None
        assert item.candidate_batch is None
        assert item.repeat_evaluation is not None


def test_repeat_phases_and_paired_labels_match_frozen_policy(
    formal_run: FormalRunArtifacts,
) -> None:
    result = formal_run.result
    warmup = [item for item in formal_run.progress if item.phase == WARMUP_REPEAT_PHASE]
    periodic = [item for item in formal_run.progress if item.phase == PERIODIC_REPEAT_PHASE]

    assert len(warmup) == 4
    assert len(periodic) == 7
    assert result.surrogate_fit.observation_count == 49
    assert result.surrogate_fit.paired_count == 11
    assert sum(record.has_pair for record in result.paired_records) == 11

    records_by_index = {
        record.base_evaluation_index: record
        for record in result.paired_records
    }
    for individual in result.history:
        record = records_by_index[individual.evaluation_id + 1]
        # Population/search fitness remains the first-seed accuracy even when a
        # repeat happens to obtain a different accuracy.
        assert individual.fitness == pytest.approx(record.accuracy_1)


def test_training_seeds_follow_first_and_repeat_replica_contract(
    formal_run: FormalRunArtifacts,
) -> None:
    training_seed_base = 20_260_830
    for event in formal_run.result.budget_events:
        replicate_id = 1 if event.event_type == "first_evaluation" else 2
        assert event.training_seed == derive_training_seed(
            training_seed_base=training_seed_base,
            base_evaluation_index=event.base_evaluation_index,
            replicate_id=replicate_id,
        )

