from collections.abc import Sequence
from dataclasses import dataclass
import random

import pytest
import torch

from src.evolution.repeat_policy import RepeatPolicyConfig
from src.evolution.repeat_policy import derive_training_seed
from src.evolution.rs_sa_re import BudgetExhausted, RSSARECore
from src.search.rs_sa_re_evolution import repeat_stability_assisted_evolution
from src.surrogate.multitask_model import MultiTaskPrediction


class FakeEvaluator:
    def __init__(self, accuracies: dict[tuple[str, int], float]) -> None:
        self.accuracies = accuracies
        self.calls: list[tuple[str, int]] = []

    def __call__(self, architecture: str, training_seed: int) -> float:
        self.calls.append((architecture, training_seed))
        return self.accuracies[(architecture, training_seed)]


def test_first_and_repeat_evaluations_each_consume_one_budget() -> None:
    evaluator = FakeEvaluator({("A", 101): 0.78, ("A", 202): 0.74})
    engine = RSSARECore(
        population_size=2,
        budget=2,
        evaluator=evaluator,
    )

    engine.run_first_evaluation(
        base_evaluation_index=23,
        architecture="A",
        training_seed=101,
    )
    assert engine.budget.used == 1

    engine.run_repeat_evaluation(
        base_evaluation_index=23,
        training_seed=202,
    )
    assert engine.budget.used == 2
    assert [event.event_type for event in engine.budget.events] == [
        "first_evaluation",
        "repeat_evaluation",
    ]


def test_budget_neutral_operations_do_not_consume_budget() -> None:
    evaluator = FakeEvaluator({})
    engine = RSSARECore(
        population_size=2,
        budget=3,
        evaluator=evaluator,
    )
    mutation_number = iter(range(5))

    candidates = engine.generate_candidates(
        parent="A",
        candidate_count=5,
        mutate=lambda parent: f"{parent}-{next(mutation_number)}",
    )
    surrogate = engine.train_surrogate(lambda rows: {"rows": len(rows)}, candidates)
    predictions = engine.predict_with_surrogate(
        lambda items: [0.5] * len(items),
        candidates,
    )

    assert len(candidates) == 5
    assert surrogate == {"rows": 5}
    assert predictions == [0.5] * 5
    assert engine.budget.used == 0
    assert evaluator.calls == []


def test_repeat_preserves_population_fifo_birth_order_and_first_seed_fitness() -> None:
    evaluator = FakeEvaluator(
        {
            ("A", 101): 0.78,
            ("B", 102): 0.77,
            ("A", 201): 0.74,
        }
    )
    engine = RSSARECore(
        population_size=2,
        budget=3,
        evaluator=evaluator,
    )
    individual_a = engine.run_first_evaluation(
        base_evaluation_index=23,
        architecture="A",
        training_seed=101,
    )
    engine.run_first_evaluation(
        base_evaluation_index=24,
        architecture="B",
        training_seed=102,
    )
    before = engine.population_snapshot()
    births_before = engine.births_created

    record = engine.run_repeat_evaluation(
        base_evaluation_index=23,
        training_seed=201,
    )

    assert engine.population_snapshot() == before
    assert engine.births_created == births_before
    assert len(engine.population) == 2
    assert engine.population[0] is individual_a
    assert engine.population[0].fitness == pytest.approx(0.78)
    assert record.mean_target == pytest.approx(0.76)
    assert record.instability_target == pytest.approx(0.04)


def test_first_evaluation_applies_fifo_when_population_is_full() -> None:
    evaluator = FakeEvaluator(
        {
            ("A", 101): 0.70,
            ("B", 102): 0.71,
            ("C", 103): 0.72,
        }
    )
    engine = RSSARECore(
        population_size=2,
        budget=3,
        evaluator=evaluator,
    )
    engine.run_first_evaluation(
        base_evaluation_index=1,
        architecture="A",
        training_seed=101,
    )
    engine.run_first_evaluation(
        base_evaluation_index=2,
        architecture="B",
        training_seed=102,
    )
    engine.run_first_evaluation(
        base_evaluation_index=3,
        architecture="C",
        training_seed=103,
    )

    assert [item.architecture for item in engine.population] == ["B", "C"]
    assert [item.birth_order for item in engine.population] == [1, 2]


def test_real_training_budget_cannot_be_exceeded() -> None:
    evaluator = FakeEvaluator({("A", 101): 0.78, ("A", 202): 0.74})
    engine = RSSARECore(
        population_size=2,
        budget=1,
        evaluator=evaluator,
    )
    engine.run_first_evaluation(
        base_evaluation_index=23,
        architecture="A",
        training_seed=101,
    )

    with pytest.raises(BudgetExhausted, match="1/1"):
        engine.run_repeat_evaluation(
            base_evaluation_index=23,
            training_seed=202,
        )

    assert engine.budget.used == 1
    assert evaluator.calls == [("A", 101)]
    assert engine.records.get(23).has_pair is False


def test_repeat_does_not_merge_natural_duplicate_architectures() -> None:
    evaluator = FakeEvaluator(
        {
            ("A", 101): 0.70,
            ("A", 102): 0.72,
            ("A", 201): 0.74,
        }
    )
    engine = RSSARECore(
        population_size=2,
        budget=3,
        evaluator=evaluator,
    )
    engine.run_first_evaluation(
        base_evaluation_index=10,
        architecture="A",
        training_seed=101,
    )
    engine.run_first_evaluation(
        base_evaluation_index=25,
        architecture="A",
        training_seed=102,
    )
    engine.run_repeat_evaluation(
        base_evaluation_index=10,
        training_seed=201,
    )

    assert len(engine.records) == 2
    assert engine.records.get(10).has_pair is True
    assert engine.records.get(25).has_pair is False


@dataclass(frozen=True)
class FakeMutation:
    architecture: int
    mutation_type: str = "fake_mutation"


@dataclass(frozen=True)
class FakeFitResult:
    model: tuple[int, int]
    training_loss: float
    mean_training_mse: float
    instability_training_mse: float
    observation_count: int
    paired_count: int


class SeedAwareEvaluator:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def __call__(self, architecture: int, training_seed: int) -> float:
        self.calls.append((architecture, training_seed))
        return 0.60 + (architecture % 10) / 100.0


def _fit_records(records, encodings) -> FakeFitResult:
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


def _predict_candidates(model, encodings) -> MultiTaskPrediction:
    del model
    count = len(encodings)
    means = torch.tensor(
        [0.80 - 0.03 * index for index in range(count)],
        dtype=torch.float32,
    )
    instability = torch.tensor(
        [0.10 if index == 0 else 0.01 for index in range(count)],
        dtype=torch.float32,
    )
    return MultiTaskPrediction(means, instability)


def _run_full_engine(*, budget: int):
    initialization = iter((10, 20))
    mutation_counter = iter(range(100, 1000))
    evaluator = SeedAwareEvaluator()
    progress = []
    result = repeat_stability_assisted_evolution(
        random_architecture_fn=lambda rng: next(initialization),
        mutate_fn=lambda parent, rng: FakeMutation(next(mutation_counter)),
        evaluate_fn=evaluator,
        encode_fn=lambda architecture: [float(architecture)],
        fit_surrogate_fn=_fit_records,
        predict_surrogate_fn=_predict_candidates,
        population_size=2,
        tournament_size=1,
        budget=budget,
        candidate_count=2,
        training_seed_base=10_000,
        stability_penalty_lambda=1.0,
        repeat_seed=77,
        search_rng=random.Random(123),
        repeat_policy_config=RepeatPolicyConfig(
            initial_population_size=2,
            warmup_pairs=1,
            repeat_interval=2,
            repeat_rate_beta=0.5,
        ),
        progress_fn=progress.append,
    )
    return result, evaluator, progress


def test_full_engine_keeps_repeats_out_of_population_and_first_history() -> None:
    result, evaluator, progress = _run_full_engine(budget=8)

    assert result.real_training_runs == 8
    assert len(evaluator.calls) == 8
    assert len(result.history) == 6
    assert len(result.repeat_evaluations) == 2
    assert len(result.candidate_batches) == 4
    assert [item.evaluation_id + 1 for item in result.population] == [5, 6]
    assert result.surrogate_fit.observation_count == 6
    assert result.surrogate_fit.paired_count == 2
    assert [event.event_type for event in result.budget_events] == [
        "first_evaluation",
        "first_evaluation",
        "repeat_evaluation",
        "first_evaluation",
        "first_evaluation",
        "repeat_evaluation",
        "first_evaluation",
        "first_evaluation",
    ]
    for event in result.budget_events:
        replicate_id = 1 if event.event_type == "first_evaluation" else 2
        assert event.training_seed == derive_training_seed(
            training_seed_base=10_000,
            base_evaluation_index=event.base_evaluation_index,
            replicate_id=replicate_id,
        )

    for index, item in enumerate(progress):
        if item.event_type == "repeat_evaluation":
            assert item.individual is None
            assert item.candidate_batch is None
            assert index > 0
            assert item.population == progress[index - 1].population

    records_by_index = {
        record.base_evaluation_index: record
        for record in result.paired_records
    }
    for individual in result.history:
        record = records_by_index[individual.evaluation_id + 1]
        assert individual.fitness == pytest.approx(record.accuracy_1)


def test_child_at_budget_limit_blocks_due_repeat() -> None:
    result, evaluator, _ = _run_full_engine(budget=5)

    assert result.real_training_runs == 5
    assert len(evaluator.calls) == 5
    assert len(result.history) == 4
    assert len(result.repeat_evaluations) == 1
    assert result.budget_events[-1].event_type == "first_evaluation"


def test_scheduled_repeat_at_budget_limit_stops_before_another_child() -> None:
    result, evaluator, _ = _run_full_engine(budget=6)

    assert result.real_training_runs == 6
    assert len(evaluator.calls) == 6
    assert len(result.history) == 4
    assert len(result.repeat_evaluations) == 2
    assert result.budget_events[-1].event_type == "repeat_evaluation"
