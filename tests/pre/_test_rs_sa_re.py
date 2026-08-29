from collections.abc import Sequence

import pytest

from src.evolution.rs_sa_re import BudgetExhausted, RSSARECore


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

