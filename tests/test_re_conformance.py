"""Fast toy tests for Real et al. Algorithm 1 conformance.

No CNN, dataset, PyTorch model, or GPU is used in this test module.
"""

from __future__ import annotations

import random
import unittest
from collections import deque
from dataclasses import dataclass

from src.search.regularized_evolution import (
    RANDOM_INITIALIZATION,
    EvaluatedIndividual,
    regularized_evolution,
    sample_tournament,
    tournament_select,
)


@dataclass(frozen=True)
class ToyMutationResult:
    architecture: str
    mutation_type: str


class RepeatingChoiceRng:
    """Choice spy that always returns the first member."""

    def __init__(self):
        self.choice_calls = 0
        self.seen_sequences = []

    def choice(self, sequence):
        self.choice_calls += 1
        self.seen_sequences.append(tuple(sequence))
        return sequence[0]


class ScriptedChoiceRng:
    """Return population members in a requested evaluation-ID order."""

    def __init__(self, evaluation_ids):
        self.evaluation_ids = iter(evaluation_ids)

    def choice(self, sequence):
        requested_id = next(self.evaluation_ids)
        return next(
            individual
            for individual in sequence
            if individual.evaluation_id == requested_id
        )


def make_individual(
    evaluation_id: int,
    fitness: float,
) -> EvaluatedIndividual[str]:
    return EvaluatedIndividual(
        architecture=f"architecture-{evaluation_id}",
        fitness=fitness,
        evaluation_id=evaluation_id,
        mutation_type=RANDOM_INITIALIZATION,
    )


class TournamentConformanceTests(unittest.TestCase):
    def test_tournament_sampling_is_with_replacement(self):
        population = deque(
            [
                make_individual(0, 0.1),
                make_individual(1, 0.2),
            ]
        )
        rng = RepeatingChoiceRng()

        sample = sample_tournament(
            population=population,
            sample_size=5,
            rng=rng,
        )

        self.assertEqual(len(sample), 5)
        self.assertEqual(rng.choice_calls, 5)
        self.assertTrue(all(member is population[0] for member in sample))
        self.assertTrue(
            all(
                seen == tuple(population)
                for seen in rng.seen_sequences
            )
        )

    def test_best_sampled_parent_is_selected(self):
        population = deque(
            [
                make_individual(0, 0.80),
                make_individual(1, 0.95),
                make_individual(2, 0.70),
            ]
        )
        rng = ScriptedChoiceRng([2, 0, 2, 1])

        parent = tournament_select(
            population=population,
            sample_size=4,
            rng=rng,
        )

        self.assertEqual(parent.evaluation_id, 1)
        self.assertEqual(parent.fitness, 0.95)


class EvolutionConformanceTests(unittest.TestCase):
    def test_random_initialization_fills_deque_population(self):
        counters = {
            "random": 0,
            "mutation": 0,
            "evaluation": 0,
        }

        def random_architecture(_rng):
            architecture = f"initial-{counters['random']}"
            counters["random"] += 1
            return architecture

        def mutate(_parent, _rng):
            counters["mutation"] += 1
            return ToyMutationResult("unused", "operation")

        def evaluate(_architecture):
            counters["evaluation"] += 1
            return counters["evaluation"] / 10.0

        result = regularized_evolution(
            random_architecture_fn=random_architecture,
            mutate_fn=mutate,
            evaluate_fn=evaluate,
            population_size=4,
            tournament_size=7,
            budget=4,
            rng=random.Random(20260826),
        )

        self.assertIsInstance(result.population, deque)
        self.assertEqual(len(result.population), 4)
        self.assertEqual(len(result.history), 4)
        self.assertEqual(result.evaluation_count, 4)
        self.assertEqual(counters["random"], 4)
        self.assertEqual(counters["evaluation"], 4)
        self.assertEqual(counters["mutation"], 0)
        self.assertTrue(
            all(
                individual.mutation_type == RANDOM_INITIALIZATION
                for individual in result.history
            )
        )

    def test_each_cycle_mutates_and_evaluates_exactly_once(self):
        counters = {
            "random": 0,
            "mutation": 0,
            "evaluation": 0,
        }

        def random_architecture(_rng):
            architecture = f"initial-{counters['random']}"
            counters["random"] += 1
            return architecture

        def mutate(parent, _rng):
            counters["mutation"] += 1
            return ToyMutationResult(
                architecture=f"{parent}-child-{counters['mutation']}",
                mutation_type="hidden_state",
            )

        def evaluate(architecture):
            counters["evaluation"] += 1
            return float(len(architecture))

        result = regularized_evolution(
            random_architecture_fn=random_architecture,
            mutate_fn=mutate,
            evaluate_fn=evaluate,
            population_size=3,
            tournament_size=5,
            budget=8,
            rng=random.Random(20260826),
        )

        self.assertEqual(counters["random"], 3)
        self.assertEqual(counters["mutation"], 5)
        self.assertEqual(counters["evaluation"], 8)
        self.assertEqual(result.evaluation_count, 8)
        self.assertEqual(len(result.history), 8)
        self.assertEqual(len(result.population), 3)
        self.assertEqual(
            [member.evaluation_id for member in result.population],
            [5, 6, 7],
        )
        self.assertTrue(
            all(
                child.parent_evaluation_id is not None
                for child in result.history[3:]
            )
        )

    def test_child_is_appended_and_oldest_is_removed(self):
        next_initial = 0

        def random_architecture(_rng):
            nonlocal next_initial
            architecture = f"initial-{next_initial}"
            next_initial += 1
            return architecture

        def mutate(parent, _rng):
            return ToyMutationResult(
                architecture=f"child-of-{parent}",
                mutation_type="operation",
            )

        result = regularized_evolution(
            random_architecture_fn=random_architecture,
            mutate_fn=mutate,
            evaluate_fn=lambda _architecture: 1.0,
            population_size=3,
            tournament_size=1,
            budget=4,
            rng=random.Random(7),
        )

        self.assertEqual(
            [member.evaluation_id for member in result.population],
            [1, 2, 3],
        )
        self.assertNotIn(result.history[0], result.population)
        self.assertIs(result.population[-1], result.history[-1])
        self.assertEqual(len(result.population), 3)
        self.assertEqual(len(result.history), 4)

    def test_identity_duplicate_is_evaluated_and_consumes_budget(self):
        counters = {
            "mutation": 0,
            "evaluation": 0,
        }

        def identity_mutation(parent, _rng):
            counters["mutation"] += 1
            return ToyMutationResult(
                architecture=parent,
                mutation_type="identity",
            )

        def evaluate(_architecture):
            counters["evaluation"] += 1
            # A different score proves that every duplicate reaches evaluator.
            return counters["evaluation"] / 100.0

        result = regularized_evolution(
            random_architecture_fn=lambda _rng: "duplicate-genotype",
            mutate_fn=identity_mutation,
            evaluate_fn=evaluate,
            population_size=2,
            tournament_size=4,
            budget=6,
            rng=random.Random(20260826),
        )

        self.assertEqual(counters["evaluation"], 6)
        self.assertEqual(counters["mutation"], 4)
        self.assertEqual(result.evaluation_count, 6)
        self.assertEqual(len(result.history), 6)
        self.assertEqual(len(result.population), 2)
        self.assertTrue(
            all(
                individual.architecture == "duplicate-genotype"
                for individual in result.history
            )
        )
        self.assertEqual(
            [individual.mutation_type for individual in result.history[2:]],
            ["identity"] * 4,
        )
        self.assertEqual(
            [individual.fitness for individual in result.history],
            [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        )
        self.assertEqual(
            [member.evaluation_id for member in result.population],
            [4, 5],
        )

    def test_budget_is_exact_when_multiple_cycles_run(self):
        evaluation_calls = []
        mutation_calls = []

        def evaluate(architecture):
            evaluation_calls.append(architecture)
            return 0.5

        def mutate(parent, _rng):
            mutation_calls.append(parent)
            return ToyMutationResult(parent, "identity")

        result = regularized_evolution(
            random_architecture_fn=lambda _rng: "same",
            mutate_fn=mutate,
            evaluate_fn=evaluate,
            population_size=5,
            tournament_size=10,
            budget=17,
            rng=random.Random(1),
        )

        self.assertEqual(len(evaluation_calls), 17)
        self.assertEqual(len(mutation_calls), 12)
        self.assertEqual(result.evaluation_count, 17)
        self.assertEqual(len(result.population), 5)

    def test_invalid_budget_is_rejected_before_any_evaluation(self):
        evaluation_calls = []

        with self.assertRaisesRegex(
            ValueError,
            "budget must be at least population_size",
        ):
            regularized_evolution(
                random_architecture_fn=lambda _rng: "architecture",
                mutate_fn=lambda parent, _rng: ToyMutationResult(
                    parent,
                    "identity",
                ),
                evaluate_fn=lambda architecture: evaluation_calls.append(
                    architecture
                ),
                population_size=5,
                tournament_size=2,
                budget=4,
                rng=random.Random(0),
            )

        self.assertEqual(evaluation_calls, [])


if __name__ == "__main__":
    unittest.main()
