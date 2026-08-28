"""Toy tests for SA-RE candidate screening, budget, and aging."""

from __future__ import annotations

import random
import unittest
from dataclasses import dataclass

from src.search.surrogate_assisted_evolution import (
    SurrogateScoreResult,
    surrogate_assisted_evolution,
)
from src.search.regularized_evolution import regularized_evolution
from src.surrogate import SurrogateDataset


@dataclass(frozen=True)
class ToyMutationResult:
    architecture: str
    mutation_type: str


class AlternatingChoiceRng(random.Random):
    """Choice spy: every tournament contains repeated population members."""

    def __init__(self):
        super().__init__(0)
        self.choice_calls = 0

    def choice(self, sequence):
        member = sequence[self.choice_calls % len(sequence)]
        self.choice_calls += 1
        return member


class SurrogateAssistedEvolutionTests(unittest.TestCase):
    def test_initial_architectures_match_re_before_surrogate_is_used(self):
        def random_architecture(rng):
            return f"architecture-{rng.randrange(1_000_000_000)}"

        def mutate(parent, _rng):
            return ToyMutationResult(f"{parent}-child", "operation")

        re_result = regularized_evolution(
            random_architecture_fn=random_architecture,
            mutate_fn=mutate,
            evaluate_fn=lambda _architecture: 0.5,
            population_size=4,
            tournament_size=2,
            budget=4,
            rng=random.Random(2701),
        )
        score_calls = []
        sa_re_result = surrogate_assisted_evolution(
            random_architecture_fn=random_architecture,
            mutate_fn=mutate,
            evaluate_fn=lambda _architecture: 0.5,
            encode_fn=lambda _architecture: [1.0],
            surrogate_score_fn=lambda _dataset, _encodings: (
                score_calls.append(True)
                or SurrogateScoreResult(scores=(0.5,), training_mse=0.0)
            ),
            surrogate_dataset=SurrogateDataset(input_dim=1),
            population_size=4,
            tournament_size=2,
            budget=4,
            candidate_count=1,
            rng=random.Random(2701),
        )

        self.assertEqual(score_calls, [])
        self.assertEqual(
            [item.architecture for item in re_result.history],
            [item.architecture for item in sa_re_result.history],
        )

    def test_surrogate_cannot_consume_search_rng(self):
        search_rng = random.Random(2701)

        def score_with_rng_leak(_dataset, encodings):
            search_rng.random()
            return SurrogateScoreResult(
                scores=tuple(0.5 for _ in encodings),
                training_mse=0.0,
            )

        with self.assertRaisesRegex(RuntimeError, "consumed the search RNG"):
            surrogate_assisted_evolution(
                random_architecture_fn=lambda rng: (
                    f"architecture-{rng.randrange(1_000_000)}"
                ),
                mutate_fn=lambda parent, _rng: ToyMutationResult(
                    parent,
                    "identity",
                ),
                evaluate_fn=lambda _architecture: 0.5,
                encode_fn=lambda _architecture: [1.0],
                surrogate_score_fn=score_with_rng_leak,
                surrogate_dataset=SurrogateDataset(input_dim=1),
                population_size=2,
                tournament_size=2,
                budget=3,
                candidate_count=2,
                rng=search_rng,
            )

    def test_k5_selects_highest_prediction_and_evaluates_only_once(self):
        random_calls = []
        mutation_calls = []
        evaluation_calls = []
        progress_events = []

        def random_architecture(_rng):
            architecture = f"initial-{len(random_calls)}"
            random_calls.append(architecture)
            return architecture

        def mutate(parent, _rng):
            candidate_index = len(mutation_calls)
            result = ToyMutationResult(
                architecture=f"{parent}-candidate-{candidate_index}",
                mutation_type="operation",
            )
            mutation_calls.append(result)
            return result

        def evaluate(architecture):
            evaluation_calls.append(architecture)
            return 0.60 + len(evaluation_calls) * 0.01

        score_calls = []

        def score(dataset, encodings):
            score_calls.append((len(dataset), len(encodings)))
            return SurrogateScoreResult(
                scores=(0.65, 0.72, 0.68, 0.81, 0.70),
                training_mse=0.012,
            )

        result = surrogate_assisted_evolution(
            random_architecture_fn=random_architecture,
            mutate_fn=mutate,
            evaluate_fn=evaluate,
            encode_fn=lambda architecture: [float(len(architecture))],
            surrogate_score_fn=score,
            surrogate_dataset=SurrogateDataset(input_dim=1),
            population_size=2,
            tournament_size=3,
            budget=3,
            candidate_count=5,
            rng=random.Random(2710),
            progress_fn=progress_events.append,
        )

        self.assertEqual(len(random_calls), 2)
        self.assertEqual(len(mutation_calls), 5)
        self.assertEqual(len(evaluation_calls), 3)
        self.assertEqual(score_calls, [(2, 5)])
        self.assertEqual(result.evaluation_count, 3)
        self.assertEqual(len(result.population), 2)
        self.assertEqual(len(result.surrogate_dataset), 3)
        self.assertEqual(
            [member.evaluation_id for member in result.population],
            [1, 2],
        )
        batch = result.candidate_batches[0]
        self.assertEqual(batch.selected_candidate_index, 3)
        self.assertEqual(
            evaluation_calls[-1],
            mutation_calls[3].architecture,
        )
        self.assertEqual(
            sum(candidate.selected for candidate in batch.candidates),
            1,
        )
        self.assertEqual(progress_events[-1].candidate_batch, batch)

    def test_identity_duplicates_are_not_filtered_and_tie_selects_first(self):
        mutation_calls = []
        evaluation_calls = []

        def identity_mutation(parent, _rng):
            mutation_calls.append(parent)
            return ToyMutationResult(parent, "identity")

        def evaluate(architecture):
            evaluation_calls.append(architecture)
            return 0.50 + len(evaluation_calls) * 0.01

        result = surrogate_assisted_evolution(
            random_architecture_fn=lambda _rng: "same-architecture",
            mutate_fn=identity_mutation,
            evaluate_fn=evaluate,
            encode_fn=lambda _architecture: [1.0],
            surrogate_score_fn=lambda _dataset, _encodings: (
                SurrogateScoreResult(
                    scores=(0.5, 0.5, 0.5, 0.5, 0.5),
                    training_mse=0.0,
                )
            ),
            surrogate_dataset=SurrogateDataset(input_dim=1),
            population_size=1,
            tournament_size=4,
            budget=2,
            candidate_count=5,
            rng=random.Random(1),
        )

        self.assertEqual(len(mutation_calls), 5)
        self.assertEqual(len(evaluation_calls), 2)
        self.assertEqual(result.candidate_batches[0].selected_candidate_index, 0)
        self.assertTrue(
            all(
                candidate.architecture == "same-architecture"
                for candidate in result.candidate_batches[0].candidates
            )
        )
        self.assertEqual(result.history[-1].mutation_type, "identity")

    def test_tournament_is_with_replacement_and_uses_true_fitness(self):
        rng = AlternatingChoiceRng()
        next_initial = 0
        selected_parents = []

        def random_architecture(_rng):
            nonlocal next_initial
            architecture = f"initial-{next_initial}"
            next_initial += 1
            return architecture

        def evaluate(architecture):
            if architecture == "initial-0":
                return 0.40
            if architecture == "initial-1":
                return 0.90
            return 0.50

        def mutate(parent, _rng):
            selected_parents.append(parent)
            return ToyMutationResult(f"child-{len(selected_parents)}", "operation")

        result = surrogate_assisted_evolution(
            random_architecture_fn=random_architecture,
            mutate_fn=mutate,
            evaluate_fn=evaluate,
            encode_fn=lambda architecture: [float(len(architecture))],
            surrogate_score_fn=lambda _dataset, encodings: (
                SurrogateScoreResult(
                    scores=tuple(float(index) for index in range(len(encodings))),
                    training_mse=0.01,
                )
            ),
            surrogate_dataset=SurrogateDataset(input_dim=1),
            population_size=2,
            tournament_size=5,
            budget=3,
            candidate_count=5,
            rng=rng,
        )

        self.assertEqual(rng.choice_calls, 5)
        self.assertEqual(selected_parents, ["initial-1"] * 5)
        self.assertEqual(result.history[-1].parent_evaluation_id, 1)

    def test_each_cycle_generates_k_candidates_but_consumes_one_budget(self):
        mutation_calls = []
        evaluation_calls = []

        result = surrogate_assisted_evolution(
            random_architecture_fn=lambda _rng: f"init-{len(evaluation_calls)}",
            mutate_fn=lambda parent, _rng: (
                mutation_calls.append(parent)
                or ToyMutationResult(parent, "identity")
            ),
            evaluate_fn=lambda architecture: (
                evaluation_calls.append(architecture) or 0.5
            ),
            encode_fn=lambda _architecture: [1.0],
            surrogate_score_fn=lambda _dataset, encodings: (
                SurrogateScoreResult(
                    scores=tuple(0.5 for _ in encodings),
                    training_mse=0.01,
                )
            ),
            surrogate_dataset=SurrogateDataset(input_dim=1),
            population_size=2,
            tournament_size=3,
            budget=4,
            candidate_count=5,
            rng=random.Random(2),
        )

        self.assertEqual(len(mutation_calls), 10)
        self.assertEqual(len(evaluation_calls), 4)
        self.assertEqual(len(result.candidate_batches), 2)
        self.assertEqual(len(result.population), 2)
        self.assertEqual(
            [member.evaluation_id for member in result.population],
            [2, 3],
        )


if __name__ == "__main__":
    unittest.main()
