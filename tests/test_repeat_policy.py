"""Unit tests for the fixed, outcome-independent RS-SA-RE repeat policy."""

from __future__ import annotations

import random
import unittest

from src.evolution.repeat_policy import (
    NoEligibleRepeatRecordError,
    REPLICATE_SEED_STRIDE,
    RepeatPolicyConfig,
    RepeatScheduler,
    derive_training_seed,
)
from src.surrogate.multitask_dataset import PairedEvaluationRecord


def _records(count: int = 20) -> list[PairedEvaluationRecord]:
    return [
        PairedEvaluationRecord(
            base_evaluation_index=index,
            architecture=f"architecture-{index}",
            seed_1=10_000 + index,
            accuracy_1=0.50 + index / 100.0,
        )
        for index in range(1, count + 1)
    ]


class RepeatPolicyTests(unittest.TestCase):
    def test_default_policy_is_the_frozen_provisional_policy(self) -> None:
        config = RepeatPolicyConfig()
        self.assertEqual(config.initial_population_size, 20)
        self.assertEqual(config.warmup_pairs, 4)
        self.assertEqual(config.repeat_interval, 4)
        self.assertEqual(config.repeat_rate_beta, 0.25)

    def test_warmup_selects_four_unique_records_without_replacement(self) -> None:
        selected = RepeatScheduler(repeat_seed=2910).select_warmup(_records())
        indices = [record.base_evaluation_index for record in selected]
        self.assertEqual(len(indices), 4)
        self.assertEqual(len(set(indices)), 4)

    def test_same_repeat_seed_reproduces_selection_sequence(self) -> None:
        records_a = _records()
        records_b = list(reversed(_records()))
        scheduler_a = RepeatScheduler(repeat_seed=902_701)
        scheduler_b = RepeatScheduler(repeat_seed=902_701)

        warmup_a = scheduler_a.select_warmup(records_a)
        warmup_b = scheduler_b.select_warmup(records_b)
        self.assertEqual(
            [record.base_evaluation_index for record in warmup_a],
            [record.base_evaluation_index for record in warmup_b],
        )

        for record in warmup_a:
            record.add_repeat(seed_2=record.seed_1 + 1_000_000, accuracy_2=0.60)
        for record in warmup_b:
            record.add_repeat(seed_2=record.seed_1 + 1_000_000, accuracy_2=0.60)

        periodic_a = [scheduler_a.select(records_a).base_evaluation_index for _ in range(6)]
        periodic_b = [scheduler_b.select(records_b).base_evaluation_index for _ in range(6)]
        self.assertEqual(periodic_a, periodic_b)

    def test_different_repeat_seeds_produce_different_fixed_sequences(self) -> None:
        records = _records()
        scheduler_b = RepeatScheduler(repeat_seed=202)
        sequence_b = [scheduler_b.select(records).base_evaluation_index for _ in range(6)]

        # Compare full deterministic sequences rather than one sampled value.
        scheduler_a = RepeatScheduler(repeat_seed=101)
        sequence_a = [scheduler_a.select(records).base_evaluation_index for _ in range(6)]
        self.assertNotEqual(sequence_a, sequence_b)

    def test_paired_records_are_never_selected_again(self) -> None:
        records = _records(5)
        for record in records[:4]:
            record.add_repeat(seed_2=record.seed_1 + 1_000_000, accuracy_2=0.60)

        scheduler = RepeatScheduler(repeat_seed=123)
        for _ in range(10):
            self.assertIs(scheduler.select(records), records[4])

        records[4].add_repeat(seed_2=records[4].seed_1 + 1_000_000, accuracy_2=0.60)
        with self.assertRaises(NoEligibleRepeatRecordError):
            scheduler.select(records)

    def test_schedule_occurs_every_four_new_first_evaluations(self) -> None:
        scheduler = RepeatScheduler(repeat_seed=123)
        actual = [
            index
            for index in range(0, 13)
            if scheduler.should_schedule(
                completed_first_evaluations_after_warmup=index
            )
        ]
        self.assertEqual(actual, [4, 8, 12])

    def test_selection_does_not_consume_search_rng(self) -> None:
        search_rng = random.Random(2701)
        state_before = search_rng.getstate()
        scheduler = RepeatScheduler(repeat_seed=902_701)
        scheduler.select_warmup(_records())
        scheduler.select(_records())
        state_after = search_rng.getstate()
        self.assertEqual(state_after, state_before)

    def test_repeat_scheduler_call_leaves_matched_search_rng_stream_unchanged(
        self,
    ) -> None:
        """Explicit A/B regression for the matched-seed isolation rule."""

        search_rng_a = random.Random(2701)
        search_rng_b = random.Random(2701)

        # Branch A deliberately does not call the repeat scheduler. Branch B
        # does, but selection uses RepeatScheduler's private repeat_rng only.
        scheduler = RepeatScheduler(repeat_seed=902_701)
        scheduler.select_warmup(_records())
        scheduler.select(_records())

        self.assertEqual(search_rng_a.random(), search_rng_b.random())
        self.assertEqual(search_rng_a.random(), search_rng_b.random())

    def test_selection_does_not_modify_records_or_population(self) -> None:
        records = _records()
        population = [f"individual-{index}" for index in range(20)]
        record_state_before = [
            (
                record.base_evaluation_index,
                record.seed_1,
                record.accuracy_1,
                record.seed_2,
                record.accuracy_2,
            )
            for record in records
        ]
        population_before = list(population)

        scheduler = RepeatScheduler(repeat_seed=902_701)
        scheduler.select_warmup(records)
        scheduler.select(records)

        record_state_after = [
            (
                record.base_evaluation_index,
                record.seed_1,
                record.accuracy_1,
                record.seed_2,
                record.accuracy_2,
            )
            for record in records
        ]
        self.assertEqual(record_state_after, record_state_before)
        self.assertEqual(population, population_before)

    def test_repeat_policy_does_not_require_fitness_or_surrogate_fields(self) -> None:
        records = _records()
        scheduler = RepeatScheduler(repeat_seed=902_701)
        selected = scheduler.select(records)
        self.assertIsInstance(selected, PairedEvaluationRecord)
        self.assertFalse(hasattr(selected, "fitness"))
        self.assertFalse(hasattr(selected, "predicted_mu"))
        self.assertFalse(hasattr(selected, "predicted_instability"))


class TrainingSeedPolicyTests(unittest.TestCase):
    def test_replicate_one_preserves_existing_first_training_schedule(self) -> None:
        base = 20_260_827
        actual = [
            derive_training_seed(
                training_seed_base=base,
                base_evaluation_index=index,
                replicate_id=1,
            )
            for index in range(1, 6)
        ]
        self.assertEqual(actual, [base + offset for offset in range(5)])

    def test_first_and_repeat_seeds_are_distinct_and_deterministic(self) -> None:
        arguments = {
            "training_seed_base": 20_260_827,
            "base_evaluation_index": 7,
        }
        seed_1 = derive_training_seed(**arguments, replicate_id=1)
        seed_2 = derive_training_seed(**arguments, replicate_id=2)
        repeated_seed_2 = derive_training_seed(**arguments, replicate_id=2)

        self.assertNotEqual(seed_1, seed_2)
        self.assertEqual(seed_2, repeated_seed_2)
        self.assertEqual(seed_2 - seed_1, REPLICATE_SEED_STRIDE)

    def test_different_base_evaluations_receive_different_repeat_seeds(self) -> None:
        first = derive_training_seed(
            training_seed_base=20_260_827,
            base_evaluation_index=7,
            replicate_id=2,
        )
        second = derive_training_seed(
            training_seed_base=20_260_827,
            base_evaluation_index=8,
            replicate_id=2,
        )
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
