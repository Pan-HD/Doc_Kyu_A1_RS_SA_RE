"""Tests for RS-SA-RE label construction without policy or lambda tuning."""

from __future__ import annotations

import math
import unittest

import torch

from src.search.rs_sa_re import RSSARENotConfiguredError, RSSAREScaffold
from src.surrogate import (
    MultiTaskPrediction,
    MultiTaskSurrogate,
    MultiTaskSurrogateDataset,
    StabilityRecord,
)


class DummyMultiTaskSurrogate(MultiTaskSurrogate):
    def forward(self, inputs: torch.Tensor) -> MultiTaskPrediction:
        self.validate_inputs(inputs)
        batch_size = inputs.shape[0]
        return MultiTaskPrediction(
            predicted_mean=torch.zeros(batch_size),
            predicted_instability=torch.zeros(batch_size),
        )


class StabilityLabelTests(unittest.TestCase):
    def test_paired_accuracy_constructs_mean_and_instability(self) -> None:
        record = StabilityRecord(
            architecture="architecture-a",
            accuracy_seed_1=0.70,
            accuracy_seed_2=0.76,
        )

        self.assertTrue(record.has_pair)
        self.assertTrue(record.mean_target_available)
        self.assertTrue(record.instability_target_available)
        self.assertAlmostEqual(record.mean_target, 0.73)
        self.assertAlmostEqual(record.instability_target, 0.06)

    def test_single_accuracy_has_mean_but_no_instability_label(self) -> None:
        record = StabilityRecord(
            architecture="architecture-b",
            accuracy_seed_1=0.70,
            accuracy_seed_2=None,
        )

        self.assertFalse(record.has_pair)
        self.assertTrue(record.mean_target_available)
        self.assertFalse(record.instability_target_available)
        self.assertAlmostEqual(record.mean_target, 0.70)
        self.assertIsNone(record.instability_target)

    def test_accuracy_labels_must_be_finite_unit_fractions(self) -> None:
        invalid_values = (-0.01, 1.01, math.nan, math.inf, -math.inf)
        for invalid_value in invalid_values:
            with self.subTest(field="accuracy_seed_1", value=invalid_value):
                with self.assertRaises(ValueError):
                    StabilityRecord("architecture", invalid_value)
            with self.subTest(field="accuracy_seed_2", value=invalid_value):
                with self.assertRaises(ValueError):
                    StabilityRecord("architecture", 0.70, invalid_value)

    def test_dataset_builds_masked_multi_task_tensors(self) -> None:
        dataset = MultiTaskSurrogateDataset(input_dim=280)
        dataset.add(
            record=StabilityRecord("single", 0.70),
            encoding=torch.zeros(280),
        )
        dataset.add(
            record=StabilityRecord("paired", 0.70, 0.76),
            encoding=torch.ones(280),
        )

        features, means, instabilities, mask = dataset.tensors()

        self.assertEqual(tuple(features.shape), (2, 280))
        torch.testing.assert_close(means, torch.tensor([0.70, 0.73]))
        torch.testing.assert_close(instabilities, torch.tensor([0.00, 0.06]))
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(mask.tolist(), [False, True])
        self.assertEqual(instabilities[0].item(), 0.0)
        self.assertFalse(mask[0].item())

    def test_dataset_rejects_bad_encoding(self) -> None:
        dataset = MultiTaskSurrogateDataset(input_dim=280)
        record = StabilityRecord("architecture", 0.70)
        with self.assertRaisesRegex(ValueError, "280"):
            dataset.add(record=record, encoding=torch.zeros(279))
        encoding = torch.zeros(280)
        encoding[5] = math.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            dataset.add(record=record, encoding=encoding)

    def test_model_interface_and_scaffold_remain_non_runnable(self) -> None:
        surrogate = DummyMultiTaskSurrogate(input_dim=280)
        prediction = surrogate(torch.zeros(3, 280))
        self.assertEqual(tuple(prediction.predicted_mean.shape), (3,))
        self.assertEqual(tuple(prediction.predicted_instability.shape), (3,))

        scaffold = RSSAREScaffold(
            surrogate=surrogate,
            stability_dataset=MultiTaskSurrogateDataset(input_dim=280),
        )
        self.assertFalse(scaffold.repeat_policy_configured)
        with self.assertRaisesRegex(RSSARENotConfiguredError, "repeat policy"):
            scaffold.require_repeat_policy()
        with self.assertRaisesRegex(RSSARENotConfiguredError, "not implemented"):
            scaffold.run()


if __name__ == "__main__":
    unittest.main()
