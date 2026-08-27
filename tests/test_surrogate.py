"""Unit tests for the provisional SA-RE accuracy surrogate."""

from __future__ import annotations

import unittest

import torch

from src.surrogate import (
    AccuracySurrogate,
    SurrogateDataset,
    SurrogateTrainingConfig,
    predict_accuracy,
    train_accuracy_surrogate,
)


class AccuracySurrogateTests(unittest.TestCase):
    def test_model_maps_batch_280_to_finite_batch_scores(self):
        model = AccuracySurrogate()
        inputs = torch.zeros(7, 280)
        outputs = model(inputs)
        self.assertEqual(tuple(outputs.shape), (7,))
        self.assertTrue(bool(torch.isfinite(outputs).all().item()))

    def test_dataset_preserves_auditable_observations(self):
        dataset = SurrogateDataset(input_dim=4)
        architecture = {"name": "duplicate-is-allowed"}
        dataset.add(
            architecture=architecture,
            encoding=[1, 0, 0, 1],
            target_accuracy=0.70,
            evaluation_index=1,
        )
        dataset.add(
            architecture=architecture,
            encoding=[1, 0, 0, 1],
            target_accuracy=0.72,
            evaluation_index=2,
        )

        self.assertEqual(len(dataset), 2)
        self.assertIs(dataset.observations[0].architecture, architecture)
        self.assertEqual(dataset.observations[1].evaluation_index, 2)
        features, targets = dataset.tensors()
        self.assertEqual(tuple(features.shape), (2, 4))
        self.assertEqual(tuple(targets.shape), (2,))

    def test_same_data_and_seed_produce_same_trained_predictions(self):
        dataset = SurrogateDataset(input_dim=4)
        for index in range(1, 7):
            dataset.add(
                architecture={"index": index},
                encoding=[
                    float(index % 2),
                    float((index // 2) % 2),
                    float(index) / 10.0,
                    1.0,
                ],
                target_accuracy=0.50 + index * 0.03,
                evaluation_index=index,
            )
        config = SurrogateTrainingConfig(
            input_dim=4,
            hidden_dims=(8, 4),
            learning_rate=1e-3,
            weight_decay=1e-4,
            steps=30,
            seed=902710,
        )

        torch.manual_seed(123456)
        rng_state_before = torch.random.get_rng_state().clone()
        first = train_accuracy_surrogate(dataset, config)
        rng_state_after = torch.random.get_rng_state().clone()
        second = train_accuracy_surrogate(dataset, config)

        candidates = ([0.0, 1.0, 0.3, 1.0], [1.0, 0.0, 0.7, 1.0])
        first_predictions = predict_accuracy(first.model, candidates)
        second_predictions = predict_accuracy(second.model, candidates)
        torch.testing.assert_close(
            torch.tensor(first_predictions),
            torch.tensor(second_predictions),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            rng_state_before,
            rng_state_after,
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(first.observation_count, 6)
        self.assertGreaterEqual(first.training_mse, 0.0)

    def test_invalid_target_and_duplicate_evaluation_index_are_rejected(self):
        dataset = SurrogateDataset(input_dim=2)
        with self.assertRaisesRegex(ValueError, "target_accuracy"):
            dataset.add(
                architecture="bad",
                encoding=[0, 1],
                target_accuracy=72.0,
                evaluation_index=1,
            )
        dataset.add(
            architecture="valid",
            encoding=[0, 1],
            target_accuracy=0.72,
            evaluation_index=1,
        )
        with self.assertRaisesRegex(ValueError, "already present"):
            dataset.add(
                architecture="duplicate-index",
                encoding=[1, 0],
                target_accuracy=0.73,
                evaluation_index=1,
            )


if __name__ == "__main__":
    unittest.main()
