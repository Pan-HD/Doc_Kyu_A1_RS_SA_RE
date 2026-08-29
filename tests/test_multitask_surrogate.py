"""Tests for the concrete RS-SA-RE model and masked multi-task loss."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F
from torch import nn

from src.surrogate import (
    MULTITASK_LOSS_ALPHA,
    MultiTaskPrediction,
    MultiTaskSurrogate,
    masked_multitask_mse_loss,
)


class MultiTaskModelTests(unittest.TestCase):
    def test_output_shapes_are_batch_vectors(self) -> None:
        model = MultiTaskSurrogate(input_dim=280)
        prediction = model(torch.zeros(7, 280))

        self.assertEqual(tuple(prediction.predicted_mean.shape), (7,))
        self.assertEqual(tuple(prediction.predicted_instability.shape), (7,))
        mu_hat, d_hat = prediction
        self.assertIs(mu_hat, prediction.predicted_mean)
        self.assertIs(d_hat, prediction.predicted_instability)

    def test_predictions_are_finite_and_instability_is_nonnegative(self) -> None:
        torch.manual_seed(20260829)
        model = MultiTaskSurrogate(input_dim=280)
        prediction = model(torch.randn(16, 280))

        self.assertTrue(bool(torch.isfinite(prediction.predicted_mean).all()))
        self.assertTrue(
            bool(torch.isfinite(prediction.predicted_instability).all())
        )
        self.assertTrue(bool((prediction.predicted_instability >= 0).all()))

    def test_expected_280_32_16_architecture_and_no_sigmoid(self) -> None:
        model = MultiTaskSurrogate(input_dim=280)

        self.assertIsInstance(model.trunk[0], nn.Linear)
        self.assertEqual((model.trunk[0].in_features, model.trunk[0].out_features), (280, 32))
        self.assertIsInstance(model.trunk[1], nn.ReLU)
        self.assertIsInstance(model.trunk[2], nn.Linear)
        self.assertEqual((model.trunk[2].in_features, model.trunk[2].out_features), (32, 16))
        self.assertIsInstance(model.trunk[3], nn.ReLU)
        self.assertEqual((model.mean_head.in_features, model.mean_head.out_features), (16, 1))
        self.assertEqual(
            (model.instability_head.in_features, model.instability_head.out_features),
            (16, 1),
        )
        self.assertFalse(any(isinstance(module, nn.Sigmoid) for module in model.modules()))

    def test_nonfinite_input_is_rejected(self) -> None:
        model = MultiTaskSurrogate(input_dim=280)
        inputs = torch.zeros(2, 280)
        inputs[0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            model(inputs)


class MaskedMultiTaskLossTests(unittest.TestCase):
    def test_alpha_is_frozen_at_one(self) -> None:
        self.assertEqual(MULTITASK_LOSS_ALPHA, 1.0)

    def test_zero_paired_labels_produce_finite_autograd_loss(self) -> None:
        predicted_mean = torch.tensor([0.6, 0.8], requires_grad=True)
        predicted_instability = torch.tensor([0.1, 0.2], requires_grad=True)
        result = masked_multitask_mse_loss(
            MultiTaskPrediction(predicted_mean, predicted_instability),
            mean_targets=torch.tensor([0.7, 0.75]),
            instability_targets=torch.zeros(2),
            stability_mask=torch.tensor([False, False]),
        )

        self.assertTrue(bool(torch.isfinite(result.total_loss)))
        self.assertTrue(bool(torch.isfinite(result.instability_loss)))
        self.assertEqual(result.paired_count, 0)
        self.assertEqual(result.instability_loss.item(), 0.0)
        result.total_loss.backward()
        self.assertIsNotNone(predicted_instability.grad)
        torch.testing.assert_close(
            predicted_instability.grad,
            torch.zeros_like(predicted_instability),
        )

    def test_instability_loss_uses_only_masked_records(self) -> None:
        prediction = MultiTaskPrediction(
            predicted_mean=torch.tensor([0.5, 0.5]),
            predicted_instability=torch.tensor([0.9, 0.2]),
        )
        result = masked_multitask_mse_loss(
            prediction,
            mean_targets=torch.tensor([0.5, 0.5]),
            instability_targets=torch.tensor([0.0, 0.1]),
            stability_mask=torch.tensor([False, True]),
        )

        self.assertEqual(result.paired_count, 1)
        self.assertAlmostEqual(result.instability_loss.item(), 0.01, places=6)

    def test_instability_head_receives_gradient_with_paired_labels(self) -> None:
        torch.manual_seed(20260829)
        model = MultiTaskSurrogate(input_dim=280)
        prediction = model(torch.ones(4, 280))
        result = masked_multitask_mse_loss(
            prediction,
            mean_targets=torch.full((4,), 0.5),
            instability_targets=torch.zeros(4),
            stability_mask=torch.tensor([True, False, True, False]),
        )
        result.total_loss.backward()

        gradient = model.instability_head.bias.grad
        self.assertIsNotNone(gradient)
        assert gradient is not None
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.abs().sum().item()), 0.0)

    def test_mean_loss_uses_single_and_paired_records(self) -> None:
        predicted_mean = torch.tensor([0.0, 0.0], requires_grad=True)
        prediction = MultiTaskPrediction(
            predicted_mean=predicted_mean,
            predicted_instability=torch.tensor([0.2, 0.2], requires_grad=True),
        )
        mean_targets = torch.tensor([0.5, 1.0])
        result = masked_multitask_mse_loss(
            prediction,
            mean_targets=mean_targets,
            instability_targets=torch.tensor([0.0, 0.1]),
            stability_mask=torch.tensor([False, True]),
        )

        expected = F.mse_loss(predicted_mean, mean_targets)
        torch.testing.assert_close(result.mean_loss, expected)
        self.assertAlmostEqual(result.mean_loss.item(), 0.625, places=6)

    def test_shape_mismatch_is_rejected(self) -> None:
        prediction = MultiTaskPrediction(
            predicted_mean=torch.zeros(2),
            predicted_instability=torch.zeros(2),
        )
        with self.assertRaisesRegex(ValueError, "mean_targets"):
            masked_multitask_mse_loss(
                prediction,
                mean_targets=torch.zeros(3),
                instability_targets=torch.zeros(2),
                stability_mask=torch.zeros(2, dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
