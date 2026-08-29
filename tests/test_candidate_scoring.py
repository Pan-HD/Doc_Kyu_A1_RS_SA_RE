"""Explicit tests for RS-SA-RE candidate scoring."""

from __future__ import annotations

import math
import unittest

import torch

from src.evolution.candidate_scoring import (
    CandidateScoringConfig,
    score_candidates,
    select_candidate_by_score,
)
from src.surrogate.multitask_model import MultiTaskPrediction


def _prediction(
    predicted_mean: list[float],
    predicted_instability: list[float],
) -> MultiTaskPrediction:
    return MultiTaskPrediction(
        predicted_mean=torch.tensor(predicted_mean, dtype=torch.float32),
        predicted_instability=torch.tensor(
            predicted_instability,
            dtype=torch.float32,
        ),
    )


class CandidateScoringTests(unittest.TestCase):
    def test_explicit_penalty_changes_selected_candidate(self) -> None:
        prediction = _prediction(
            predicted_mean=[0.80, 0.77, 0.75],
            predicted_instability=[0.10, 0.01, 0.03],
        )
        result = score_candidates(
            prediction,
            CandidateScoringConfig(stability_penalty_lambda=1.0),
        )

        torch.testing.assert_close(
            result.scores,
            torch.tensor([0.70, 0.76, 0.72]),
        )
        self.assertEqual(result.selected_index, 1)
        self.assertEqual(int(torch.argmax(prediction.predicted_mean).item()), 0)

    def test_lambda_zero_matches_mean_argmax_exactly(self) -> None:
        prediction = _prediction(
            predicted_mean=[0.80, 0.77, 0.75],
            predicted_instability=[0.10, 0.01, 0.03],
        )
        result = score_candidates(
            prediction,
            CandidateScoringConfig(stability_penalty_lambda=0.0),
        )

        torch.testing.assert_close(result.scores, prediction.predicted_mean)
        self.assertEqual(
            result.selected_index,
            int(torch.argmax(prediction.predicted_mean).item()),
        )

    def test_five_candidates_produce_five_scores(self) -> None:
        prediction = _prediction(
            predicted_mean=[0.70, 0.71, 0.72, 0.73, 0.74],
            predicted_instability=[0.05, 0.04, 0.03, 0.02, 0.01],
        )
        result = score_candidates(prediction, CandidateScoringConfig())

        self.assertEqual(tuple(result.scores.shape), (5,))
        self.assertEqual(result.selected_index, int(torch.argmax(result.scores).item()))
        self.assertEqual(result.stability_penalty_lambda, 1.0)

    def test_convenience_selector_matches_full_result(self) -> None:
        prediction = _prediction(
            predicted_mean=[0.65, 0.72, 0.71],
            predicted_instability=[0.01, 0.08, 0.02],
        )
        config = CandidateScoringConfig(stability_penalty_lambda=0.5)

        self.assertEqual(
            select_candidate_by_score(prediction, config),
            score_candidates(prediction, config).selected_index,
        )

    def test_tie_selects_first_candidate_without_rng(self) -> None:
        prediction = _prediction(
            predicted_mean=[0.80, 0.80, 0.70],
            predicted_instability=[0.10, 0.10, 0.00],
        )
        result = score_candidates(prediction, CandidateScoringConfig())

        torch.testing.assert_close(
            result.scores,
            torch.tensor([0.70, 0.70, 0.70]),
        )
        self.assertEqual(result.selected_index, 0)

    def test_negative_lambda_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            CandidateScoringConfig(stability_penalty_lambda=-0.1)

    def test_nonfinite_lambda_is_rejected(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    CandidateScoringConfig(stability_penalty_lambda=value)

    def test_negative_instability_is_rejected(self) -> None:
        prediction = _prediction(
            predicted_mean=[0.70, 0.71],
            predicted_instability=[0.01, -0.01],
        )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            score_candidates(prediction, CandidateScoringConfig())

    def test_nonfinite_prediction_is_rejected(self) -> None:
        prediction = _prediction(
            predicted_mean=[0.70, math.nan],
            predicted_instability=[0.01, 0.02],
        )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            score_candidates(prediction, CandidateScoringConfig())

    def test_empty_candidate_batch_is_rejected(self) -> None:
        prediction = _prediction([], [])
        with self.assertRaisesRegex(ValueError, "at least one"):
            score_candidates(prediction, CandidateScoringConfig())


if __name__ == "__main__":
    unittest.main()
