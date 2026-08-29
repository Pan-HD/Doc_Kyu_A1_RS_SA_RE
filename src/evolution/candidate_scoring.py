"""Pure candidate scoring for RS-SA-RE.

This module implements only the stability-aware ranking rule::

    score(a) = mu_hat(a) - lambda * d_hat(a)

It does not generate candidates, train a surrogate, evaluate a CNN, modify a
population, or consume real-training budget. Python's reserved word ``lambda``
is represented by the explicit configuration field
``stability_penalty_lambda``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..surrogate.multitask_model import MultiTaskPrediction


@dataclass(frozen=True, slots=True)
class CandidateScoringConfig:
    """Configuration for stability-aware candidate ranking.

    The default value 1.0 is a debug value for code-path validation, not a
    formally selected hyperparameter.
    """

    stability_penalty_lambda: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.stability_penalty_lambda, bool):
            raise TypeError("stability_penalty_lambda must be numeric")
        try:
            value = float(self.stability_penalty_lambda)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "stability_penalty_lambda must be numeric"
            ) from error
        if not math.isfinite(value):
            raise ValueError("stability_penalty_lambda must be finite")
        if value < 0.0:
            raise ValueError("stability_penalty_lambda must be non-negative")
        object.__setattr__(self, "stability_penalty_lambda", value)


@dataclass(frozen=True, slots=True)
class CandidateScoreResult:
    """Auditable predictions, scores, and the selected candidate index."""

    predicted_mean: torch.Tensor
    predicted_instability: torch.Tensor
    scores: torch.Tensor
    selected_index: int
    stability_penalty_lambda: float


def _validate_prediction(prediction: MultiTaskPrediction) -> None:
    if not isinstance(prediction, MultiTaskPrediction):
        raise TypeError("prediction must be a MultiTaskPrediction")
    predicted_mean = prediction.predicted_mean
    predicted_instability = prediction.predicted_instability
    if predicted_mean.ndim != 1:
        raise ValueError("candidate predictions must have shape [candidate_count]")
    if predicted_mean.numel() == 0:
        raise ValueError("at least one candidate prediction is required")
    if predicted_mean.device != predicted_instability.device:
        raise ValueError("candidate predictions must be on the same device")
    if not bool(torch.isfinite(predicted_mean).all().item()):
        raise ValueError("predicted_mean contains a non-finite value")
    if not bool(torch.isfinite(predicted_instability).all().item()):
        raise ValueError("predicted_instability contains a non-finite value")
    if bool((predicted_instability < 0.0).any().item()):
        raise ValueError("predicted_instability must be non-negative")


def score_candidates(
    prediction: MultiTaskPrediction,
    config: CandidateScoringConfig,
) -> CandidateScoreResult:
    """Score all candidates and select the first maximum deterministically.

    ``torch.argmax`` returns the first maximum, which fixes tie-breaking without
    consuming any RNG stream. With ``stability_penalty_lambda=0``, scores are
    exactly the predicted means and ranking degenerates to SA-style ranking.
    """

    if not isinstance(config, CandidateScoringConfig):
        raise TypeError("config must be a CandidateScoringConfig")
    _validate_prediction(prediction)

    scores = (
        prediction.predicted_mean
        - config.stability_penalty_lambda
        * prediction.predicted_instability
    )
    if not bool(torch.isfinite(scores).all().item()):
        raise FloatingPointError("candidate scores contain a non-finite value")

    selected_index = int(torch.argmax(scores).item())
    return CandidateScoreResult(
        predicted_mean=prediction.predicted_mean,
        predicted_instability=prediction.predicted_instability,
        scores=scores,
        selected_index=selected_index,
        stability_penalty_lambda=config.stability_penalty_lambda,
    )


def select_candidate_by_score(
    prediction: MultiTaskPrediction,
    config: CandidateScoringConfig,
) -> int:
    """Return only the selected index for callers that do not need diagnostics."""

    return score_candidates(prediction, config).selected_index


__all__ = [
    "CandidateScoreResult",
    "CandidateScoringConfig",
    "score_candidates",
    "select_candidate_by_score",
]
