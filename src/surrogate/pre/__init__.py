"""Accuracy-surrogate components for surrogate-assisted evolution."""

from .dataset import SurrogateDataset, SurrogateObservation
from .model import AccuracySurrogate
from .trainer import (
    SurrogateTrainingConfig,
    SurrogateTrainingResult,
    predict_accuracy,
    train_accuracy_surrogate,
)

__all__ = [
    "AccuracySurrogate",
    "SurrogateDataset",
    "SurrogateObservation",
    "SurrogateTrainingConfig",
    "SurrogateTrainingResult",
    "predict_accuracy",
    "train_accuracy_surrogate",
]
