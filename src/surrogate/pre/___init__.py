"""Accuracy-surrogate components for surrogate-assisted evolution."""

from .dataset import SurrogateDataset, SurrogateObservation
from .model import AccuracySurrogate
from .multitask_dataset import (
    MultiTaskSurrogateDataset,
    MultiTaskSurrogateObservation,
    StabilityRecord,
)
from .multitask_model import MultiTaskPrediction, MultiTaskSurrogate
from .trainer import (
    SurrogateTrainingConfig,
    SurrogateTrainingResult,
    predict_accuracy,
    train_accuracy_surrogate,
)

__all__ = [
    "AccuracySurrogate",
    "MultiTaskPrediction",
    "MultiTaskSurrogate",
    "MultiTaskSurrogateDataset",
    "MultiTaskSurrogateObservation",
    "StabilityRecord",
    "SurrogateDataset",
    "SurrogateObservation",
    "SurrogateTrainingConfig",
    "SurrogateTrainingResult",
    "predict_accuracy",
    "train_accuracy_surrogate",
]
