"""Accuracy-surrogate components for surrogate-assisted evolution."""

from .dataset import SurrogateDataset, SurrogateObservation
from .model import AccuracySurrogate
from .multitask_dataset import (
    MultiTaskSurrogateDataset,
    MultiTaskSurrogateObservation,
    StabilityRecord,
)
from .multitask_loss import (
    MULTITASK_LOSS_ALPHA,
    MultiTaskLoss,
    masked_multitask_mse_loss,
)
from .multitask_model import MultiTaskPrediction, MultiTaskSurrogate
from .multitask_trainer import (
    MultiTaskTrainingConfig,
    MultiTaskTrainingResult,
    predict_multitask,
    train_multitask_surrogate,
)
from .trainer import (
    SurrogateTrainingConfig,
    SurrogateTrainingResult,
    predict_accuracy,
    train_accuracy_surrogate,
)

__all__ = [
    "AccuracySurrogate",
    "MULTITASK_LOSS_ALPHA",
    "MultiTaskLoss",
    "MultiTaskPrediction",
    "MultiTaskSurrogate",
    "MultiTaskSurrogateDataset",
    "MultiTaskSurrogateObservation",
    "MultiTaskTrainingConfig",
    "MultiTaskTrainingResult",
    "StabilityRecord",
    "SurrogateDataset",
    "SurrogateObservation",
    "SurrogateTrainingConfig",
    "SurrogateTrainingResult",
    "masked_multitask_mse_loss",
    "predict_accuracy",
    "predict_multitask",
    "train_accuracy_surrogate",
    "train_multitask_surrogate",
]
