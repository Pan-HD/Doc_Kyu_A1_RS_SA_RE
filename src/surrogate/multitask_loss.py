"""Masked multi-task objective for the RS-SA-RE surrogate."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .multitask_model import MultiTaskPrediction


MULTITASK_LOSS_ALPHA = 1.0


@dataclass(frozen=True)
class MultiTaskLoss:
    """Total loss and its auditable task-specific components."""

    total_loss: torch.Tensor
    mean_loss: torch.Tensor
    instability_loss: torch.Tensor
    paired_count: int


def _validate_target_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    expected_shape: torch.Size,
    expected_device: torch.device,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {tuple(expected_shape)}; "
            f"got {tuple(tensor.shape)}"
        )
    if tensor.device != expected_device:
        raise ValueError(f"{name} must be on device {expected_device}")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} contains a non-finite value")


def masked_multitask_mse_loss(
    prediction: MultiTaskPrediction,
    mean_targets: torch.Tensor,
    instability_targets: torch.Tensor,
    stability_mask: torch.Tensor,
) -> MultiTaskLoss:
    """Compute ``MSE(mu) + MSE(d on paired records)`` with alpha fixed at 1.

    Mean supervision always uses every record. Instability supervision uses
    only records whose boolean mask is True. With zero paired records, the
    instability component is an autograd-connected zero rather than the mean of
    an empty tensor, so the result is finite and head gradients are explicit
    zeros.
    """

    if not isinstance(prediction, MultiTaskPrediction):
        raise TypeError("prediction must be a MultiTaskPrediction")
    predicted_mean = prediction.predicted_mean
    predicted_instability = prediction.predicted_instability
    if predicted_mean.ndim != 1:
        raise ValueError("multi-task predictions must have shape [batch]")
    if predicted_mean.device != predicted_instability.device:
        raise ValueError("multi-task predictions must be on the same device")
    if not bool(torch.isfinite(predicted_mean).all().item()):
        raise ValueError("predicted_mean contains a non-finite value")
    if not bool(torch.isfinite(predicted_instability).all().item()):
        raise ValueError("predicted_instability contains a non-finite value")
    if bool((predicted_instability < 0).any().item()):
        raise ValueError("predicted_instability must be non-negative")

    expected_shape = predicted_mean.shape
    expected_device = predicted_mean.device
    _validate_target_tensor(
        mean_targets,
        name="mean_targets",
        expected_shape=expected_shape,
        expected_device=expected_device,
    )
    _validate_target_tensor(
        instability_targets,
        name="instability_targets",
        expected_shape=expected_shape,
        expected_device=expected_device,
    )
    if not isinstance(stability_mask, torch.Tensor):
        raise TypeError("stability_mask must be a torch.Tensor")
    if stability_mask.shape != expected_shape:
        raise ValueError(
            f"stability_mask must have shape {tuple(expected_shape)}; "
            f"got {tuple(stability_mask.shape)}"
        )
    if stability_mask.device != expected_device:
        raise ValueError(f"stability_mask must be on device {expected_device}")
    if stability_mask.dtype != torch.bool:
        raise TypeError("stability_mask must have dtype torch.bool")
    if bool(((mean_targets < 0.0) | (mean_targets > 1.0)).any().item()):
        raise ValueError("mean_targets must be in [0, 1]")
    if bool((instability_targets < 0.0).any().item()):
        raise ValueError("instability_targets must be non-negative")

    mean_loss = F.mse_loss(predicted_mean, mean_targets)
    paired_count = int(stability_mask.sum().item())
    if paired_count:
        instability_loss = F.mse_loss(
            predicted_instability[stability_mask],
            instability_targets[stability_mask],
        )
    else:
        instability_loss = predicted_instability.sum() * 0.0

    total_loss = mean_loss + MULTITASK_LOSS_ALPHA * instability_loss
    if not bool(torch.isfinite(total_loss).item()):
        raise FloatingPointError("multi-task loss is non-finite")

    return MultiTaskLoss(
        total_loss=total_loss,
        mean_loss=mean_loss,
        instability_loss=instability_loss,
        paired_count=paired_count,
    )


__all__ = [
    "MULTITASK_LOSS_ALPHA",
    "MultiTaskLoss",
    "masked_multitask_mse_loss",
]
