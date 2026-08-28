"""Non-runnable scaffolding for future robustness-sensitive SA-RE.

This module deliberately does not implement a search loop, repeat scheduling,
budget allocation, a multi-task loss, or a lambda value. Those decisions must
be made explicitly before RS-SA-RE can become an executable method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.surrogate.multitask_dataset import (
    MultiTaskSurrogateDataset,
    StabilityRecord,
)
from src.surrogate.multitask_model import MultiTaskSurrogate


class RepeatPolicy(Protocol):
    """Interface only; no repeat policy is selected by this scaffold."""

    def should_repeat(
        self,
        *,
        record: StabilityRecord,
        remaining_real_budget: int,
    ) -> bool:
        """Return whether a second real training should be requested."""


class RSSARENotConfiguredError(RuntimeError):
    """Raised when unfinished scaffolding is treated as a runnable method."""


@dataclass(frozen=True)
class RSSAREScaffold:
    """Wire multi-task labels/model without selecting experimental policy."""

    surrogate: MultiTaskSurrogate
    stability_dataset: MultiTaskSurrogateDataset
    repeat_policy: RepeatPolicy | None = None

    @property
    def repeat_policy_configured(self) -> bool:
        return self.repeat_policy is not None

    def require_repeat_policy(self) -> RepeatPolicy:
        if self.repeat_policy is None:
            raise RSSARENotConfiguredError(
                "RS-SA-RE repeat policy is intentionally unresolved"
            )
        return self.repeat_policy

    def run(self) -> None:
        raise RSSARENotConfiguredError(
            "RS-SA-RE search is not implemented by the Part F scaffold"
        )


__all__ = [
    "RSSARENotConfiguredError",
    "RSSAREScaffold",
    "RepeatPolicy",
]
