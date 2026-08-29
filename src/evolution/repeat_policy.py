"""Deterministic, outcome-independent repeat policy for RS-SA-RE.

The scheduler owns only repeat-target selection. It never trains a CNN,
modifies a paired record, writes population state, or calls a surrogate. Its
private RNG is independent of the search and surrogate RNG streams.

Search-stage instability remains a two-seed retraining-instability proxy. A
record may receive at most one scheduled repeat; paired records are therefore
excluded from every eligible target pool.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from numbers import Integral
from typing import Iterable

from ..surrogate.multitask_dataset import PairedEvaluationRecord


# The existing RE/SA-RE first-training schedule is base + evaluation offset.
# Reserving a deterministic namespace per replicate preserves that schedule for
# replicate_id=1 while preventing first/repeat collisions for practical search
# budgets (up to one million base evaluations per replicate namespace).
REPLICATE_SEED_STRIDE = 1_000_000


class NoEligibleRepeatRecordError(RuntimeError):
    """Raised when no evaluated, unpaired record can receive a repeat."""


def _validated_int(value: int, *, field_name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")
    return value


def derive_training_seed(
    *,
    training_seed_base: int,
    base_evaluation_index: int,
    replicate_id: int,
) -> int:
    """Return a deterministic seed for one evaluation replicate.

    Evaluation indices are one-based. ``replicate_id=1`` exactly preserves the
    existing first-training schedule::

        training_seed_base + base_evaluation_index - 1

    ``replicate_id=2`` uses the next non-overlapping seed namespace and is the
    scheduled second-seed retraining used by the search-stage proxy.
    """

    base = _validated_int(
        training_seed_base,
        field_name="training_seed_base",
        minimum=0,
    )
    evaluation_index = _validated_int(
        base_evaluation_index,
        field_name="base_evaluation_index",
        minimum=1,
    )
    replicate = _validated_int(
        replicate_id,
        field_name="replicate_id",
        minimum=1,
    )
    if evaluation_index > REPLICATE_SEED_STRIDE:
        raise ValueError(
            "base_evaluation_index exceeds the collision-free replicate namespace"
        )

    return (
        base
        + evaluation_index
        - 1
        + (replicate - 1) * REPLICATE_SEED_STRIDE
    )


@dataclass(frozen=True, slots=True)
class RepeatPolicyConfig:
    """Frozen provisional repeat policy for the August 29 implementation."""

    initial_population_size: int = 20
    warmup_pairs: int = 4
    repeat_interval: int = 4
    repeat_rate_beta: float = 0.25

    def __post_init__(self) -> None:
        population_size = _validated_int(
            self.initial_population_size,
            field_name="initial_population_size",
            minimum=1,
        )
        warmup_pairs = _validated_int(
            self.warmup_pairs,
            field_name="warmup_pairs",
            minimum=0,
        )
        repeat_interval = _validated_int(
            self.repeat_interval,
            field_name="repeat_interval",
            minimum=1,
        )
        if warmup_pairs > population_size:
            raise ValueError(
                "warmup_pairs cannot exceed initial_population_size"
            )

        beta = float(self.repeat_rate_beta)
        if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
            raise ValueError("repeat_rate_beta must be finite and in [0, 1]")
        expected_beta = 1.0 / repeat_interval
        if not math.isclose(beta, expected_beta, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "repeat_rate_beta must equal 1 / repeat_interval for the "
                "fixed periodic policy"
            )


class RepeatScheduler:
    """Select repeat targets using a private, deterministic RNG stream."""

    def __init__(
        self,
        *,
        repeat_seed: int,
        config: RepeatPolicyConfig | None = None,
    ) -> None:
        checked_seed = _validated_int(
            repeat_seed,
            field_name="repeat_seed",
            minimum=0,
        )
        self.config = config if config is not None else RepeatPolicyConfig()
        if not isinstance(self.config, RepeatPolicyConfig):
            raise TypeError("config must be a RepeatPolicyConfig")
        self.repeat_seed = checked_seed
        self._repeat_rng = random.Random(checked_seed)

    @staticmethod
    def _canonical_unpaired_records(
        records: Iterable[PairedEvaluationRecord],
    ) -> tuple[PairedEvaluationRecord, ...]:
        materialized = tuple(records)
        if any(not isinstance(record, PairedEvaluationRecord) for record in materialized):
            raise TypeError("records must contain only PairedEvaluationRecord values")

        indices = [record.base_evaluation_index for record in materialized]
        if len(indices) != len(set(indices)):
            raise ValueError("records contain duplicate base_evaluation_index values")

        return tuple(
            sorted(
                (record for record in materialized if not record.has_pair),
                key=lambda record: record.base_evaluation_index,
            )
        )

    def select_warmup(
        self,
        records: Iterable[PairedEvaluationRecord],
    ) -> tuple[PairedEvaluationRecord, ...]:
        """Uniformly sample warm-up targets without replacement."""

        eligible = self._canonical_unpaired_records(records)
        required = self.config.warmup_pairs
        if len(eligible) < required:
            raise NoEligibleRepeatRecordError(
                f"warm-up requires {required} unpaired records; got {len(eligible)}"
            )
        if required == 0:
            return ()
        return tuple(self._repeat_rng.sample(eligible, required))

    def should_schedule(
        self,
        *,
        completed_first_evaluations_after_warmup: int,
    ) -> bool:
        """Return True at 4, 8, 12, ... new first evaluations."""

        completed = _validated_int(
            completed_first_evaluations_after_warmup,
            field_name="completed_first_evaluations_after_warmup",
            minimum=0,
        )
        return completed > 0 and completed % self.config.repeat_interval == 0

    def select(
        self,
        records: Iterable[PairedEvaluationRecord],
    ) -> PairedEvaluationRecord:
        """Uniformly select one currently evaluated, unpaired record."""

        eligible = self._canonical_unpaired_records(records)
        if not eligible:
            raise NoEligibleRepeatRecordError(
                "no evaluated, unpaired record is available for repeat"
            )
        return self._repeat_rng.choice(eligible)


__all__ = [
    "NoEligibleRepeatRecordError",
    "REPLICATE_SEED_STRIDE",
    "RepeatPolicyConfig",
    "RepeatScheduler",
    "derive_training_seed",
]
