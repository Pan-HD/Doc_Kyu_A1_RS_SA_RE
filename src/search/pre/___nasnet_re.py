"""End-to-end NASNet adapter for official regularized evolution.

This module connects the generic Algorithm 1 loop to fresh NASNet model
construction, fixed-split CIFAR-10 training, deterministic per-evaluation
seeds, and crash-resilient experiment logs. Imports of PyTorch-facing project
modules are lazy so the orchestration can be unit-tested with toy objects.
"""

from __future__ import annotations

import csv
import gc
import io
import json
import math
import os
import random
from dataclasses import dataclass, is_dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from .regularized_evolution import (
    EVOLUTION_PHASE,
    INITIALIZATION_PHASE,
    EvaluationOutcome,
    EvolutionProgress,
    EvolutionResult,
    regularized_evolution,
)


CSV_FIELDS = (
    "method",
    "search_seed",
    "training_seed",
    "evaluation_index",
    "budget",
    "phase",
    "architecture",
    "parent_architecture",
    "mutation_type",
    "fitness",
    "final_val_accuracy",
    "best_val_accuracy",
    "parameter_count",
    "training_time",
    "population_age_order",
)

REQUIRED_TRAINING_METADATA = (
    "training_seed",
    "final_val_accuracy",
    "best_val_accuracy",
    "parameter_count",
    "training_time_seconds",
)


class TrainingEvaluatorLike(Protocol):
    real_training_runs: int

    def __call__(self, architecture: Any) -> EvaluationOutcome:
        ...


@dataclass(frozen=True)
class NASNetREResult:
    evolution: EvolutionResult[Any]
    real_training_runs: int
    output_dir: Path
    config_path: Path
    evaluations_csv_path: Path
    history_json_path: Path
    run_log_path: Path

    @property
    def best_fitness(self) -> float:
        return self.evolution.best_individual.fitness


def seed_training_event(seed: int) -> None:
    """Seed model initialization before the trainer receives the model."""

    random.seed(seed)

    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _default_model_builder(
    architecture: Any,
    N: int,
    F: int,
    num_classes: int,
):
    from src.nasnet.network import build_nasnet

    return build_nasnet(
        architecture,
        N=N,
        F=F,
        num_classes=num_classes,
    )


def _default_training_config_factory(**values):
    from src.training.nasnet_trainer import TrainingConfig

    return TrainingConfig(**values)


def _default_trainer(
    model,
    train_loader,
    val_loader,
    training_config,
    device,
):
    from src.training.nasnet_trainer import train_nasnet

    return train_nasnet(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device,
    )


def _default_cleanup(_model: Any, device: Any) -> None:
    """Release references and cached CUDA blocks between architectures."""

    gc.collect()
    try:
        import torch

        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        # Cleanup must never hide a successfully completed evaluation.
        pass


class NASNetTrainingEvaluator:
    """Build and train a fresh NASNet for every RE evaluation event."""

    def __init__(
        self,
        *,
        loader_factory: Callable[[int], tuple[Any, Any]],
        training_config_values: Mapping[str, Any],
        training_seed_base: int,
        device: Any,
        N: int = 3,
        F: int = 24,
        num_classes: int = 10,
        model_builder: Callable[[Any, int, int, int], Any]
        | None = None,
        training_config_factory: Callable[..., Any] | None = None,
        trainer_fn: Callable[[Any, Any, Any, Any, Any], Any]
        | None = None,
        seed_fn: Callable[[int], None] = seed_training_event,
        cleanup_fn: Callable[[Any, Any], None] = _default_cleanup,
    ) -> None:
        if N <= 0 or F <= 0 or num_classes <= 0:
            raise ValueError("N, F, and num_classes must be positive")
        if not isinstance(training_seed_base, int):
            raise TypeError("training_seed_base must be an integer")

        self.loader_factory = loader_factory
        self.training_config_values = dict(training_config_values)
        self.training_seed_base = training_seed_base
        self.device = device
        self.N = N
        self.F = F
        self.num_classes = num_classes
        self.model_builder = model_builder or _default_model_builder
        self.training_config_factory = (
            training_config_factory
            or _default_training_config_factory
        )
        self.trainer_fn = trainer_fn or _default_trainer
        self.seed_fn = seed_fn
        self.cleanup_fn = cleanup_fn
        self.real_training_runs = 0
        self.completed_training_seeds: list[int] = []

    def __call__(self, architecture: Any) -> EvaluationOutcome:
        """Preserve the official RE/SA-RE sequential first-seed schedule."""

        training_seed = self.training_seed_base + self.real_training_runs
        return self.evaluate_with_seed(architecture, training_seed)

    def evaluate_with_seed(
        self,
        architecture: Any,
        training_seed: int,
    ) -> EvaluationOutcome:
        """Run one real evaluation with a caller-derived deterministic seed.

        RS-SA-RE uses this narrow extension for replicate-aware first/repeat
        seeds. ``__call__`` still derives exactly the original sequential
        schedule, so RE and SA-RE behavior is unchanged.
        """

        if isinstance(training_seed, bool) or not isinstance(training_seed, int):
            raise TypeError("training_seed must be an integer")
        if training_seed < 0:
            raise ValueError("training_seed must be non-negative")

        # Seeding happens before build_nasnet, so weight initialization is
        # controlled. train_nasnet seeds again before the epoch loop.
        self.seed_fn(training_seed)
        model = self.model_builder(
            architecture,
            self.N,
            self.F,
            self.num_classes,
        )
        train_loader, val_loader = self.loader_factory(training_seed)

        config_values = dict(self.training_config_values)
        config_values["training_seed"] = training_seed
        training_config = self.training_config_factory(**config_values)

        try:
            training_result = self.trainer_fn(
                model,
                train_loader,
                val_loader,
                training_config,
                self.device,
            )
        finally:
            self.cleanup_fn(model, self.device)

        metadata = {
            "training_seed": training_seed,
            "final_val_accuracy": float(
                training_result.final_val_accuracy
            ),
            "best_val_accuracy": float(
                training_result.best_val_accuracy
            ),
            "parameter_count": int(training_result.parameter_count),
            "training_time_seconds": float(
                training_result.training_time_seconds
            ),
        }

        for key in ("final_val_accuracy", "best_val_accuracy"):
            value = metadata[key]
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{key} must be a finite value in [0, 1]")
        if metadata["parameter_count"] <= 0:
            raise ValueError("parameter_count must be positive")
        if (
            not math.isfinite(metadata["training_time_seconds"])
            or metadata["training_time_seconds"] < 0.0
        ):
            raise ValueError("training_time_seconds must be finite and non-negative")

        self.real_training_runs += 1
        self.completed_training_seeds.append(training_seed)

        # Fixed-horizon fitness is final validation accuracy, never best.
        return EvaluationOutcome(
            fitness=metadata["final_val_accuracy"],
            metadata=metadata,
        )


def _architecture_to_dict(architecture: Any) -> dict[str, Any]:
    if hasattr(architecture, "to_dict"):
        value = architecture.to_dict()
    elif is_dataclass(architecture):
        value = asdict(architecture)
    elif isinstance(architecture, Mapping):
        value = dict(architecture)
    else:
        raise TypeError(
            "architecture must provide to_dict(), be a dataclass, or be a mapping"
        )
    if not isinstance(value, Mapping):
        raise TypeError("architecture serialization must produce a mapping")
    return dict(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    os.replace(temporary_path, path)


def _write_evaluations_csv(
    records: list[dict[str, Any]],
    path: Path,
) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for record in records:
        csv_record = dict(record)
        for key in (
            "architecture",
            "parent_architecture",
            "population_age_order",
        ):
            value = csv_record[key]
            csv_record[key] = "" if value is None else _json_dumps(value)
        writer.writerow(csv_record)
    _atomic_write_text(path, stream.getvalue())


def _write_history_json(
    *,
    records: list[dict[str, Any]],
    path: Path,
    method: str,
    search_seed: int,
    population_size: int,
    tournament_size: int,
    budget: int,
    final_population_order: list[int] | None,
    completed: bool,
) -> None:
    payload = {
        "method": method,
        "search_seed": search_seed,
        "population_size": population_size,
        "tournament_size": tournament_size,
        "budget": budget,
        "real_training_runs": len(records),
        "completed": completed,
        "final_population_order": final_population_order,
        "evaluations": records,
    }
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append_run_log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{_timestamp()} {message}\n")


def _prepare_output_files(
    output_dir: Path,
    overwrite: bool,
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        output_dir / "config.yaml",
        output_dir / "evaluations.csv",
        output_dir / "history.json",
        output_dir / "run.log",
    )
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"experiment output already exists ({names}); choose a new "
            "output_dir or set experiment.overwrite=true"
        )
    return paths


def run_nasnet_re(
    *,
    evaluator: TrainingEvaluatorLike,
    output_dir: str | Path,
    config_text: str,
    method: str,
    search_seed: int,
    population_size: int,
    tournament_size: int,
    budget: int,
    overwrite: bool = False,
    random_architecture_fn: Callable[[random.Random], Any] | None = None,
    mutate_fn: Callable[[Any, random.Random], Any] | None = None,
    print_fn: Callable[[str], None] = print,
) -> NASNetREResult:
    """Run real NASNet RE and incrementally persist all required logs."""

    if method.upper() != "RE":
        raise ValueError("Part F runner requires experiment.method=RE")
    method = "RE"
    if not isinstance(search_seed, int):
        raise TypeError("search_seed must be an integer")

    if random_architecture_fn is None:
        from src.nasnet.genotype import random_architecture

        random_architecture_fn = random_architecture
    if mutate_fn is None:
        from src.nasnet.mutation import mutate_architecture

        mutate_fn = mutate_architecture

    output_dir = Path(output_dir)
    (
        config_path,
        evaluations_csv_path,
        history_json_path,
        run_log_path,
    ) = _prepare_output_files(output_dir, overwrite)

    _atomic_write_text(
        config_path,
        config_text.rstrip() + "\n",
    )
    _atomic_write_text(run_log_path, "")
    records: list[dict[str, Any]] = []
    individuals_by_id: dict[int, Any] = {}

    start_message = (
        f"START method={method} search_seed={search_seed} "
        f"population_size={population_size} "
        f"tournament_size={tournament_size} budget={budget}"
    )
    _append_run_log(run_log_path, start_message)
    print_fn(start_message)

    def on_progress(progress: EvolutionProgress[Any]) -> None:
        if progress.history_length != evaluator.real_training_runs:
            raise RuntimeError(
                "history length does not match real training runs"
            )
        if evaluator.real_training_runs > budget:
            raise RuntimeError("real training runs exceeded budget")
        if (
            progress.phase == EVOLUTION_PHASE
            and len(progress.population) != population_size
        ):
            raise RuntimeError("evolution population size changed")
        if (
            progress.phase == INITIALIZATION_PHASE
            and progress.history_length == population_size
            and len(progress.population) != population_size
        ):
            raise RuntimeError("initial population size is inconsistent")

        individual = progress.individual
        missing_metadata = [
            key
            for key in REQUIRED_TRAINING_METADATA
            if key not in individual.metadata
        ]
        if missing_metadata:
            raise RuntimeError(
                "training evaluator omitted metadata: "
                + ", ".join(missing_metadata)
            )

        parent = (
            individuals_by_id.get(individual.parent_evaluation_id)
            if individual.parent_evaluation_id is not None
            else None
        )
        if individual.parent_evaluation_id is not None and parent is None:
            raise RuntimeError("parent evaluation is missing from history")

        evaluation_index = individual.evaluation_id + 1
        population_order = [
            member.evaluation_id + 1
            for member in progress.population
        ]
        population_ages = [
            individual.evaluation_id - member.evaluation_id
            for member in progress.population
        ]

        record = {
            "method": method,
            "search_seed": search_seed,
            "training_seed": int(individual.metadata["training_seed"]),
            "evaluation_index": evaluation_index,
            "budget": budget,
            "phase": progress.phase,
            "architecture": _architecture_to_dict(
                individual.architecture
            ),
            "parent_architecture": (
                _architecture_to_dict(parent.architecture)
                if parent is not None
                else None
            ),
            "mutation_type": individual.mutation_type,
            "fitness": float(individual.fitness),
            "final_val_accuracy": float(
                individual.metadata["final_val_accuracy"]
            ),
            "best_val_accuracy": float(
                individual.metadata["best_val_accuracy"]
            ),
            "parameter_count": int(
                individual.metadata["parameter_count"]
            ),
            "training_time": float(
                individual.metadata["training_time_seconds"]
            ),
            "population_age_order": {
                "oldest_to_youngest_evaluation_indices": population_order,
                "ages_in_evaluations": population_ages,
            },
        }
        if record["fitness"] != record["final_val_accuracy"]:
            raise RuntimeError("RE fitness must equal final_val_accuracy")

        records.append(record)
        individuals_by_id[individual.evaluation_id] = individual
        _write_evaluations_csv(records, evaluations_csv_path)
        _write_history_json(
            records=records,
            path=history_json_path,
            method=method,
            search_seed=search_seed,
            population_size=population_size,
            tournament_size=tournament_size,
            budget=budget,
            final_population_order=population_order,
            completed=False,
        )

        progress_message = (
            f"[{evaluation_index}/{budget}] {progress.phase} "
            f"mutation={individual.mutation_type} "
            f"fitness={individual.fitness:.6f} "
            f"population={population_order}"
        )
        _append_run_log(run_log_path, progress_message)
        print_fn(progress_message)

    rng = random.Random(search_seed)
    try:
        evolution = regularized_evolution(
            random_architecture_fn=random_architecture_fn,
            mutate_fn=mutate_fn,
            evaluate_fn=evaluator,
            population_size=population_size,
            tournament_size=tournament_size,
            budget=budget,
            rng=rng,
            progress_fn=on_progress,
        )
    except Exception as error:
        _append_run_log(
            run_log_path,
            f"FAILED {type(error).__name__}: {error}",
        )
        raise

    if len(evolution.population) != population_size:
        raise RuntimeError("final population size does not match config")
    if len(evolution.history) != evaluator.real_training_runs:
        raise RuntimeError("final history/training-run count mismatch")
    if evaluator.real_training_runs != budget:
        raise RuntimeError("real training runs did not consume exact budget")
    if len(records) != budget:
        raise RuntimeError("evaluation log did not consume exact budget")

    final_population_order = [
        member.evaluation_id + 1
        for member in evolution.population
    ]
    _write_history_json(
        records=records,
        path=history_json_path,
        method=method,
        search_seed=search_seed,
        population_size=population_size,
        tournament_size=tournament_size,
        budget=budget,
        final_population_order=final_population_order,
        completed=True,
    )

    completed_message = (
        f"RE completed real_training_runs={evaluator.real_training_runs} "
        f"best_fitness={evolution.best_individual.fitness:.6f}"
    )
    _append_run_log(run_log_path, completed_message)
    print_fn(completed_message)

    return NASNetREResult(
        evolution=evolution,
        real_training_runs=evaluator.real_training_runs,
        output_dir=output_dir,
        config_path=config_path,
        evaluations_csv_path=evaluations_csv_path,
        history_json_path=history_json_path,
        run_log_path=run_log_path,
    )


__all__ = [
    "CSV_FIELDS",
    "NASNetREResult",
    "NASNetTrainingEvaluator",
    "REQUIRED_TRAINING_METADATA",
    "run_nasnet_re",
    "seed_training_event",
]
