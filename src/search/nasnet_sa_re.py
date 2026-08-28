"""NASNet adapter, logging, and surrogate controller for SA-RE."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import random
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from src.surrogate import (
    SurrogateDataset,
    SurrogateTrainingConfig,
    predict_accuracy,
    train_accuracy_surrogate,
)

from .nasnet_re import REQUIRED_TRAINING_METADATA
from .regularized_evolution import EVOLUTION_PHASE, INITIALIZATION_PHASE
from .surrogate_assisted_evolution import (
    SAEvolutionProgress,
    SAEvolutionResult,
    SurrogateScoreResult,
    surrogate_assisted_evolution,
)


EVALUATION_CSV_FIELDS = (
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
    "observed_final_val_accuracy",
    "best_val_accuracy",
    "parameter_count",
    "training_time",
    "population_age_order",
    "predicted_mu_before_training",
    "surrogate_training_size",
    "selected_candidate_index",
    "surrogate_training_mse",
)

CANDIDATE_CSV_FIELDS = (
    "method",
    "search_seed",
    "evaluation_index",
    "parent_evaluation_index",
    "parent_architecture",
    "candidate_index",
    "candidate_architecture",
    "mutation_type",
    "predicted_mu",
    "selected",
    "surrogate_training_size",
    "surrogate_training_mse",
)


class TrainingEvaluatorLike(Protocol):
    real_training_runs: int

    def __call__(self, architecture: Any):
        ...


@dataclass(frozen=True)
class NASNetSAREResult:
    evolution: SAEvolutionResult[Any]
    real_training_runs: int
    output_dir: Path
    config_path: Path
    evaluations_csv_path: Path
    candidate_predictions_csv_path: Path
    history_json_path: Path
    run_log_path: Path

    @property
    def best_fitness(self) -> float:
        return self.evolution.best_individual.fitness


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
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    os.replace(temporary_path, path)


def _write_csv(
    *,
    records: list[dict[str, Any]],
    path: Path,
    fieldnames: tuple[str, ...],
    json_fields: tuple[str, ...],
) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for record in records:
        csv_record = dict(record)
        for key in json_fields:
            value = csv_record[key]
            csv_record[key] = "" if value is None else _json_dumps(value)
        writer.writerow(csv_record)
    _atomic_write_text(path, stream.getvalue())


def _write_history_json(
    *,
    evaluation_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    path: Path,
    search_seed: int,
    population_size: int,
    tournament_size: int,
    budget: int,
    candidate_count: int,
    surrogate_seed: int,
    final_population_order: list[int] | None,
    completed: bool,
) -> None:
    payload = {
        "method": "SA-RE",
        "search_seed": search_seed,
        "surrogate_seed": surrogate_seed,
        "population_size": population_size,
        "tournament_size": tournament_size,
        "budget": budget,
        "candidate_count": candidate_count,
        "real_training_runs": len(evaluation_records),
        "candidate_predictions": len(candidate_records),
        "completed": completed,
        "final_population_order": final_population_order,
        "evaluations": evaluation_records,
        "candidates": candidate_records,
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
) -> tuple[Path, Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        output_dir / "config.yaml",
        output_dir / "evaluations.csv",
        output_dir / "candidate_predictions.csv",
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


def run_nasnet_sa_re(
    *,
    evaluator: TrainingEvaluatorLike,
    output_dir: str | Path,
    config_text: str,
    method: str,
    search_seed: int,
    population_size: int,
    tournament_size: int,
    budget: int,
    candidate_count: int,
    surrogate_config_values: Mapping[str, Any],
    overwrite: bool = False,
    random_architecture_fn: Callable[[random.Random], Any] | None = None,
    mutate_fn: Callable[[Any, random.Random], Any] | None = None,
    encode_fn: Callable[[Any], Any] | None = None,
    surrogate_score_fn: Callable[
        [SurrogateDataset, Sequence[Any]],
        SurrogateScoreResult,
    ]
    | None = None,
    print_fn: Callable[[str], None] = print,
) -> NASNetSAREResult:
    """Run real NASNet SA-RE and persist evaluation/candidate audit logs."""

    if method.upper() != "SA-RE":
        raise ValueError("SA-RE runner requires experiment.method=SA-RE")
    if not isinstance(search_seed, int):
        raise TypeError("search_seed must be an integer")

    surrogate_values = dict(surrogate_config_values)
    seed_offset = int(surrogate_values.pop("seed_offset", 900000))
    surrogate_seed = seed_offset + search_seed
    surrogate_config = SurrogateTrainingConfig(
        input_dim=int(surrogate_values.pop("input_dim", 280)),
        hidden_dims=tuple(surrogate_values.pop("hidden_dims", (32, 16))),
        optimizer=str(surrogate_values.pop("optimizer", "Adam")),
        learning_rate=float(surrogate_values.pop("learning_rate", 1e-3)),
        weight_decay=float(surrogate_values.pop("weight_decay", 1e-4)),
        steps=int(surrogate_values.pop("steps", 200)),
        seed=surrogate_seed,
    )
    if surrogate_values:
        unknown = ", ".join(sorted(surrogate_values))
        raise ValueError(f"unknown surrogate configuration fields: {unknown}")

    if random_architecture_fn is None:
        from src.nasnet.genotype import random_architecture

        random_architecture_fn = random_architecture
    if mutate_fn is None:
        from src.nasnet.mutation import mutate_architecture

        mutate_fn = mutate_architecture
    if encode_fn is None:
        from src.nasnet.encoding import encode_architecture

        encode_fn = encode_architecture

    if surrogate_score_fn is None:

        def score_candidates(
            dataset: SurrogateDataset,
            encodings: Sequence[Any],
        ) -> SurrogateScoreResult:
            training_result = train_accuracy_surrogate(
                dataset,
                surrogate_config,
            )
            return SurrogateScoreResult(
                scores=predict_accuracy(training_result.model, encodings),
                training_mse=training_result.training_mse,
            )

        active_score_fn = score_candidates
    else:
        active_score_fn = surrogate_score_fn

    output_dir = Path(output_dir)
    (
        config_path,
        evaluations_csv_path,
        candidate_predictions_csv_path,
        history_json_path,
        run_log_path,
    ) = _prepare_output_files(output_dir, overwrite)

    _atomic_write_text(config_path, config_text.rstrip() + "\n")
    _atomic_write_text(run_log_path, "")
    evaluation_records: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    individuals_by_id: dict[int, Any] = {}

    start_message = (
        f"START method=SA-RE search_seed={search_seed} "
        f"surrogate_seed={surrogate_seed} population_size={population_size} "
        f"tournament_size={tournament_size} budget={budget} "
        f"candidate_count={candidate_count}"
    )
    _append_run_log(run_log_path, start_message)
    print_fn(start_message)

    def on_progress(progress: SAEvolutionProgress[Any]) -> None:
        if progress.history_length != evaluator.real_training_runs:
            raise RuntimeError("history length does not match real training runs")
        if evaluator.real_training_runs > budget:
            raise RuntimeError("real training runs exceeded budget")
        if progress.phase == EVOLUTION_PHASE:
            if len(progress.population) != population_size:
                raise RuntimeError("SA-RE population size changed")
            if progress.candidate_batch is None:
                raise RuntimeError("evolution progress omitted candidate batch")
        elif progress.phase == INITIALIZATION_PHASE:
            if progress.candidate_batch is not None:
                raise RuntimeError("initialization must not have candidates")
        else:
            raise RuntimeError(f"unknown SA-RE phase: {progress.phase}")

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
            member.evaluation_id + 1 for member in progress.population
        ]
        population_ages = [
            individual.evaluation_id - member.evaluation_id
            for member in progress.population
        ]

        predicted_mu = individual.metadata.get("predicted_mu_before_training")
        surrogate_training_size = individual.metadata.get(
            "surrogate_training_size"
        )
        selected_candidate_index = individual.metadata.get(
            "selected_candidate_index"
        )
        surrogate_training_mse = individual.metadata.get(
            "surrogate_training_mse"
        )
        final_accuracy = float(individual.metadata["final_val_accuracy"])

        if progress.phase == INITIALIZATION_PHASE:
            if any(
                value is not None
                for value in (
                    predicted_mu,
                    surrogate_training_size,
                    selected_candidate_index,
                    surrogate_training_mse,
                )
            ):
                raise RuntimeError(
                    "initialization evaluation contains surrogate metadata"
                )
        else:
            if any(
                value is None
                for value in (
                    predicted_mu,
                    surrogate_training_size,
                    selected_candidate_index,
                    surrogate_training_mse,
                )
            ):
                raise RuntimeError("evolution evaluation omitted surrogate metadata")
            if not math.isfinite(float(predicted_mu)):
                raise RuntimeError("predicted_mu_before_training is non-finite")
            if int(surrogate_training_size) != evaluation_index - 1:
                raise RuntimeError(
                    "surrogate_training_size must equal prior real evaluations"
                )

        evaluation_record = {
            "method": "SA-RE",
            "search_seed": search_seed,
            "training_seed": int(individual.metadata["training_seed"]),
            "evaluation_index": evaluation_index,
            "budget": budget,
            "phase": progress.phase,
            "architecture": _architecture_to_dict(individual.architecture),
            "parent_architecture": (
                _architecture_to_dict(parent.architecture)
                if parent is not None
                else None
            ),
            "mutation_type": individual.mutation_type,
            "fitness": float(individual.fitness),
            "final_val_accuracy": final_accuracy,
            "observed_final_val_accuracy": final_accuracy,
            "best_val_accuracy": float(
                individual.metadata["best_val_accuracy"]
            ),
            "parameter_count": int(individual.metadata["parameter_count"]),
            "training_time": float(
                individual.metadata["training_time_seconds"]
            ),
            "population_age_order": {
                "oldest_to_youngest_evaluation_indices": population_order,
                "ages_in_evaluations": population_ages,
            },
            "predicted_mu_before_training": predicted_mu,
            "surrogate_training_size": surrogate_training_size,
            "selected_candidate_index": selected_candidate_index,
            "surrogate_training_mse": surrogate_training_mse,
        }
        if evaluation_record["fitness"] != final_accuracy:
            raise RuntimeError("SA-RE fitness must equal final_val_accuracy")
        evaluation_records.append(evaluation_record)

        if progress.candidate_batch is not None:
            batch = progress.candidate_batch
            if batch.evaluation_index != evaluation_index:
                raise RuntimeError("candidate batch/evaluation index mismatch")
            if len(batch.candidates) != candidate_count:
                raise RuntimeError("candidate batch has incorrect size")
            if batch.surrogate_training_size != evaluation_index - 1:
                raise RuntimeError(
                    "candidate batch has incorrect surrogate training size"
                )
            if sum(candidate.selected for candidate in batch.candidates) != 1:
                raise RuntimeError("candidate batch must select exactly one row")
            for candidate in batch.candidates:
                candidate_records.append(
                    {
                        "method": "SA-RE",
                        "search_seed": search_seed,
                        "evaluation_index": evaluation_index,
                        "parent_evaluation_index": (
                            batch.parent_evaluation_id + 1
                        ),
                        "parent_architecture": _architecture_to_dict(
                            parent.architecture
                        ),
                        "candidate_index": candidate.candidate_index,
                        "candidate_architecture": _architecture_to_dict(
                            candidate.architecture
                        ),
                        "mutation_type": candidate.mutation_type,
                        "predicted_mu": candidate.predicted_mu,
                        "selected": candidate.selected,
                        "surrogate_training_size": (
                            batch.surrogate_training_size
                        ),
                        "surrogate_training_mse": (
                            batch.surrogate_training_mse
                        ),
                    }
                )

        individuals_by_id[individual.evaluation_id] = individual
        _write_csv(
            records=evaluation_records,
            path=evaluations_csv_path,
            fieldnames=EVALUATION_CSV_FIELDS,
            json_fields=(
                "architecture",
                "parent_architecture",
                "population_age_order",
            ),
        )
        _write_csv(
            records=candidate_records,
            path=candidate_predictions_csv_path,
            fieldnames=CANDIDATE_CSV_FIELDS,
            json_fields=(
                "parent_architecture",
                "candidate_architecture",
            ),
        )
        _write_history_json(
            evaluation_records=evaluation_records,
            candidate_records=candidate_records,
            path=history_json_path,
            search_seed=search_seed,
            population_size=population_size,
            tournament_size=tournament_size,
            budget=budget,
            candidate_count=candidate_count,
            surrogate_seed=surrogate_seed,
            final_population_order=population_order,
            completed=False,
        )

        progress_message = (
            f"[{evaluation_index}/{budget}] {progress.phase} "
            f"mutation={individual.mutation_type} "
            f"fitness={individual.fitness:.6f}"
        )
        if selected_candidate_index is not None:
            progress_message += (
                f" selected_candidate={selected_candidate_index} "
                f"predicted_mu={float(predicted_mu):.6f}"
            )
        _append_run_log(run_log_path, progress_message)
        print_fn(progress_message)

    surrogate_dataset = SurrogateDataset(input_dim=surrogate_config.input_dim)
    rng = random.Random(search_seed)
    try:
        evolution = surrogate_assisted_evolution(
            random_architecture_fn=random_architecture_fn,
            mutate_fn=mutate_fn,
            evaluate_fn=evaluator,
            encode_fn=encode_fn,
            surrogate_score_fn=active_score_fn,
            surrogate_dataset=surrogate_dataset,
            population_size=population_size,
            tournament_size=tournament_size,
            budget=budget,
            candidate_count=candidate_count,
            rng=rng,
            progress_fn=on_progress,
        )
    except Exception as error:
        _append_run_log(
            run_log_path,
            f"FAILED {type(error).__name__}: {error}",
        )
        raise

    expected_candidate_rows = (budget - population_size) * candidate_count
    if len(evolution.population) != population_size:
        raise RuntimeError("final population size does not match config")
    if len(surrogate_dataset) != budget:
        raise RuntimeError(
            "final surrogate dataset size does not match budget"
        )
    if evaluator.real_training_runs != budget:
        raise RuntimeError("real training runs did not consume exact budget")
    if len(evaluation_records) != budget:
        raise RuntimeError("evaluation log did not consume exact budget")
    if len(candidate_records) != expected_candidate_rows:
        raise RuntimeError("candidate log row count is inconsistent")
    if sum(bool(record["selected"]) for record in candidate_records) != (
        budget - population_size
    ):
        raise RuntimeError("candidate log selected-count is inconsistent")

    final_population_order = [
        member.evaluation_id + 1 for member in evolution.population
    ]
    _write_history_json(
        evaluation_records=evaluation_records,
        candidate_records=candidate_records,
        path=history_json_path,
        search_seed=search_seed,
        population_size=population_size,
        tournament_size=tournament_size,
        budget=budget,
        candidate_count=candidate_count,
        surrogate_seed=surrogate_seed,
        final_population_order=final_population_order,
        completed=True,
    )

    completed_message = (
        f"SA-RE completed real_training_runs={evaluator.real_training_runs} "
        f"candidate_predictions={len(candidate_records)} "
        f"best_fitness={evolution.best_individual.fitness:.6f}"
    )
    _append_run_log(run_log_path, completed_message)
    print_fn(completed_message)

    return NASNetSAREResult(
        evolution=evolution,
        real_training_runs=evaluator.real_training_runs,
        output_dir=output_dir,
        config_path=config_path,
        evaluations_csv_path=evaluations_csv_path,
        candidate_predictions_csv_path=candidate_predictions_csv_path,
        history_json_path=history_json_path,
        run_log_path=run_log_path,
    )


__all__ = [
    "CANDIDATE_CSV_FIELDS",
    "EVALUATION_CSV_FIELDS",
    "NASNetSAREResult",
    "run_nasnet_sa_re",
]
