"""NASNet adapter and audit logging for repeat-stability-aware RE."""

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

from src.evolution.repeat_policy import RepeatPolicyConfig
from src.surrogate import (
    MultiTaskSurrogateDataset,
    MultiTaskTrainingConfig,
    MultiTaskTrainingResult,
    predict_multitask,
    train_multitask_surrogate,
)
from src.surrogate.multitask_dataset import PairedEvaluationRecord

from .nasnet_re import REQUIRED_TRAINING_METADATA
from .rs_sa_re_evolution import (
    RSCandidateBatch,
    RSSAREvolutionProgress,
    RSSAREvolutionResult,
    repeat_stability_assisted_evolution,
)


EVALUATION_CSV_FIELDS = (
    "method",
    "search_seed",
    "budget_index",
    "budget",
    "event_type",
    "phase",
    "base_evaluation_index",
    "training_seed",
    "architecture",
    "parent_architecture",
    "mutation_type",
    "population_inserted",
    "fitness",
    "final_val_accuracy",
    "observed_final_val_accuracy",
    "best_val_accuracy",
    "parameter_count",
    "training_time",
    "population_age_order",
    "predicted_mu_before_training",
    "predicted_d_before_training",
    "lambda",
    "score",
    "selected_candidate_index",
    "surrogate_training_size",
    "paired_label_count",
    "surrogate_training_loss",
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
    "predicted_d",
    "lambda",
    "score",
    "selected",
    "surrogate_training_size",
    "paired_label_count",
    "surrogate_training_loss",
    "mean_training_mse",
    "instability_training_mse",
)

REPEAT_CSV_FIELDS = (
    "method",
    "search_seed",
    "base_evaluation_index",
    "architecture",
    "seed_1",
    "accuracy_1",
    "seed_2",
    "accuracy_2",
    "mean_target",
    "instability_target",
    "budget_index",
)


class ExplicitSeedTrainingEvaluatorLike(Protocol):
    real_training_runs: int

    def evaluate_with_seed(self, architecture: Any, training_seed: int):
        ...


@dataclass(frozen=True)
class NASNetRSSAREResult:
    evolution: RSSAREvolutionResult[Any, MultiTaskTrainingResult]
    real_training_runs: int
    output_dir: Path
    config_path: Path
    evaluations_csv_path: Path
    candidate_predictions_csv_path: Path
    repeat_evaluations_csv_path: Path
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
        row = dict(record)
        for key in json_fields:
            value = row[key]
            row[key] = "" if value is None else _json_dumps(value)
        writer.writerow(row)
    _atomic_write_text(path, stream.getvalue())


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append_run_log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{_timestamp()} {message}\n")


def _prepare_output_files(
    output_dir: Path,
    overwrite: bool,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        output_dir / "config.yaml",
        output_dir / "evaluations.csv",
        output_dir / "candidate_predictions.csv",
        output_dir / "repeat_evaluations.csv",
        output_dir / "history.json",
        output_dir / "run.log",
    )
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"experiment output already exists ({names}); choose a new "
            "output_dir or enable overwrite"
        )
    return paths


def _write_history_json(
    *,
    path: Path,
    search_seed: int,
    repeat_seed: int,
    surrogate_seed: int,
    population_size: int,
    tournament_size: int,
    budget: int,
    candidate_count: int,
    stability_penalty_lambda: float,
    evaluation_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    repeat_records: list[dict[str, Any]],
    final_population_order: list[int] | None,
    completed: bool,
) -> None:
    payload = {
        "method": "RS-SA-RE",
        "search_seed": search_seed,
        "repeat_seed": repeat_seed,
        "surrogate_seed": surrogate_seed,
        "population_size": population_size,
        "tournament_size": tournament_size,
        "budget": budget,
        "candidate_count": candidate_count,
        "stability_penalty_lambda": stability_penalty_lambda,
        "real_training_runs": len(evaluation_records),
        "first_evaluations": sum(
            row["event_type"] == "first_evaluation"
            for row in evaluation_records
        ),
        "repeat_evaluations": len(repeat_records),
        "completed": completed,
        "final_population_order": final_population_order,
        "evaluations": evaluation_records,
        "candidates": candidate_records,
        "repeats": repeat_records,
    }
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def run_nasnet_rs_sa_re(
    *,
    evaluator: ExplicitSeedTrainingEvaluatorLike,
    output_dir: str | Path,
    config_text: str,
    method: str,
    search_seed: int,
    repeat_seed: int,
    training_seed_base: int,
    population_size: int,
    tournament_size: int,
    budget: int,
    candidate_count: int,
    stability_penalty_lambda: float,
    surrogate_config_values: Mapping[str, Any],
    repeat_policy_config: RepeatPolicyConfig | None = None,
    overwrite: bool = False,
    random_architecture_fn: Callable[[random.Random], Any] | None = None,
    mutate_fn: Callable[[Any, random.Random], Any] | None = None,
    encode_fn: Callable[[Any], Any] | None = None,
    fit_surrogate_fn: Callable[
        [Sequence[PairedEvaluationRecord], Sequence[Any]],
        MultiTaskTrainingResult,
    ]
    | None = None,
    predict_surrogate_fn: Callable[[Any, Sequence[Any]], Any] | None = None,
    print_fn: Callable[[str], None] = print,
) -> NASNetRSSAREResult:
    """Run NASNet RS-SA-RE and persist separate event/candidate/repeat logs."""

    if method.upper() != "RS-SA-RE":
        raise ValueError("RS-SA-RE runner requires experiment.method=RS-SA-RE")
    if not isinstance(search_seed, int):
        raise TypeError("search_seed must be an integer")
    if not isinstance(repeat_seed, int):
        raise TypeError("repeat_seed must be an integer")
    if evaluator.real_training_runs != 0:
        raise ValueError("evaluator must not contain prior real training runs")

    surrogate_values = dict(surrogate_config_values)
    seed_offset = int(surrogate_values.pop("seed_offset", 900000))
    surrogate_seed = seed_offset + search_seed
    surrogate_config = MultiTaskTrainingConfig(
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

    if fit_surrogate_fn is None:

        def fit_records(
            records: Sequence[PairedEvaluationRecord],
            encodings: Sequence[Any],
        ) -> MultiTaskTrainingResult:
            if len(records) != len(encodings):
                raise ValueError("record/encoding counts differ")
            dataset = MultiTaskSurrogateDataset(
                input_dim=surrogate_config.input_dim
            )
            for record, encoding in zip(records, encodings, strict=True):
                dataset.add_paired_evaluation(
                    record=record,
                    encoding=encoding,
                )
            return train_multitask_surrogate(dataset, surrogate_config)

        active_fit_fn = fit_records
    else:
        active_fit_fn = fit_surrogate_fn
    active_predict_fn = predict_surrogate_fn or predict_multitask

    output_dir = Path(output_dir)
    (
        config_path,
        evaluations_csv_path,
        candidate_predictions_csv_path,
        repeat_evaluations_csv_path,
        history_json_path,
        run_log_path,
    ) = _prepare_output_files(output_dir, overwrite)
    _atomic_write_text(config_path, config_text.rstrip() + "\n")
    _atomic_write_text(run_log_path, "")

    evaluation_records: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    repeat_records: list[dict[str, Any]] = []
    individuals_by_id: dict[int, Any] = {}

    start_message = (
        f"START method=RS-SA-RE search_seed={search_seed} "
        f"repeat_seed={repeat_seed} surrogate_seed={surrogate_seed} "
        f"population_size={population_size} tournament_size={tournament_size} "
        f"budget={budget} candidate_count={candidate_count} "
        f"lambda={float(stability_penalty_lambda)}"
    )
    _append_run_log(run_log_path, start_message)
    print_fn(start_message)

    def persist(
        *,
        population_order: list[int],
        completed: bool,
    ) -> None:
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
        _write_csv(
            records=repeat_records,
            path=repeat_evaluations_csv_path,
            fieldnames=REPEAT_CSV_FIELDS,
            json_fields=("architecture",),
        )
        _write_history_json(
            path=history_json_path,
            search_seed=search_seed,
            repeat_seed=repeat_seed,
            surrogate_seed=surrogate_seed,
            population_size=population_size,
            tournament_size=tournament_size,
            budget=budget,
            candidate_count=candidate_count,
            stability_penalty_lambda=float(stability_penalty_lambda),
            evaluation_records=evaluation_records,
            candidate_records=candidate_records,
            repeat_records=repeat_records,
            final_population_order=population_order,
            completed=completed,
        )

    def on_progress(progress: RSSAREvolutionProgress[Any]) -> None:
        if progress.real_training_runs != evaluator.real_training_runs:
            raise RuntimeError("engine/evaluator real-training counts differ")
        if progress.real_training_runs > budget:
            raise RuntimeError("real-training budget was exceeded")

        event = progress.budget_event
        population_order = [
            member.evaluation_id + 1 for member in progress.population
        ]
        newest_first_id = progress.history_length - 1
        population_ages = [
            newest_first_id - member.evaluation_id
            for member in progress.population
        ]

        if progress.event_type == "first_evaluation":
            individual = progress.individual
            if individual is None or progress.repeat_evaluation is not None:
                raise RuntimeError("first event has inconsistent progress payload")
            metadata = individual.metadata
            parent = (
                individuals_by_id.get(individual.parent_evaluation_id)
                if individual.parent_evaluation_id is not None
                else None
            )
            if individual.parent_evaluation_id is not None and parent is None:
                raise RuntimeError("parent evaluation is missing from history")
            population_inserted = True
            fitness: float | None = float(individual.fitness)
            parent_architecture = (
                _architecture_to_dict(parent.architecture)
                if parent is not None
                else None
            )
            mutation_type = individual.mutation_type
            candidate_batch = progress.candidate_batch
            individuals_by_id[individual.evaluation_id] = individual
        elif progress.event_type == "repeat_evaluation":
            repeat_result = progress.repeat_evaluation
            if (
                repeat_result is None
                or progress.individual is not None
                or progress.candidate_batch is not None
            ):
                raise RuntimeError("repeat event has inconsistent progress payload")
            metadata = repeat_result.metadata
            population_inserted = False
            fitness = None
            parent_architecture = None
            mutation_type = "scheduled_repeat"
            candidate_batch = None
            record = repeat_result.record
            repeat_records.append(
                {
                    "method": "RS-SA-RE",
                    "search_seed": search_seed,
                    "base_evaluation_index": record.base_evaluation_index,
                    "architecture": _architecture_to_dict(record.architecture),
                    "seed_1": record.seed_1,
                    "accuracy_1": record.accuracy_1,
                    "seed_2": record.seed_2,
                    "accuracy_2": record.accuracy_2,
                    "mean_target": record.mean_target,
                    "instability_target": record.instability_target,
                    "budget_index": event.budget_index,
                }
            )
        else:
            raise RuntimeError(f"unknown event_type: {progress.event_type}")

        missing_metadata = [
            name for name in REQUIRED_TRAINING_METADATA if name not in metadata
        ]
        if missing_metadata:
            raise RuntimeError(
                "training evaluator omitted metadata: "
                + ", ".join(missing_metadata)
            )
        if int(metadata["training_seed"]) != event.training_seed:
            raise RuntimeError("logged training seed does not match budget event")

        selected = (
            candidate_batch.candidates[candidate_batch.selected_candidate_index]
            if candidate_batch is not None
            else None
        )
        evaluation_records.append(
            {
                "method": "RS-SA-RE",
                "search_seed": search_seed,
                "budget_index": event.budget_index,
                "budget": budget,
                "event_type": progress.event_type,
                "phase": progress.phase,
                "base_evaluation_index": event.base_evaluation_index,
                "training_seed": event.training_seed,
                "architecture": _architecture_to_dict(event.architecture),
                "parent_architecture": parent_architecture,
                "mutation_type": mutation_type,
                "population_inserted": population_inserted,
                "fitness": fitness,
                "final_val_accuracy": float(metadata["final_val_accuracy"]),
                "observed_final_val_accuracy": event.accuracy,
                "best_val_accuracy": float(metadata["best_val_accuracy"]),
                "parameter_count": int(metadata["parameter_count"]),
                "training_time": float(metadata["training_time_seconds"]),
                "population_age_order": {
                    "oldest_to_youngest_evaluation_indices": population_order,
                    "ages_in_first_evaluations": population_ages,
                },
                "predicted_mu_before_training": (
                    selected.predicted_mu if selected is not None else None
                ),
                "predicted_d_before_training": (
                    selected.predicted_d if selected is not None else None
                ),
                "lambda": (
                    selected.stability_penalty_lambda
                    if selected is not None
                    else None
                ),
                "score": selected.score if selected is not None else None,
                "selected_candidate_index": (
                    selected.candidate_index if selected is not None else None
                ),
                "surrogate_training_size": (
                    candidate_batch.surrogate_training_size
                    if candidate_batch is not None
                    else None
                ),
                "paired_label_count": (
                    candidate_batch.paired_label_count
                    if candidate_batch is not None
                    else None
                ),
                "surrogate_training_loss": (
                    candidate_batch.surrogate_training_loss
                    if candidate_batch is not None
                    else None
                ),
            }
        )

        if candidate_batch is not None:
            parent = individuals_by_id[candidate_batch.parent_evaluation_id]
            for candidate in candidate_batch.candidates:
                candidate_records.append(
                    {
                        "method": "RS-SA-RE",
                        "search_seed": search_seed,
                        "evaluation_index": candidate_batch.evaluation_index,
                        "parent_evaluation_index": (
                            candidate_batch.parent_evaluation_id + 1
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
                        "predicted_d": candidate.predicted_d,
                        "lambda": candidate.stability_penalty_lambda,
                        "score": candidate.score,
                        "selected": candidate.selected,
                        "surrogate_training_size": (
                            candidate_batch.surrogate_training_size
                        ),
                        "paired_label_count": candidate_batch.paired_label_count,
                        "surrogate_training_loss": (
                            candidate_batch.surrogate_training_loss
                        ),
                        "mean_training_mse": candidate_batch.mean_training_mse,
                        "instability_training_mse": (
                            candidate_batch.instability_training_mse
                        ),
                    }
                )

        persist(population_order=population_order, completed=False)
        message = (
            f"[{event.budget_index}/{budget}] {progress.phase} "
            f"event={progress.event_type} base_eval={event.base_evaluation_index} "
            f"accuracy={event.accuracy:.6f} inserted={population_inserted}"
        )
        _append_run_log(run_log_path, message)
        print_fn(message)

    try:
        evolution = repeat_stability_assisted_evolution(
            random_architecture_fn=random_architecture_fn,
            mutate_fn=mutate_fn,
            evaluate_fn=evaluator.evaluate_with_seed,
            encode_fn=encode_fn,
            fit_surrogate_fn=active_fit_fn,
            predict_surrogate_fn=active_predict_fn,
            population_size=population_size,
            tournament_size=tournament_size,
            budget=budget,
            candidate_count=candidate_count,
            training_seed_base=training_seed_base,
            stability_penalty_lambda=stability_penalty_lambda,
            repeat_seed=repeat_seed,
            search_rng=random.Random(search_seed),
            repeat_policy_config=repeat_policy_config,
            progress_fn=on_progress,
        )
    except Exception as error:
        _append_run_log(
            run_log_path,
            f"FAILED {type(error).__name__}: {error}",
        )
        raise

    if evaluator.real_training_runs != budget:
        raise RuntimeError("real-training runs did not consume exact budget")
    if evolution.real_training_runs != budget or len(evaluation_records) != budget:
        raise RuntimeError("event log did not consume exact budget")
    if len(repeat_records) != len(evolution.repeat_evaluations):
        raise RuntimeError("repeat log count is inconsistent")
    if len(candidate_records) != len(evolution.candidate_batches) * candidate_count:
        raise RuntimeError("candidate log count is inconsistent")
    if sum(bool(row["selected"]) for row in candidate_records) != len(
        evolution.candidate_batches
    ):
        raise RuntimeError("candidate selected-count is inconsistent")

    final_population_order = [
        member.evaluation_id + 1 for member in evolution.population
    ]
    persist(population_order=final_population_order, completed=True)
    completed_message = (
        f"RS-SA-RE completed real_training_runs={evolution.real_training_runs} "
        f"first_evaluations={len(evolution.history)} "
        f"repeat_evaluations={len(evolution.repeat_evaluations)} "
        f"best_fitness={evolution.best_individual.fitness:.6f}"
    )
    _append_run_log(run_log_path, completed_message)
    print_fn(completed_message)

    return NASNetRSSAREResult(
        evolution=evolution,
        real_training_runs=evaluator.real_training_runs,
        output_dir=output_dir,
        config_path=config_path,
        evaluations_csv_path=evaluations_csv_path,
        candidate_predictions_csv_path=candidate_predictions_csv_path,
        repeat_evaluations_csv_path=repeat_evaluations_csv_path,
        history_json_path=history_json_path,
        run_log_path=run_log_path,
    )


__all__ = [
    "CANDIDATE_CSV_FIELDS",
    "EVALUATION_CSV_FIELDS",
    "NASNetRSSAREResult",
    "REPEAT_CSV_FIELDS",
    "run_nasnet_rs_sa_re",
]
