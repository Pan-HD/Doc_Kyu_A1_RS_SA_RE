"""Mocked NASNet SA-RE integration test; no CNN or CIFAR-10 is loaded."""

from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from src.search.nasnet_re import NASNetTrainingEvaluator
from src.search.nasnet_sa_re import (
    CANDIDATE_CSV_FIELDS,
    EVALUATION_CSV_FIELDS,
    run_nasnet_sa_re,
)
from src.search.surrogate_assisted_evolution import SurrogateScoreResult


@dataclass(frozen=True)
class ToyArchitecture:
    name: str

    def to_dict(self):
        return {"name": self.name}


@dataclass(frozen=True)
class ToyMutationResult:
    architecture: ToyArchitecture
    mutation_type: str


class NASNetSAREndToEndTests(unittest.TestCase):
    def test_p4_s2_b6_k5_trains_six_times_and_logs_ten_candidates(self):
        architecture_calls = []
        mutation_calls = []
        loader_calls = []
        model_calls = []
        trainer_calls = []
        seed_calls = []
        cleanup_calls = []
        score_calls = []
        console = []

        def random_architecture(_rng):
            architecture = ToyArchitecture(
                f"initial-{len(architecture_calls) + 1}"
            )
            architecture_calls.append(architecture)
            return architecture

        def mutate(parent, _rng):
            candidate_index = len(mutation_calls) % 5
            if candidate_index == 0:
                child = parent
                mutation_type = "identity"
            elif candidate_index in (1, 2):
                child = ToyArchitecture(
                    f"{parent.name}-hidden-{len(mutation_calls)}"
                )
                mutation_type = "hidden_state"
            else:
                child = ToyArchitecture(
                    f"{parent.name}-operation-{len(mutation_calls)}"
                )
                mutation_type = "operation"
            result = ToyMutationResult(child, mutation_type)
            mutation_calls.append((parent, result))
            return result

        def loader_factory(training_seed):
            loader_calls.append(training_seed)
            return (
                f"search-train-{training_seed}",
                f"search-val-{training_seed}",
            )

        def model_builder(architecture, N, F, num_classes):
            model = (architecture, N, F, num_classes)
            model_calls.append(model)
            return model

        def training_config_factory(**values):
            return SimpleNamespace(**values)

        def trainer_fn(model, train_loader, val_loader, config, device):
            run_index = len(trainer_calls)
            trainer_calls.append(
                {
                    "model": model,
                    "train_loader": train_loader,
                    "val_loader": val_loader,
                    "config": config,
                    "device": device,
                }
            )
            final_accuracy = 0.60 + run_index * 0.02
            return SimpleNamespace(
                final_val_accuracy=final_accuracy,
                best_val_accuracy=final_accuracy + 0.05,
                parameter_count=10_000 + run_index,
                training_time_seconds=5.0 + run_index,
            )

        evaluator = NASNetTrainingEvaluator(
            loader_factory=loader_factory,
            training_config_values={
                "epochs": 1,
                "batch_size": 128,
                "learning_rate": 0.025,
            },
            training_seed_base=20_260_827,
            device="cuda:0",
            N=3,
            F=24,
            num_classes=10,
            model_builder=model_builder,
            training_config_factory=training_config_factory,
            trainer_fn=trainer_fn,
            seed_fn=seed_calls.append,
            cleanup_fn=lambda model, device: cleanup_calls.append(
                (model, device)
            ),
        )

        def score_candidates(dataset, encodings):
            score_calls.append((len(dataset), len(encodings)))
            return SurrogateScoreResult(
                scores=(0.65, 0.72, 0.68, 0.81, 0.70),
                training_mse=0.01 * len(score_calls),
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "sa_re_2710"
            result = run_nasnet_sa_re(
                evaluator=evaluator,
                output_dir=output_dir,
                config_text="experiment:\n  method: SA-RE\n",
                method="SA-RE",
                search_seed=2710,
                population_size=4,
                tournament_size=2,
                budget=6,
                candidate_count=5,
                surrogate_config_values={
                    "input_dim": 1,
                    "hidden_dims": [4, 2],
                    "optimizer": "Adam",
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                    "steps": 1,
                    "seed_offset": 900000,
                },
                random_architecture_fn=random_architecture,
                mutate_fn=mutate,
                encode_fn=lambda architecture: torch.tensor(
                    [float(len(architecture.name))]
                ),
                surrogate_score_fn=score_candidates,
                print_fn=console.append,
            )

            self.assertEqual(len(architecture_calls), 4)
            self.assertEqual(len(mutation_calls), 10)
            self.assertEqual(len(trainer_calls), 6)
            self.assertEqual(evaluator.real_training_runs, 6)
            self.assertEqual(result.real_training_runs, 6)
            self.assertEqual(len(result.evolution.history), 6)
            self.assertEqual(len(result.evolution.population), 4)
            self.assertEqual(len(result.evolution.surrogate_dataset), 6)
            self.assertEqual(
                [member.evaluation_id for member in result.evolution.population],
                [2, 3, 4, 5],
            )
            self.assertEqual(score_calls, [(4, 5), (5, 5)])

            expected_seeds = list(range(20_260_827, 20_260_833))
            self.assertEqual(seed_calls, expected_seeds)
            self.assertEqual(loader_calls, expected_seeds)
            self.assertEqual(
                [call["config"].training_seed for call in trainer_calls],
                expected_seeds,
            )
            self.assertEqual(len(model_calls), 6)
            self.assertEqual(len(cleanup_calls), 6)

            expected_files = {
                "config.yaml",
                "evaluations.csv",
                "candidate_predictions.csv",
                "history.json",
                "run.log",
            }
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                expected_files,
            )

            with result.evaluations_csv_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as stream:
                evaluation_rows = list(csv.DictReader(stream))
            self.assertEqual(len(evaluation_rows), 6)
            self.assertEqual(
                tuple(evaluation_rows[0].keys()),
                EVALUATION_CSV_FIELDS,
            )
            self.assertEqual(
                [row["phase"] for row in evaluation_rows],
                ["initialization"] * 4 + ["evolution"] * 2,
            )
            self.assertTrue(
                all(int(row["budget"]) == 6 for row in evaluation_rows)
            )
            self.assertTrue(
                all(
                    float(row["fitness"])
                    == float(row["observed_final_val_accuracy"])
                    for row in evaluation_rows
                )
            )
            self.assertTrue(
                all(
                    row["predicted_mu_before_training"] == ""
                    for row in evaluation_rows[:4]
                )
            )
            self.assertEqual(
                [int(row["selected_candidate_index"]) for row in evaluation_rows[4:]],
                [3, 3],
            )

            with result.candidate_predictions_csv_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as stream:
                candidate_rows = list(csv.DictReader(stream))
            self.assertEqual(len(candidate_rows), 10)
            self.assertEqual(
                tuple(candidate_rows[0].keys()),
                CANDIDATE_CSV_FIELDS,
            )
            selected_rows = [
                row for row in candidate_rows if row["selected"] == "True"
            ]
            self.assertEqual(len(selected_rows), 2)
            self.assertEqual(
                [int(row["candidate_index"]) for row in selected_rows],
                [3, 3],
            )
            self.assertEqual(
                [int(row["evaluation_index"]) for row in selected_rows],
                [5, 6],
            )
            for evaluation_index in (5, 6):
                batch_rows = [
                    row
                    for row in candidate_rows
                    if int(row["evaluation_index"]) == evaluation_index
                ]
                self.assertEqual(len(batch_rows), 5)
                self.assertEqual(
                    sum(row["selected"] == "True" for row in batch_rows),
                    1,
                )
                self.assertTrue(
                    all(
                        math.isfinite(float(row["predicted_mu"]))
                        for row in batch_rows
                    )
                )
                self.assertTrue(
                    all(
                        int(row["surrogate_training_size"])
                        == evaluation_index - 1
                        for row in batch_rows
                    )
                )

            history = json.loads(
                result.history_json_path.read_text(encoding="utf-8")
            )
            self.assertTrue(history["completed"])
            self.assertEqual(history["real_training_runs"], 6)
            self.assertEqual(history["candidate_predictions"], 10)
            self.assertEqual(history["final_population_order"], [3, 4, 5, 6])
            self.assertEqual(history["surrogate_seed"], 902710)

            self.assertEqual(
                result.config_path.read_text(encoding="utf-8"),
                "experiment:\n  method: SA-RE\n",
            )
            run_log = result.run_log_path.read_text(encoding="utf-8")
            self.assertIn("[1/6] initialization", run_log)
            self.assertIn("[6/6] evolution", run_log)
            self.assertIn("candidate_predictions=10", run_log)
            self.assertTrue(console[-1].startswith("SA-RE completed"))


if __name__ == "__main__":
    unittest.main()
