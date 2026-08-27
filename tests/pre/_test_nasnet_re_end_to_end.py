"""Mocked Part F end-to-end test; no CNN or CIFAR-10 is loaded."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from src.search.nasnet_re import (
    CSV_FIELDS,
    NASNetTrainingEvaluator,
    run_nasnet_re,
)


@dataclass(frozen=True)
class ToyArchitecture:
    name: str

    def to_dict(self):
        return {"name": self.name}


@dataclass(frozen=True)
class ToyMutationResult:
    architecture: ToyArchitecture
    mutation_type: str


class NASNetREEndToEndTests(unittest.TestCase):
    def test_p4_s2_b8_trains_exactly_eight_times_and_writes_logs(self):
        architecture_calls = []
        mutation_calls = []
        loader_calls = []
        model_calls = []
        trainer_calls = []
        seed_calls = []
        cleanup_calls = []
        console = []

        def random_architecture(_rng):
            index = len(architecture_calls) + 1
            architecture = ToyArchitecture(f"initial-{index}")
            architecture_calls.append(architecture)
            return architecture

        mutation_types = iter(
            ["hidden_state", "identity", "operation", "hidden_state"]
        )

        def mutate(parent, _rng):
            mutation_type = next(mutation_types)
            mutation_calls.append((parent, mutation_type))
            if mutation_type == "identity":
                child = parent
            else:
                child = ToyArchitecture(
                    f"{parent.name}-{mutation_type}-{len(mutation_calls)}"
                )
            return ToyMutationResult(child, mutation_type)

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
            final_accuracy = 0.50 + run_index * 0.01
            return SimpleNamespace(
                final_val_accuracy=final_accuracy,
                best_val_accuracy=final_accuracy + 0.10,
                parameter_count=1_000 + run_index,
                training_time_seconds=10.0 + run_index,
            )

        evaluator = NASNetTrainingEvaluator(
            loader_factory=loader_factory,
            training_config_values={
                "epochs": 1,
                "batch_size": 128,
                "learning_rate": 0.025,
            },
            training_seed_base=20_260_826,
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

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "nasnet_re_20260826"
            result = run_nasnet_re(
                evaluator=evaluator,
                output_dir=output_dir,
                config_text="experiment:\n  method: RE\n",
                method="RE",
                search_seed=1001,
                population_size=4,
                tournament_size=2,
                budget=8,
                random_architecture_fn=random_architecture,
                mutate_fn=mutate,
                print_fn=console.append,
            )

            self.assertEqual(len(architecture_calls), 4)
            self.assertEqual(len(mutation_calls), 4)
            self.assertEqual(len(trainer_calls), 8)
            self.assertEqual(evaluator.real_training_runs, 8)
            self.assertEqual(result.real_training_runs, 8)
            self.assertEqual(len(result.evolution.history), 8)
            self.assertEqual(len(result.evolution.population), 4)
            self.assertEqual(
                [member.evaluation_id for member in result.evolution.population],
                [4, 5, 6, 7],
            )

            expected_seeds = list(range(20_260_826, 20_260_834))
            self.assertEqual(seed_calls, expected_seeds)
            self.assertEqual(loader_calls, expected_seeds)
            self.assertEqual(
                [call["config"].training_seed for call in trainer_calls],
                expected_seeds,
            )
            self.assertEqual(len(model_calls), 8)
            self.assertEqual(len(cleanup_calls), 8)

            expected_files = {
                "config.yaml",
                "evaluations.csv",
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
                csv_rows = list(csv.DictReader(stream))

            self.assertEqual(len(csv_rows), 8)
            self.assertEqual(tuple(csv_rows[0].keys()), CSV_FIELDS)
            self.assertEqual(
                [int(row["evaluation_index"]) for row in csv_rows],
                list(range(1, 9)),
            )
            self.assertEqual(
                [row["phase"] for row in csv_rows],
                ["initialization"] * 4 + ["evolution"] * 4,
            )
            self.assertTrue(
                all(
                    float(row["fitness"])
                    == float(row["final_val_accuracy"])
                    for row in csv_rows
                )
            )
            self.assertTrue(
                all(row["parent_architecture"] == "" for row in csv_rows[:4])
            )
            self.assertTrue(
                all(row["parent_architecture"] != "" for row in csv_rows[4:])
            )
            self.assertEqual(
                [row["mutation_type"] for row in csv_rows[4:]],
                ["hidden_state", "identity", "operation", "hidden_state"],
            )

            identity_row = csv_rows[5]
            self.assertEqual(
                json.loads(identity_row["architecture"]),
                json.loads(identity_row["parent_architecture"]),
            )
            self.assertEqual(len(trainer_calls), 8)

            history = json.loads(
                result.history_json_path.read_text(encoding="utf-8")
            )
            self.assertTrue(history["completed"])
            self.assertEqual(history["real_training_runs"], 8)
            self.assertEqual(history["budget"], 8)
            self.assertEqual(history["final_population_order"], [5, 6, 7, 8])
            self.assertEqual(len(history["evaluations"]), 8)

            self.assertEqual(
                result.config_path.read_text(encoding="utf-8"),
                "experiment:\n  method: RE\n",
            )
            run_log = result.run_log_path.read_text(encoding="utf-8")
            self.assertIn("[1/8] initialization", run_log)
            self.assertIn("[8/8] evolution", run_log)
            self.assertIn("RE completed real_training_runs=8", run_log)
            self.assertTrue(any("[8/8] evolution" in line for line in console))
            self.assertTrue(console[-1].startswith("RE completed"))

    def test_existing_output_is_not_overwritten_without_permission(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            existing_config = output_dir / "config.yaml"
            existing_config.write_text("keep: true\n", encoding="utf-8")

            evaluator = SimpleNamespace(real_training_runs=0)
            with self.assertRaisesRegex(FileExistsError, "output already exists"):
                run_nasnet_re(
                    evaluator=evaluator,
                    output_dir=output_dir,
                    config_text="new: value\n",
                    method="RE",
                    search_seed=1001,
                    population_size=4,
                    tournament_size=2,
                    budget=8,
                    random_architecture_fn=lambda _rng: ToyArchitecture("x"),
                    mutate_fn=lambda parent, _rng: ToyMutationResult(
                        parent,
                        "identity",
                    ),
                    print_fn=lambda _message: None,
                )

            self.assertEqual(
                existing_config.read_text(encoding="utf-8"),
                "keep: true\n",
            )


if __name__ == "__main__":
    unittest.main()
