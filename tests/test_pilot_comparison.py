import csv
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.summarize_pilot_comparison import ComparisonError, summarize_pilots


EVALUATION_FIELDS = [
    "method",
    "search_seed",
    "training_seed",
    "evaluation_index",
    "budget",
    "phase",
    "architecture",
    "parent_architecture",
    "mutation_type",
    "final_val_accuracy",
    "parameter_count",
    "training_time",
    "predicted_mu_before_training",
    "surrogate_training_size",
    "selected_candidate_index",
]
CANDIDATE_FIELDS = [
    "method",
    "search_seed",
    "evaluation_index",
    "candidate_index",
    "predicted_mu",
    "selected",
    "surrogate_training_size",
]


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _make_run(root: Path, method: str, seed: int) -> None:
    directory_name = f"{'re' if method == 'RE' else 'sa_re'}_{seed}"
    run_dir = root / directory_name
    run_dir.mkdir(parents=True)
    config: dict[str, object] = {
        "experiment": {"method": method, "search_seed": seed},
        "dataset": {"name": "CIFAR10", "split_seed": 7},
        "network": {"N": 3, "F": 24, "num_classes": 10},
        "training": {"epochs": 5, "batch_size": 128, "training_seed_base": 100},
        "evolution": {"population_size": 20, "tournament_size": 5, "budget": 30},
        "device": {"use_cuda": True, "cuda_index": 0},
    }
    if method == "SA-RE":
        config["evolution"]["candidate_count"] = 5
        config["surrogate"] = {"input_dim": 280, "hidden_dims": [32, 16]}
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    evaluations: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    for index in range(1, 31):
        initialization = index <= 20
        base_accuracy = 0.70 + index / 1000.0
        if method == "SA-RE" and not initialization:
            base_accuracy -= 0.01
        selected_index = 4 if not initialization and method == "SA-RE" else ""
        predicted = 0.74 + index / 1000.0 if method == "SA-RE" and not initialization else ""
        evaluations.append(
            {
                "method": method,
                "search_seed": seed,
                "training_seed": 100 + index,
                "evaluation_index": index,
                "budget": 30,
                "phase": "initialization" if initialization else "evolution",
                "architecture": f"seed-{seed}-initial-{index}" if initialization else f"{method}-{seed}-{index}",
                "parent_architecture": "" if initialization else "parent",
                "mutation_type": "random_initialization" if initialization else "operation",
                "final_val_accuracy": base_accuracy,
                "parameter_count": 1000 + index,
                "training_time": 2.0,
                "predicted_mu_before_training": predicted,
                "surrogate_training_size": index - 1 if method == "SA-RE" and not initialization else "",
                "selected_candidate_index": selected_index,
            }
        )
        if method == "SA-RE" and not initialization:
            for candidate_index in range(5):
                candidates.append(
                    {
                        "method": method,
                        "search_seed": seed,
                        "evaluation_index": index,
                        "candidate_index": candidate_index,
                        "predicted_mu": predicted if candidate_index == 4 else 0.60 + candidate_index / 100,
                        "selected": candidate_index == 4,
                        "surrogate_training_size": index - 1,
                    }
                )
    _write_csv(run_dir / "evaluations.csv", EVALUATION_FIELDS, evaluations)
    if method == "SA-RE":
        _write_csv(run_dir / "candidate_predictions.csv", CANDIDATE_FIELDS, candidates)
    history = {
        "completed": True,
        "real_training_runs": 30,
        "final_population_order": list(range(11, 31)),
    }
    (run_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")
    (run_dir / "run.log").write_text(
        f"2026-08-28T00:00:00+00:00 START method={method}\n"
        f"2026-08-28T00:10:00+00:00 {method} completed\n",
        encoding="utf-8",
    )


def _make_four_runs(root: Path) -> None:
    for method, seed in (("RE", 2701), ("SA-RE", 2701), ("RE", 2702), ("SA-RE", 2702)):
        _make_run(root, method, seed)


class PilotComparisonTests(unittest.TestCase):
    def test_generates_summary_audit_and_plot_with_expected_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "pilot"
            output = Path(temporary_directory) / "output"
            _make_four_runs(root)

            audit = summarize_pilots(root, output_dir=output)

            self.assertEqual(audit["result"], "PASS")
            self.assertFalse(audit["performance_superiority_required_for_pass"])
            self.assertTrue((output / "pilot_best_so_far_diagnostic.png").is_file())
            self.assertGreater((output / "pilot_best_so_far_diagnostic.png").stat().st_size, 1000)
            with (output / "pilot_comparison_summary.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            re_2701 = rows[0]
            self.assertEqual(re_2701["method"], "RE")
            self.assertAlmostEqual(float(re_2701["best_at_budget_20"]), 0.720)
            self.assertAlmostEqual(float(re_2701["best_at_budget_25"]), 0.725)
            self.assertAlmostEqual(float(re_2701["best_at_budget_30"]), 0.730)
            self.assertAlmostEqual(float(re_2701["runtime_seconds"]), 600.0)
            self.assertAlmostEqual(float(re_2701["mean_training_time_seconds"]), 2.0)
            self.assertAlmostEqual(float(re_2701["final_population_best"]), 0.730)
            self.assertEqual(int(re_2701["parameter_count_of_best"]), 1030)

    def test_matched_pairing_rejects_initial_architecture_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "pilot"
            _make_four_runs(root)
            path = root / "sa_re_2701" / "evaluations.csv"
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["architecture"] = "mismatched-architecture"
            _write_csv(path, EVALUATION_FIELDS, rows)

            with self.assertRaisesRegex(ComparisonError, "matched_initial_conditions"):
                summarize_pilots(root, output_dir=Path(temporary_directory) / "output")

    def test_final_population_order_must_follow_fifo(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "pilot"
            _make_four_runs(root)
            path = root / "re_2702" / "history.json"
            history = json.loads(path.read_text(encoding="utf-8"))
            history["final_population_order"] = list(range(1, 21))
            path.write_text(json.dumps(history), encoding="utf-8")

            with self.assertRaisesRegex(ComparisonError, "all_runs_complete_and_budget_exact"):
                summarize_pilots(root, output_dir=Path(temporary_directory) / "output")

    def test_selected_candidate_must_have_maximum_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "pilot"
            _make_four_runs(root)
            path = root / "sa_re_2702" / "candidate_predictions.csv"
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["predicted_mu"] = "0.999"
            _write_csv(path, CANDIDATE_FIELDS, rows)

            with self.assertRaisesRegex(ComparisonError, "k5_candidate_screening_valid"):
                summarize_pilots(root, output_dir=Path(temporary_directory) / "output")


if __name__ == "__main__":
    unittest.main()
