import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_surrogate_pilot import DiagnosticError, summarize_experiment


EVALUATION_FIELDS = [
    "method",
    "search_seed",
    "evaluation_index",
    "budget",
    "phase",
    "mutation_type",
    "observed_final_val_accuracy",
    "predicted_mu_before_training",
    "surrogate_training_size",
    "selected_candidate_index",
]
CANDIDATE_FIELDS = [
    "method",
    "search_seed",
    "evaluation_index",
    "candidate_index",
    "mutation_type",
    "predicted_mu",
    "selected",
    "surrogate_training_size",
]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_valid_pilot(directory: Path) -> None:
    evaluations: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    for evaluation_index in range(1, 31):
        initialization = evaluation_index <= 20
        selected_index = (evaluation_index - 21) % 5 if not initialization else ""
        predicted = 0.70 + evaluation_index / 1000 if not initialization else ""
        evaluations.append(
            {
                "method": "SA-RE",
                "search_seed": 2701,
                "evaluation_index": evaluation_index,
                "budget": 30,
                "phase": "initialization" if initialization else "evolution",
                "mutation_type": "random_initialization" if initialization else "operation",
                "observed_final_val_accuracy": 0.68 + evaluation_index / 1000,
                "predicted_mu_before_training": predicted,
                "surrogate_training_size": "" if initialization else evaluation_index - 1,
                "selected_candidate_index": selected_index,
            }
        )
        if not initialization:
            for candidate_index in range(5):
                candidate_prediction = (
                    predicted if candidate_index == selected_index else 0.60 + candidate_index / 100
                )
                candidates.append(
                    {
                        "method": "SA-RE",
                        "search_seed": 2701,
                        "evaluation_index": evaluation_index,
                        "candidate_index": candidate_index,
                        "mutation_type": "operation",
                        "predicted_mu": candidate_prediction,
                        "selected": candidate_index == selected_index,
                        "surrogate_training_size": evaluation_index - 1,
                    }
                )
    _write_csv(directory / "evaluations.csv", EVALUATION_FIELDS, evaluations)
    _write_csv(directory / "candidate_predictions.csv", CANDIDATE_FIELDS, candidates)


class SurrogateDiagnosticTests(unittest.TestCase):
    def test_valid_pilot_writes_expected_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            _make_valid_pilot(directory)

            summary = summarize_experiment(directory)

            self.assertEqual(summary["budget_sanity"]["real_evaluations"], 30)
            self.assertEqual(summary["budget_sanity"]["candidate_rows"], 50)
            self.assertEqual(summary["budget_sanity"]["selected_candidate_rows"], 10)
            self.assertEqual(
                summary["surrogate_training_size"]["prediction_time_sizes"],
                list(range(20, 30)),
            )
            self.assertEqual(
                summary["surrogate_training_size"][
                    "final_dataset_size_after_last_real_evaluation"
                ],
                30,
            )
            self.assertTrue(summary["target_accuracy"]["all_in_unit_interval"])
            self.assertTrue(math.isfinite(summary["selected_online_diagnostic"]["mae"]))

            with (directory / "surrogate_online_diagnostic.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                online_rows = list(csv.DictReader(handle))
            self.assertEqual(len(online_rows), 10)
            self.assertEqual(
                list(online_rows[0]),
                [
                    "evaluation_index",
                    "predicted_mu",
                    "observed_accuracy",
                    "error",
                    "absolute_error",
                ],
            )
            stored_summary = json.loads(
                (directory / "surrogate_diagnostic_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(stored_summary, summary)

    def test_rejects_mixed_accuracy_units(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            _make_valid_pilot(directory)
            with (directory / "evaluations.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["observed_final_val_accuracy"] = "72.35"
            _write_csv(directory / "evaluations.csv", EVALUATION_FIELDS, rows)

            with self.assertRaisesRegex(DiagnosticError, "0.0-1.0 scale"):
                summarize_experiment(directory)

    def test_rejects_non_finite_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            _make_valid_pilot(directory)
            path = directory / "candidate_predictions.csv"
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["predicted_mu"] = "nan"
            _write_csv(path, CANDIDATE_FIELDS, rows)

            with self.assertRaisesRegex(DiagnosticError, "must be finite"):
                summarize_experiment(directory)

    def test_rejects_wrong_prediction_time_training_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            _make_valid_pilot(directory)
            path = directory / "candidate_predictions.csv"
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows[:5]:
                row["surrogate_training_size"] = "21"
            _write_csv(path, CANDIDATE_FIELDS, rows)

            with self.assertRaisesRegex(DiagnosticError, "must be 20"):
                summarize_experiment(directory)


if __name__ == "__main__":
    unittest.main()
