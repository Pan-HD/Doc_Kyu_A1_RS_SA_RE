"""Tests for the RE/SA-RE matched-initialization audit."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.audit_matched_pilot import (
    AuditError,
    audit_matched_initialization,
)


FIELDS = (
    "method",
    "search_seed",
    "training_seed",
    "evaluation_index",
    "phase",
    "architecture",
    "final_val_accuracy",
)


def _write_initialization_csv(
    path: Path,
    *,
    method: str,
    search_seed: int,
    population_size: int = 4,
    architecture_override: dict[int, str] | None = None,
    training_seed_override: dict[int, int] | None = None,
    accuracy_override: dict[int, float] | None = None,
) -> None:
    architecture_override = architecture_override or {}
    training_seed_override = training_seed_override or {}
    accuracy_override = accuracy_override or {}
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for index in range(1, population_size + 1):
            writer.writerow(
                {
                    "method": method,
                    "search_seed": search_seed,
                    "training_seed": training_seed_override.get(
                        index, 20_260_827 + index - 1
                    ),
                    "evaluation_index": index,
                    "phase": "initialization",
                    "architecture": architecture_override.get(
                        index, f'{{"index":{index},"op":"identity"}}'
                    ),
                    "final_val_accuracy": accuracy_override.get(
                        index, 0.60 + index * 0.01
                    ),
                }
            )


class MatchedPilotAuditTests(unittest.TestCase):
    def test_matching_initialization_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            re_path = Path(directory) / "re.csv"
            sa_path = Path(directory) / "sa.csv"
            _write_initialization_csv(re_path, method="RE", search_seed=2701)
            _write_initialization_csv(sa_path, method="SA-RE", search_seed=2701)

            result = audit_matched_initialization(
                re_path,
                sa_path,
                population_size=4,
                expected_search_seed=2701,
                require_accuracy_match=True,
            )

            self.assertTrue(result.passed)
            self.assertEqual(result.architecture_matches, 4)
            self.assertEqual(result.training_seed_matches, 4)
            self.assertEqual(result.accuracy_matches, 4)

    def test_architecture_mismatch_is_a_hard_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            re_path = Path(directory) / "re.csv"
            sa_path = Path(directory) / "sa.csv"
            _write_initialization_csv(re_path, method="RE", search_seed=2701)
            _write_initialization_csv(
                sa_path,
                method="SA-RE",
                search_seed=2701,
                architecture_override={3: '{"index":999}'},
            )

            result = audit_matched_initialization(
                re_path,
                sa_path,
                population_size=4,
            )

            self.assertFalse(result.passed)
            self.assertEqual(result.architecture_matches, 3)
            self.assertEqual(result.mismatches[0]["evaluation_index"], 3)

    def test_accuracy_can_be_diagnostic_or_required(self):
        with tempfile.TemporaryDirectory() as directory:
            re_path = Path(directory) / "re.csv"
            sa_path = Path(directory) / "sa.csv"
            _write_initialization_csv(re_path, method="RE", search_seed=2702)
            _write_initialization_csv(
                sa_path,
                method="SA-RE",
                search_seed=2702,
                accuracy_override={2: 0.99},
            )

            diagnostic = audit_matched_initialization(
                re_path,
                sa_path,
                population_size=4,
            )
            strict = audit_matched_initialization(
                re_path,
                sa_path,
                population_size=4,
                require_accuracy_match=True,
            )

            self.assertTrue(diagnostic.passed)
            self.assertFalse(diagnostic.accuracy_passed)
            self.assertFalse(strict.passed)

    def test_search_seed_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            re_path = Path(directory) / "re.csv"
            sa_path = Path(directory) / "sa.csv"
            _write_initialization_csv(re_path, method="RE", search_seed=2701)
            _write_initialization_csv(sa_path, method="SA-RE", search_seed=2702)

            with self.assertRaisesRegex(AuditError, "search-seed mismatch"):
                audit_matched_initialization(
                    re_path,
                    sa_path,
                    population_size=4,
                )


if __name__ == "__main__":
    unittest.main()
