from __future__ import annotations

import csv
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "experiments" / "formal" / "manifest.csv"
METHODS = {
    "RE": "re",
    "SA-RE": "sa_re",
    "RS-SA-RE": "rs_sa_re",
}
SEARCH_SEEDS = tuple(range(1001, 1011))
FIELDS = (
    "method",
    "search_seed",
    "config_path",
    "status",
    "start_time",
    "end_time",
    "output_directory",
    "real_training_runs",
    "first_evaluations",
    "repeat_evaluations",
    "exit_code",
    "audit_status",
    "notes",
)
VALID_STATUSES = {"pending", "running", "completed", "failed", "audited"}


def _rows() -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with MANIFEST_PATH.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = tuple(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def test_manifest_has_exact_schema_and_thirty_pending_rows() -> None:
    fieldnames, rows = _rows()
    assert fieldnames == FIELDS
    assert len(rows) == 30
    expected_pairs = [
        (method, str(search_seed))
        for search_seed in SEARCH_SEEDS
        for method in METHODS
    ]
    assert [(row["method"], row["search_seed"]) for row in rows] == expected_pairs
    assert len(set(expected_pairs)) == 30

    for row in rows:
        assert row["status"] in VALID_STATUSES
        assert row["status"] == "pending"
        assert row["audit_status"] == "pending"
        for field in (
            "start_time",
            "end_time",
            "real_training_runs",
            "first_evaluations",
            "repeat_evaluations",
            "exit_code",
            "notes",
        ):
            assert row[field] == ""


def test_manifest_paths_match_the_frozen_configs() -> None:
    _, rows = _rows()
    config_paths: set[str] = set()
    output_directories: set[str] = set()
    for row in rows:
        method = row["method"]
        search_seed = int(row["search_seed"])
        slug = METHODS[method]
        expected_config = f"configs/formal/{slug}/{slug}_{search_seed}.yaml"
        expected_output = f"experiments/formal/{slug}_{search_seed}"
        assert row["config_path"] == expected_config
        assert row["output_directory"] == expected_output
        assert expected_config not in config_paths
        assert expected_output not in output_directories
        config_paths.add(expected_config)
        output_directories.add(expected_output)

        config_path = PROJECT_ROOT / expected_config
        assert config_path.is_file()
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert config["experiment"]["method"] == method
        assert config["experiment"]["search_seed"] == search_seed
        assert config["experiment"]["output_dir"] == expected_output


def test_manifest_status_vocabulary_is_closed_and_unambiguous() -> None:
    assert VALID_STATUSES == {
        "pending",
        "running",
        "completed",
        "failed",
        "audited",
    }
