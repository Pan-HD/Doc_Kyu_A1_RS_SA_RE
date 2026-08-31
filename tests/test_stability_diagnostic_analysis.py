from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import pytest

from scripts.analyze_stability_diagnostic import (
    EXPECTED_ARCHITECTURE_IDS,
    EXPECTED_TRAINING_SEEDS,
    analyze,
    read_completed_observations,
    spearman_with_p_value,
    summarize_architectures,
)


RAW_FIELDS = (
    "architecture_id",
    "training_seed",
    "accuracy_epoch_5",
    "accuracy_epoch_25",
    "training_time",
    "parameter_count",
    "status",
)


def _write_complete_raw_results(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS)
        writer.writeheader()
        for index, architecture_id in enumerate(
            EXPECTED_ARCHITECTURE_IDS,
            start=1,
        ):
            mean_5 = 0.65 + index * 0.005
            mean_25 = 0.70 + index * 0.005
            sd_5 = index * 0.001
            sd_25 = index * 0.0015
            for offset, training_seed in zip(
                (-1, 0, 1),
                EXPECTED_TRAINING_SEEDS,
            ):
                writer.writerow(
                    {
                        "architecture_id": architecture_id,
                        "training_seed": training_seed,
                        "accuracy_epoch_5": mean_5 + offset * sd_5,
                        "accuracy_epoch_25": mean_25 + offset * sd_25,
                        "training_time": 10.0,
                        "parameter_count": 1000 + index,
                        "status": "completed",
                    }
                )


def test_summary_uses_three_seed_mean_and_sample_sd(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw_results.csv"
    _write_complete_raw_results(raw_path)
    summaries = summarize_architectures(read_completed_observations(raw_path))
    assert len(summaries) == 12
    first = summaries[0]
    expected_5 = [0.654, 0.655, 0.656]
    expected_25 = [0.7035, 0.705, 0.7065]
    assert first.n_seeds == 3
    assert first.mean_acc_5 == pytest.approx(statistics.fmean(expected_5))
    assert first.sd_acc_5 == pytest.approx(statistics.stdev(expected_5))
    assert first.mean_acc_25 == pytest.approx(statistics.fmean(expected_25))
    assert first.sd_acc_25 == pytest.approx(statistics.stdev(expected_25))


def test_analysis_writes_12_rows_and_correlation_metadata(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw_results.csv"
    summary_path = tmp_path / "stability_summary.csv"
    correlation_path = tmp_path / "stability_correlation.json"
    _write_complete_raw_results(raw_path)
    payload = analyze(raw_path, summary_path, correlation_path)
    assert payload["n"] == 12
    assert payload["rho"] == pytest.approx(1.0)
    assert payload["p_value"] == pytest.approx(0.0)
    assert payload["go_threshold"] is None

    with summary_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12
    assert [row["architecture_id"] for row in rows] == list(
        EXPECTED_ARCHITECTURE_IDS
    )
    persisted = json.loads(correlation_path.read_text(encoding="utf-8"))
    assert persisted["interpretation"] == (
        "review_on_2026-08-31_without_fixed_go_threshold"
    )


def test_incomplete_input_does_not_create_outputs(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw_results.csv"
    summary_path = tmp_path / "stability_summary.csv"
    correlation_path = tmp_path / "stability_correlation.json"
    _write_complete_raw_results(raw_path)
    rows = raw_path.read_text(encoding="utf-8").splitlines()
    raw_path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="incomplete"):
        analyze(raw_path, summary_path, correlation_path)
    assert not summary_path.exists()
    assert not correlation_path.exists()


def test_duplicate_run_key_is_rejected(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw_results.csv"
    _write_complete_raw_results(raw_path)
    lines = raw_path.read_text(encoding="utf-8").splitlines()
    raw_path.write_text("\n".join(lines + [lines[1]]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate run key"):
        read_completed_observations(raw_path)


def test_spearman_supports_ties_and_direction() -> None:
    positive_rho, positive_p = spearman_with_p_value(
        [1.0, 1.0, 2.0, 3.0],
        [1.0, 1.0, 2.0, 3.0],
    )
    negative_rho, negative_p = spearman_with_p_value(
        [1.0, 2.0, 3.0, 4.0],
        [4.0, 3.0, 2.0, 1.0],
    )
    assert positive_rho == pytest.approx(1.0)
    assert positive_p == pytest.approx(0.0)
    assert negative_rho == pytest.approx(-1.0)
    assert negative_p == pytest.approx(0.0)
