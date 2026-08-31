from __future__ import annotations

import csv
import importlib.util
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "analyze_rs_sa_re_pilot_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("analyze_rs_sa_re_pilot_diagnostics", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def candidate(candidate_id: int, mu: float, predicted_d: float):
    return MODULE.Candidate(str(candidate_id), mu, predicted_d, None, None, 4)


def test_instability_label_uses_absolute_difference_and_arithmetic_mean():
    label = MODULE.make_label_pair(2701, 3, 0.80, 0.7806, 23)
    assert math.isclose(label.mean_target, 0.7903)
    assert math.isclose(label.instability_target, 0.0194)


def test_descriptive_statistics_use_sample_standard_deviation_and_linear_iqr():
    summary = MODULE.describe([0.0004, 0.0052, 0.0102, 0.0108, 0.0194], 2701)
    assert summary["n"] == 5
    assert math.isclose(summary["median"], 0.0102)
    assert math.isclose(summary["std"], 0.007089428749906441)
    assert math.isclose(summary["q1"], 0.0052)
    assert math.isclose(summary["q3"], 0.0108)
    assert math.isclose(summary["iqr"], 0.0056)


def test_ranking_force_ratio_matches_definition():
    ratio = MODULE.range_ratio(
        mu_values=[0.80, 0.82, 0.81],
        d_values=[0.04, 0.10, 0.07],
        lambda_value=1.0,
        epsilon=1e-12,
    )
    assert math.isclose(ratio, 0.06 / (0.02 + 1e-12))


def test_lambda_zero_equals_argmax_mu_and_lambda_one_can_change_selection():
    candidates = [
        candidate(1, 0.809312, 0.071039),
        candidate(2, 0.790000, 0.060000),
        candidate(3, 0.795000, 0.064000),
        candidate(4, 0.805813, 0.059761),
        candidate(5, 0.780000, 0.050000),
    ]
    assert MODULE.choose_candidate(candidates, 0.0).candidate_id == "1"
    assert MODULE.choose_candidate(candidates, 1.0).candidate_id == "4"


def test_output_templates_have_exact_headers_and_no_fabricated_rows():
    expected = {
        "rs_sa_re_instability_labels.csv": MODULE.LABEL_COLUMNS,
        "rs_sa_re_instability_summary.csv": MODULE.SUMMARY_COLUMNS,
        "rs_sa_re_surrogate_diagnostic.csv": MODULE.DIAGNOSTIC_COLUMNS,
        "rs_sa_re_lambda_sensitivity.csv": MODULE.SENSITIVITY_COLUMNS,
        "rs_sa_re_lambda_sensitivity_summary.csv": MODULE.SENSITIVITY_SUMMARY_COLUMNS,
    }
    pilot_root = PROJECT_ROOT / "experiments" / "pilot"
    for filename, columns in expected.items():
        path = pilot_root / filename
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        assert tuple(rows[0]) == tuple(columns)
        # The distributed files are schema-only. Real rows are written locally
        # by the analyzer from the user's frozen pilot logs.
        if len(rows) > 1:
            assert all(cell != "" for cell in rows[1][:2])


def test_generated_outputs_when_present_have_expected_row_counts():
    pilot_root = PROJECT_ROOT / "experiments" / "pilot"
    labels_path = pilot_root / "rs_sa_re_instability_labels.csv"
    with labels_path.open("r", encoding="utf-8-sig", newline="") as handle:
        labels = list(csv.DictReader(handle))
    if not labels:
        return
    assert len(labels) == 10

    with (pilot_root / "rs_sa_re_surrogate_diagnostic.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        diagnostics = list(csv.DictReader(handle))
    with (pilot_root / "rs_sa_re_lambda_sensitivity.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        sensitivity = list(csv.DictReader(handle))
    assert len(diagnostics) == 10
    assert len(sensitivity) == 10
    assert all(row["argmax_mu"] == row["argmax_lambda_0"] for row in sensitivity)
