from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "audit_three_method_pilots.py"
SPEC = importlib.util.spec_from_file_location("audit_three_method_pilots", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_architecture_canonicalisation_ignores_mapping_order_and_whitespace():
    left = '{"normal": [1, 2], "reduction": {"b": 2, "a": 1}}'
    right = '{ "reduction": {"a": 1, "b": 2}, "normal": [1,2] }'
    assert MODULE.canonicalise(left) == MODULE.canonicalise(right)


def test_repeat_rows_are_excluded_from_initialization():
    rows = [
        {
            "evaluation_index": str(index),
            "event": "first_evaluation",
            "architecture": f"arch-{index}",
            "training_seed": str(1000 + index),
            "inserted": "True",
        }
        for index in range(1, 21)
    ]
    rows.insert(
        4,
        {
            "evaluation_index": "21",
            "event": "repeat_evaluation",
            "architecture": "arch-1",
            "training_seed": "9999",
            "inserted": "False",
            "base_evaluation_index": "1",
        },
    )
    selected = MODULE.select_initialization_rows(rows, 20)
    assert len(selected) == 20
    assert all(row["event"] == "first_evaluation" for row in selected)


def test_comparison_reports_exact_mismatch_positions():
    left = [
        MODULE.InitializationRecord(i, f"arch-{i}", str(1000 + i))
        for i in range(1, 21)
    ]
    right = list(left)
    right[2] = MODULE.InitializationRecord(3, "different-arch", "1003")
    right[6] = MODULE.InitializationRecord(7, "arch-7", "different-seed")

    result = MODULE.compare_initializations(left, right)

    assert result["architecture_matches"] == 19
    assert result["architecture_mismatch_indices"] == [3]
    assert result["training_seed_matches"] == 19
    assert result["training_seed_mismatch_indices"] == [7]
    assert result["pass"] is False


def test_method_fairness_note_contains_three_method_contract():
    note = (PROJECT_ROOT / "notes" / "method_fairness.md").read_text(encoding="utf-8")
    for required in (
        "| Item | RE | SA-RE | RS-SA-RE |",
        "argmax(mu_hat)",
        "argmax(mu_hat - lambda * d_hat)",
        "warmup_pairs=4",
        "repeat_interval=4",
        "B=30, including repeats",
        "20/20",
    ):
        assert required in note


def test_machine_readable_audit_has_four_complete_comparisons():
    path = PROJECT_ROOT / "experiments" / "pilot" / "pilot_comparison_audit.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["initialization_size"] == 20
    assert payload["overall_pass"] is True
    assert set(payload["seeds"]) == {"2701", "2702"}

    comparisons_seen = 0
    for seed_result in payload["seeds"].values():
        assert seed_result["initialization_counts"] == {
            "RE": 20,
            "SA-RE": 20,
            "RS-SA-RE": 20,
        }
        for comparison in seed_result["comparisons"].values():
            comparisons_seen += 1
            assert comparison["architecture_matches"] == 20
            assert comparison["architecture_total"] == 20
            assert comparison["training_seed_matches"] == 20
            assert comparison["training_seed_total"] == 20
            assert comparison["pass"] is True
    assert comparisons_seen == 4

