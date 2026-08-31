from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "consolidate_pilot_results.py"
SPEC = importlib.util.spec_from_file_location("consolidate_pilot_results", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


EXPECTED_DIRS = {
    "re_2701",
    "re_2702",
    "sa_re_2701",
    "sa_re_2702",
    "rs_sa_re_2701",
    "rs_sa_re_2702",
}


def make_evaluation(
    budget: int,
    accuracy: float,
    *,
    repeat: bool = False,
    inserted: bool | None = True,
):
    return MODULE.Evaluation(
        budget=budget,
        accuracy=accuracy,
        event_type="repeat_evaluation" if repeat else "first_evaluation",
        is_repeat=repeat,
        inserted=False if repeat else inserted,
        training_time=10.0,
        parameter_count=1000 + budget,
    )


def test_repeat_consumes_budget_but_does_not_update_search_best():
    evaluations = [make_evaluation(i, 0.50 + i / 1000) for i in range(1, 29)]
    evaluations.append(make_evaluation(29, 0.999, repeat=True))
    evaluations.append(make_evaluation(30, 0.600))

    curve = MODULE.build_curve(evaluations, "RS-SA-RE", 2701)

    assert curve[27]["search_best_so_far"] == pytest.approx(0.528)
    assert curve[28]["budget"] == 29
    assert curve[28]["accuracy"] == pytest.approx(0.999)
    assert curve[28]["search_best_so_far"] == pytest.approx(0.528)
    assert curve[29]["search_best_so_far"] == pytest.approx(0.600)


def test_summary_excludes_repeat_from_final_best_and_population():
    evaluations = [make_evaluation(i, 0.50 + i / 1000) for i in range(1, 29)]
    evaluations.append(make_evaluation(29, 0.999, repeat=True))
    evaluations.append(make_evaluation(30, 0.600))

    summary, _ = MODULE.summarise_run(evaluations, "RS-SA-RE", 2701, 20)

    assert summary["real_training_runs"] == 30
    assert summary["first_evaluation_count"] == 29
    assert summary["repeat_evaluation_count"] == 1
    assert summary["final_best"] == pytest.approx(0.600)
    assert summary["final_population_best"] == pytest.approx(0.600)
    assert summary["parameter_count_of_best"] == 1030


def test_required_pilot_directories_exist_when_pilot_root_is_present():
    pilot_root = PROJECT_ROOT / "experiments" / "pilot"
    if not pilot_root.exists():
        pytest.skip("package test: experiments/pilot is supplied by the target project")
    missing = sorted(name for name in EXPECTED_DIRS if not (pilot_root / name).is_dir())
    assert not missing, f"missing pilot directories: {missing}"


def test_generated_summary_has_six_unique_method_seed_rows_when_present():
    path = PROJECT_ROOT / "experiments" / "pilot" / "pilot_comparison_summary.csv"
    if not path.exists():
        pytest.skip("run consolidate_pilot_results.py to generate the summary")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    pairs = {(row["method"], int(row["search_seed"])) for row in rows}
    assert len(rows) == 6
    assert pairs == {
        ("RE", 2701),
        ("SA-RE", 2701),
        ("RS-SA-RE", 2701),
        ("RE", 2702),
        ("SA-RE", 2702),
        ("RS-SA-RE", 2702),
    }
