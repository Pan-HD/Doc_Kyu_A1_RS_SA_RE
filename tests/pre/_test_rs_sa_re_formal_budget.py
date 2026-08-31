from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "audit_rs_sa_re_repeat_policy.py"
SPEC = importlib.util.spec_from_file_location("audit_rs_sa_re_repeat_policy", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


EXPECTED_B60 = {
    "real_training_runs": 60,
    "first_evaluations": 49,
    "repeat_evaluations": 11,
    "initial_evaluations": 20,
    "evolution_children": 29,
    "warmup_repeats": 4,
    "periodic_repeats": 7,
    "candidate_rows": 145,
    "selected_rows": 29,
    "final_population": 20,
}


def test_b30_toy_schedule_matches_both_completed_pilots():
    events = MODULE.simulate_repeat_policy(30)
    summary = MODULE.summarise_events(events)
    assert summary == {
        "real_training_runs": 30,
        "first_evaluations": 25,
        "repeat_evaluations": 5,
        "initial_evaluations": 20,
        "evolution_children": 5,
        "warmup_repeats": 4,
        "periodic_repeats": 1,
        "candidate_rows": 25,
        "selected_rows": 5,
        "final_population": 20,
    }


def test_b60_formal_budget_counts_are_exact():
    events = MODULE.simulate_repeat_policy(60)
    assert MODULE.summarise_events(events) == EXPECTED_B60


def test_scheduler_stops_exactly_at_60_without_b61():
    events = MODULE.simulate_repeat_policy(60)
    assert len(events) == 60
    assert events[0].budget == 1
    assert events[-1].budget == 60
    assert {event.budget for event in events} == set(range(1, 61))


def test_expected_warmup_periodic_and_final_event_positions():
    events = MODULE.simulate_repeat_policy(60)
    warmup_budgets = [event.budget for event in events if event.repeat_kind == "warmup"]
    periodic_budgets = [event.budget for event in events if event.repeat_kind == "periodic"]
    assert warmup_budgets == [21, 22, 23, 24]
    assert periodic_budgets == [29, 34, 39, 44, 49, 54, 59]
    assert events[-1].event_type == "first_evaluation"
    assert events[-1].candidate_rows == 5


def test_repeats_never_change_population_best_or_candidate_counts():
    events = MODULE.simulate_repeat_policy(60)
    MODULE.validate_repeat_invariants(events)
    repeats = [event for event in events if event.event_type == "repeat_evaluation"]
    assert len(repeats) == 11
    assert all(event.inserted is False for event in repeats)
    assert all(event.population_before == event.population_after for event in repeats)
    assert all(event.search_best_before == event.search_best_after for event in repeats)
    assert all(event.candidate_rows == 0 for event in repeats)
    assert all(event.selected_rows == 0 for event in repeats)


def test_only_evolutionary_first_evaluations_generate_k_candidates():
    events = MODULE.simulate_repeat_policy(60)
    initialization = events[:20]
    evolution_first = [
        event
        for event in events[20:]
        if event.event_type == "first_evaluation"
    ]
    assert all(event.candidate_rows == 0 for event in initialization)
    assert len(evolution_first) == 29
    assert all(event.candidate_rows == 5 for event in evolution_first)
    assert all(event.selected_rows == 1 for event in evolution_first)


def test_distributed_machine_readable_audit_contains_freeze_gate_values():
    path = PROJECT_ROOT / "experiments" / "pilot" / "rs_sa_re_repeat_policy_audit.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["policy"] == {
        "population_size": 20,
        "candidate_count": 5,
        "warmup_pairs": 4,
        "repeat_interval": 4,
    }
    formal = payload["formal_B60_expected"]
    for key, expected in EXPECTED_B60.items():
        assert formal[key] == expected
    assert payload["overall_pass"] is True

