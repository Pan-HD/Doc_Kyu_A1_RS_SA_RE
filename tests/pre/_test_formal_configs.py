from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMAL_DIR = PROJECT_ROOT / "configs" / "formal"
METHOD_FILES = {
    "RE": "re.yaml",
    "SA-RE": "sa_re.yaml",
    "RS-SA-RE": "rs_sa_re.yaml",
}


def _load(name: str) -> dict:
    value = yaml.safe_load((FORMAL_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _configs() -> dict[str, dict]:
    return {method: _load(name) for method, name in METHOD_FILES.items()}


def test_all_formal_configs_remain_pre_freeze_and_non_runnable() -> None:
    for method, config in _configs().items():
        experiment = config["experiment"]
        assert experiment["method"] == method
        assert experiment["mode"] == "formal"
        assert experiment["status"] == "pre-freeze"
        assert experiment["do_not_run"] is True
        assert experiment["search_seed"] is None
        assert experiment["freeze_date"] == "2026-08-31"


def test_common_fairness_fields_are_identical() -> None:
    configs = _configs()
    reference = configs["RE"]
    for method, config in configs.items():
        assert config["dataset"] == reference["dataset"], method
        assert config["network"] == reference["network"], method
        assert config["training"] == reference["training"], method
        assert config["implementation"] == reference["implementation"], method
        assert config["device"] == reference["device"], method
        assert config["evolution"]["population_size"] == 20
        assert config["evolution"]["tournament_size"] == 5
        assert config["evolution"]["budget"] == 60


def test_only_intended_method_mechanisms_differ() -> None:
    configs = _configs()
    re_config = configs["RE"]
    sa_config = configs["SA-RE"]
    rs_config = configs["RS-SA-RE"]

    assert re_config["selection"]["candidate_count"] == 1
    assert "surrogate" not in re_config
    assert "stability" not in re_config

    assert sa_config["evolution"]["candidate_count"] == 5
    assert sa_config["selection"]["child_selection"] == (
        "argmax_predicted_mean"
    )
    assert "surrogate" in sa_config
    assert "stability" not in sa_config

    assert rs_config["evolution"]["candidate_count"] == 5
    assert rs_config["selection"]["child_selection"] == (
        "argmax_penalized_score"
    )
    assert "surrogate" in rs_config
    assert rs_config["stability"]["warmup_pairs"] == 4
    assert rs_config["stability"]["repeat_interval"] == 4
    assert rs_config["stability"]["lambda"] == 1.0


def test_sa_and_rs_surrogate_training_settings_match() -> None:
    configs = _configs()
    assert configs["SA-RE"]["surrogate"] == configs["RS-SA-RE"][
        "surrogate"
    ]
    surrogate = configs["SA-RE"]["surrogate"]
    assert surrogate["input_dim"] == 280
    assert surrogate["hidden_dims"] == [32, 16]
    assert surrogate["optimizer"] == "Adam"
    assert surrogate["learning_rate"] == 0.001
    assert surrogate["weight_decay"] == 0.0001
    assert surrogate["loss"] == "MSE"
    assert surrogate["steps"] == 200


def test_all_methods_reference_one_matched_seed_manifest() -> None:
    configs = _configs()
    manifest_reference = "configs/formal/seeds.yaml"
    for config in configs.values():
        assert config["experiment"]["seed_manifest"] == manifest_reference

    seeds = _load("seeds.yaml")
    assert seeds["status"] == "pre-freeze"
    assert seeds["do_not_run"] is True
    assert seeds["methods"] == ["RE", "SA-RE", "RS-SA-RE"]
    assert seeds["search_seeds"] == list(range(1001, 1011))
    assert len(set(seeds["search_seeds"])) == 10
    assert seeds["expected_runs_per_method"] == 10
    assert seeds["expected_total_runs"] == 30


def test_formal_budget_audit_expectations_are_exact() -> None:
    configs = _configs()
    for config in configs.values():
        assert config["audit_expectations"]["real_training_runs"] == 60
        assert config["audit_expectations"]["final_population"] == 20

    rs_audit = configs["RS-SA-RE"]["audit_expectations"]
    assert rs_audit["first_evaluations"] == 49
    assert rs_audit["repeat_evaluations"] == 11
    assert rs_audit["evolution_children"] == 29
    assert rs_audit["candidate_rows"] == 145
    assert rs_audit["selected_rows"] == 29
