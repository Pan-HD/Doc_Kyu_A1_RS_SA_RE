from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMAL_DIR = PROJECT_ROOT / "configs" / "formal"
SEED_MANIFEST_PATH = FORMAL_DIR / "matched_seed_manifest.yaml"
SEARCH_SEEDS = tuple(range(1001, 1011))
METHODS = {
    "RE": "re",
    "SA-RE": "sa_re",
    "RS-SA-RE": "rs_sa_re",
}


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict), path
    return value


def _path(method: str, search_seed: int) -> Path:
    slug = METHODS[method]
    return FORMAL_DIR / slug / f"{slug}_{search_seed}.yaml"


def _config(method: str, search_seed: int) -> dict:
    return _load(_path(method, search_seed))


def test_exactly_thirty_seed_specific_formal_configs_exist() -> None:
    discovered: list[Path] = []
    for method, slug in METHODS.items():
        paths = sorted((FORMAL_DIR / slug).glob("*.yaml"))
        assert paths == [_path(method, seed) for seed in SEARCH_SEEDS]
        discovered.extend(paths)
    assert len(discovered) == 30


def test_every_formal_config_has_frozen_concrete_run_identity() -> None:
    output_directories: set[str] = set()
    for method, slug in METHODS.items():
        for search_seed in SEARCH_SEEDS:
            config = _config(method, search_seed)
            experiment = config["experiment"]
            assert experiment["name"] == f"nasnet_{slug}_formal_{search_seed}"
            assert experiment["method"] == method
            assert experiment["mode"] == "formal"
            assert experiment["status"] == "frozen"
            assert experiment["do_not_run"] is False
            assert experiment["freeze_date"] == "2026-08-31"
            assert experiment["search_seed"] == search_seed
            assert experiment["seed_manifest"] == (
                "configs/formal/matched_seed_manifest.yaml"
            )
            assert experiment["output_dir"] == (
                f"experiments/formal/{slug}_{search_seed}"
            )
            assert "{" not in experiment["output_dir"]
            assert experiment["overwrite"] is False
            assert experiment["output_dir"] not in output_directories
            output_directories.add(experiment["output_dir"])
    assert len(output_directories) == 30


def test_common_fairness_fields_are_identical_for_every_matched_seed() -> None:
    common_sections = ("dataset", "network", "training", "implementation", "device")
    for search_seed in SEARCH_SEEDS:
        configs = {
            method: _config(method, search_seed)
            for method in METHODS
        }
        reference = configs["RE"]
        for method, config in configs.items():
            for section in common_sections:
                assert config[section] == reference[section], (method, section)
            assert config["evolution"]["population_size"] == 20
            assert config["evolution"]["tournament_size"] == 5
            assert config["evolution"]["budget"] == 60


def test_shared_formal_protocol_values_are_exact() -> None:
    config = _config("RE", SEARCH_SEEDS[0])
    assert config["dataset"] == {
        "name": "CIFAR10",
        "data_root": "data/cifar10",
        "split_dir": "data/splits/nasnet_v041",
        "split_seed": 20260823,
        "train_size": 45000,
        "val_size": 5000,
        "official_test_size": 10000,
        "download": True,
        "augment_train": True,
        "num_workers": 4,
        "pin_memory": True,
    }
    assert config["network"] == {"N": 3, "F": 24, "num_classes": 10}
    training = config["training"]
    assert training["epochs"] == 5
    assert training["batch_size"] == 128
    assert training["optimizer"] == "SGD"
    assert training["learning_rate"] == 0.025
    assert training["momentum"] == 0.9
    assert training["weight_decay"] == 0.0005
    assert training["amp"] is True
    assert training["training_seed_base"] == 20260827
    implementation = config["implementation"]
    assert implementation["tournament"] == "with_replacement"
    assert implementation["aging"] == "FIFO"
    assert config["selection"]["score"] == (
        "true_first_seed_final_val_accuracy"
    )


def test_only_frozen_method_mechanisms_differ() -> None:
    for search_seed in SEARCH_SEEDS:
        re_config = _config("RE", search_seed)
        sa_config = _config("SA-RE", search_seed)
        rs_config = _config("RS-SA-RE", search_seed)

        assert re_config["selection"]["candidate_count"] == 1
        assert "surrogate" not in re_config
        assert "stability" not in re_config

        assert sa_config["evolution"]["candidate_count"] == 5
        assert sa_config["selection"]["child_selection"] == (
            "argmax_predicted_mean"
        )
        assert sa_config["selection"]["score"] == "predicted_mean_accuracy"
        assert "surrogate" in sa_config
        assert "stability" not in sa_config

        assert rs_config["evolution"]["candidate_count"] == 5
        assert rs_config["selection"]["child_selection"] == (
            "argmax_penalized_score"
        )
        assert rs_config["selection"]["score"] == (
            "predicted_mean_minus_lambda_times_predicted_instability"
        )
        assert rs_config["stability"]["warmup_pairs"] == 4
        assert rs_config["stability"]["repeat_interval"] == 4
        assert rs_config["stability"]["repeat_rate_beta"] == 0.25
        assert rs_config["stability"]["lambda"] == 1.0


def test_sa_and_rs_surrogate_training_settings_match() -> None:
    for search_seed in SEARCH_SEEDS:
        sa_surrogate = _config("SA-RE", search_seed)["surrogate"]
        rs_surrogate = _config("RS-SA-RE", search_seed)["surrogate"]
        assert sa_surrogate == rs_surrogate
        assert sa_surrogate == {
            "input_dim": 280,
            "hidden_dims": [32, 16],
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "loss": "MSE",
            "steps": 200,
            "seed_offset": 900000,
        }


def test_matched_seed_manifest_freezes_independent_rng_namespaces() -> None:
    manifest = _load(SEED_MANIFEST_PATH)
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "frozen"
    assert manifest["freeze_date"] == "2026-08-31"
    assert manifest["methods"] == list(METHODS)
    assert manifest["search_seeds"] == list(SEARCH_SEEDS)
    assert manifest["expected_runs_per_method"] == 10
    assert manifest["expected_total_runs"] == 30

    rng = manifest["rng_namespaces"]
    assert rng["search"]["shared_across_methods"] is True
    assert rng["training"]["base"] == 20260827
    assert rng["training"]["replica_one_preserves_first_training_schedule"] is True
    assert rng["surrogate"]["offset"] == 900000
    assert rng["surrogate"]["advances_search_rng"] is False
    assert rng["repeat"]["offset"] == 1800000
    assert rng["repeat"]["advances_search_rng"] is False

    expected_runs = [
        {
            "search_seed": seed,
            "surrogate_seed": 900000 + seed,
            "repeat_seed": 1800000 + seed,
        }
        for seed in SEARCH_SEEDS
    ]
    assert manifest["runs"] == expected_runs
    for item in expected_runs:
        rs_config = _config("RS-SA-RE", item["search_seed"])
        assert rs_config["stability"]["repeat_seed"] == item["repeat_seed"]


def test_formal_budget_audit_expectations_are_exact() -> None:
    for search_seed in SEARCH_SEEDS:
        re_audit = _config("RE", search_seed)["audit_expectations"]
        sa_audit = _config("SA-RE", search_seed)["audit_expectations"]
        rs_audit = _config("RS-SA-RE", search_seed)["audit_expectations"]

        assert re_audit == {
            "real_training_runs": 60,
            "first_evaluations": 60,
            "repeat_evaluations": 0,
            "final_population": 20,
        }
        assert sa_audit == {
            "real_training_runs": 60,
            "first_evaluations": 60,
            "repeat_evaluations": 0,
            "evolution_children": 40,
            "candidate_rows": 200,
            "selected_rows": 40,
            "final_population": 20,
        }
        assert rs_audit == {
            "real_training_runs": 60,
            "first_evaluations": 49,
            "repeat_evaluations": 11,
            "initialization_first_evaluations": 20,
            "evolution_children": 29,
            "candidate_rows": 145,
            "selected_rows": 29,
            "final_population": 20,
        }
