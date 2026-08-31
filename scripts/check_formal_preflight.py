"""Perform the final no-GPU audit before the first A1 formal run."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEARCH_SEEDS = tuple(range(1001, 1011))
METHODS = {"RE": "re", "SA-RE": "sa_re", "RS-SA-RE": "rs_sa_re"}


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def _config_path(root: Path, method: str, search_seed: int) -> Path:
    slug = METHODS[method]
    return root / "configs" / "formal" / slug / f"{slug}_{search_seed}.yaml"


def _configs(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    values: dict[tuple[str, int], dict[str, Any]] = {}
    for method in METHODS:
        for seed in SEARCH_SEEDS:
            values[(method, seed)] = _load_yaml(_config_path(root, method, seed))
    return values


def _check_config_set(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    discovered = sorted((root / "configs" / "formal").glob("*/*.yaml"))
    expected = sorted(
        _config_path(root, method, seed)
        for method in METHODS
        for seed in SEARCH_SEEDS
    )
    if discovered != expected:
        raise ValueError("formal config set must contain exactly 30 files")
    configs = _configs(root)
    for (method, seed), config in configs.items():
        experiment = config["experiment"]
        if experiment["method"] != method:
            raise ValueError(f"method mismatch for {method} seed {seed}")
        if experiment["mode"] != "formal" or experiment["status"] != "frozen":
            raise ValueError(f"unfrozen config for {method} seed {seed}")
        if experiment["do_not_run"] is not False:
            raise ValueError(f"do_not_run is not false for {method} seed {seed}")
        if int(experiment["search_seed"]) != seed:
            raise ValueError(f"search seed mismatch for {method} seed {seed}")
        if bool(experiment["overwrite"]):
            raise ValueError(f"overwrite must be false for {method} seed {seed}")
        if "{" in str(experiment["output_dir"]):
            raise ValueError(f"output_dir is not concrete for {method} seed {seed}")
    return configs


def _check_common(
    configs: Mapping[tuple[str, int], Mapping[str, Any]],
    section: str,
) -> None:
    reference = configs[("RE", 1001)][section]
    for key, config in configs.items():
        if config[section] != reference:
            raise ValueError(f"{section} differs for {key}")


def _check_dataset(configs: Mapping[tuple[str, int], Mapping[str, Any]]) -> None:
    _check_common(configs, "dataset")
    dataset = configs[("RE", 1001)]["dataset"]
    expected = {
        "name": "CIFAR10",
        "split_seed": 20260823,
        "train_size": 45000,
        "val_size": 5000,
    }
    for name, value in expected.items():
        if dataset[name] != value:
            raise ValueError(f"dataset.{name} must equal {value}")


def _check_trainer(configs: Mapping[tuple[str, int], Mapping[str, Any]]) -> None:
    _check_common(configs, "training")
    _check_common(configs, "implementation")
    training = configs[("RE", 1001)]["training"]
    expected = {
        "epochs": 5,
        "batch_size": 128,
        "optimizer": "SGD",
        "learning_rate": 0.025,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "amp": True,
        "training_seed_base": 20260827,
    }
    for name, value in expected.items():
        if training[name] != value:
            raise ValueError(f"training.{name} must equal {value}")


def _check_nasnet(configs: Mapping[tuple[str, int], Mapping[str, Any]]) -> None:
    _check_common(configs, "network")
    if configs[("RE", 1001)]["network"] != {
        "N": 3,
        "F": 24,
        "num_classes": 10,
    }:
        raise ValueError("formal NASNet must use N=3, F=24, classes=10")


def _check_method(
    configs: Mapping[tuple[str, int], Mapping[str, Any]],
    method: str,
) -> None:
    for seed in SEARCH_SEEDS:
        config = configs[(method, seed)]
        evolution = config["evolution"]
        if (
            int(evolution["population_size"]),
            int(evolution["tournament_size"]),
            int(evolution["budget"]),
        ) != (20, 5, 60):
            raise ValueError(f"{method} seed {seed} has invalid P/S/B")
        if method == "RE":
            if "surrogate" in config or "stability" in config:
                raise ValueError("RE must not have surrogate or stability sections")
        elif method == "SA-RE":
            if int(evolution["candidate_count"]) != 5 or "stability" in config:
                raise ValueError("SA-RE mechanism differs from the freeze")
        else:
            stability = config["stability"]
            if int(evolution["candidate_count"]) != 5:
                raise ValueError("RS-SA-RE candidate_count must equal 5")
            if (
                int(stability["warmup_pairs"]),
                int(stability["repeat_interval"]),
                float(stability["lambda"]),
            ) != (4, 4, 1.0):
                raise ValueError("RS-SA-RE stability policy differs from the freeze")


def _check_seed_manifest(root: Path) -> dict[str, Any]:
    manifest = _load_yaml(
        root / "configs" / "formal" / "matched_seed_manifest.yaml"
    )
    if manifest["status"] != "frozen":
        raise ValueError("matched seed manifest is not frozen")
    if manifest["methods"] != list(METHODS):
        raise ValueError("matched seed manifest methods differ")
    if manifest["search_seeds"] != list(SEARCH_SEEDS):
        raise ValueError("matched seed manifest seeds differ")
    if int(manifest["expected_total_runs"]) != 30:
        raise ValueError("matched seed manifest must describe 30 runs")
    return manifest


def _check_rng(manifest: Mapping[str, Any]) -> None:
    rng = manifest["rng_namespaces"]
    if rng["search"]["shared_across_methods"] is not True:
        raise ValueError("search RNG is not matched across methods")
    if int(rng["training"]["base"]) != 20260827:
        raise ValueError("training seed base differs")
    if int(rng["surrogate"]["offset"]) != 900000:
        raise ValueError("surrogate seed offset differs")
    if int(rng["repeat"]["offset"]) != 1800000:
        raise ValueError("repeat seed offset differs")
    if rng["surrogate"]["advances_search_rng"] is not False:
        raise ValueError("surrogate RNG must not advance search RNG")
    if rng["repeat"]["advances_search_rng"] is not False:
        raise ValueError("repeat RNG must not advance search RNG")
    expected_runs = [
        {
            "search_seed": seed,
            "surrogate_seed": 900000 + seed,
            "repeat_seed": 1800000 + seed,
        }
        for seed in SEARCH_SEEDS
    ]
    if manifest["runs"] != expected_runs:
        raise ValueError("derived RNG namespace table differs")


def _check_budget(configs: Mapping[tuple[str, int], Mapping[str, Any]]) -> None:
    expected = {
        "RE": (60, 60, 0),
        "SA-RE": (60, 60, 0),
        "RS-SA-RE": (60, 49, 11),
    }
    for (method, seed), config in configs.items():
        audit = config["audit_expectations"]
        observed = (
            int(audit["real_training_runs"]),
            int(audit["first_evaluations"]),
            int(audit["repeat_evaluations"]),
        )
        if observed != expected[method]:
            raise ValueError(f"budget audit differs for {method} seed {seed}")
        if sum(observed[1:]) != observed[0]:
            raise ValueError(f"budget does not balance for {method} seed {seed}")


def _check_run_manifest(root: Path) -> None:
    path = root / "experiments" / "formal" / "manifest.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 30:
        raise ValueError("formal run manifest must contain 30 rows")
    for row in rows:
        config = root / row["config_path"]
        if not config.is_file():
            raise ValueError(f"manifest config is missing: {config}")
        if row["status"] not in {
            "pending",
            "running",
            "completed",
            "failed",
            "audited",
        }:
            raise ValueError("formal run manifest has invalid status")


def _check_output_contract_sources(root: Path) -> None:
    required = (
        "scripts/run_formal.py",
        "tests/test_formal_output_contract.py",
        "tests/test_formal_runner.py",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError("missing output-contract files: " + ", ".join(missing))


def _check_generator(root: Path) -> None:
    result = subprocess.run(
        [sys.executable, "scripts/generate_formal_configs.py", "--check"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stdout + result.stderr)


def _check_rs_formal_runner(root: Path) -> None:
    for seed in SEARCH_SEEDS:
        config = _config_path(root, "RS-SA-RE", seed).relative_to(root)
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_rs_sa_re.py",
                "--config",
                str(config),
                "--validate-only",
            ],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stdout + result.stderr)


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_preflight(
    project_root: Path = PROJECT_ROOT,
    *,
    validate_runner_imports: bool = True,
    write_report: bool = True,
) -> dict[str, Any]:
    root = project_root.resolve()
    configs = _check_config_set(root)
    seed_manifest = _check_seed_manifest(root)
    checks: list[tuple[str, Callable[[], None]]] = [
        ("Dataset consistency", lambda: _check_dataset(configs)),
        ("Trainer consistency", lambda: _check_trainer(configs)),
        ("NASNet consistency", lambda: _check_nasnet(configs)),
        ("RE config", lambda: _check_method(configs, "RE")),
        ("SA-RE config", lambda: _check_method(configs, "SA-RE")),
        ("RS-SA-RE config", lambda: _check_method(configs, "RS-SA-RE")),
        ("Matched seed manifest", lambda: None),
        ("Budget semantics", lambda: _check_budget(configs)),
        ("RNG namespace audit", lambda: _check_rng(seed_manifest)),
        ("Formal run manifest", lambda: _check_run_manifest(root)),
        ("Output contract", lambda: _check_output_contract_sources(root)),
        ("Immutable artifact check", lambda: _check_generator(root)),
    ]
    if validate_runner_imports:
        checks.append(("RS-SA-RE formal runner", lambda: _check_rs_formal_runner(root)))

    results: dict[str, str] = {}
    failures: list[str] = []
    for label, check in checks:
        try:
            check()
        except Exception as error:
            results[label] = f"FAIL: {type(error).__name__}: {error}"
            failures.append(label)
            print(f"{label}: FAIL")
        else:
            results[label] = "PASS"
            print(f"{label}: PASS")

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "formal_config_count": 30,
        "checks": results,
        "status": "FAIL" if failures else "PASS",
        "failures": failures,
    }
    if write_report:
        _write_report(
            root / "experiments" / "formal" / "preflight_report.json",
            report,
        )
    print(f"FORMAL PREFLIGHT: {report['status']}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the A1 formal preflight.")
    parser.add_argument(
        "--skip-runner-import-check",
        action="store_true",
        help="Skip importing the real RS runner (tests/minimal source bundles only).",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not write experiments/formal/preflight_report.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_preflight(
        PROJECT_ROOT,
        validate_runner_imports=not args.skip_runner_import_check,
        write_report=not args.no_report,
    )
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
