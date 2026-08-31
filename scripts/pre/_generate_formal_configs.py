"""Generate and verify the immutable A1 formal configuration set.

The generator intentionally derives the 30 per-run files from the reviewed
pre-freeze method templates.  It never overwrites a differing formal artifact:
once a file exists, drift is an error rather than an implicit re-freeze.
"""

from __future__ import annotations

import argparse
import copy
import csv
import io
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMAL_DIR = PROJECT_ROOT / "configs" / "formal"
RUN_MANIFEST_PATH = PROJECT_ROOT / "experiments" / "formal" / "manifest.csv"
MATCHED_SEED_MANIFEST_PATH = FORMAL_DIR / "matched_seed_manifest.yaml"

FREEZE_DATE = "2026-08-31"
SEARCH_SEEDS = tuple(range(1001, 1011))
SURROGATE_SEED_OFFSET = 900_000
REPEAT_SEED_OFFSET = 1_800_000
TRAINING_SEED_BASE = 20_260_827

METHODS: Mapping[str, Mapping[str, str]] = {
    "RE": {
        "slug": "re",
        "template": "re.yaml",
        "runner": "scripts/run_re.py",
    },
    "SA-RE": {
        "slug": "sa_re",
        "template": "sa_re.yaml",
        "runner": "scripts/run_sa_re.py",
    },
    "RS-SA-RE": {
        "slug": "rs_sa_re",
        "template": "rs_sa_re.yaml",
        "runner": "scripts/run_rs_sa_re.py",
    },
}

RUN_MANIFEST_FIELDS = (
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


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def _load_source_templates() -> dict[str, dict[str, Any]]:
    templates = {
        method: _read_yaml(FORMAL_DIR / values["template"])
        for method, values in METHODS.items()
    }
    reference = templates["RE"]
    for method, config in templates.items():
        experiment = config["experiment"]
        if experiment["method"] != method:
            raise ValueError(f"source template method mismatch for {method}")
        if experiment["mode"] != "formal":
            raise ValueError(f"source template mode mismatch for {method}")
        for section in ("dataset", "network", "training", "implementation", "device"):
            if config[section] != reference[section]:
                raise ValueError(f"{method} source template drifts in {section}")
        evolution = config["evolution"]
        expected = {"population_size": 20, "tournament_size": 5, "budget": 60}
        for field, value in expected.items():
            if int(evolution[field]) != value:
                raise ValueError(f"{method} source template has invalid {field}")
        if int(config["training"]["training_seed_base"]) != TRAINING_SEED_BASE:
            raise ValueError(f"{method} source template has invalid training seed base")
    return templates


def build_formal_config(
    method: str,
    search_seed: int,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"unsupported method: {method}")
    if search_seed not in SEARCH_SEEDS:
        raise ValueError(f"unsupported formal search seed: {search_seed}")

    config = copy.deepcopy(dict(source))
    slug = METHODS[method]["slug"]
    experiment = config["experiment"]
    experiment.update(
        {
            "name": f"nasnet_{slug}_formal_{search_seed}",
            "method": method,
            "mode": "formal",
            "status": "frozen",
            "do_not_run": False,
            "freeze_date": FREEZE_DATE,
            "search_seed": search_seed,
            "seed_manifest": "configs/formal/matched_seed_manifest.yaml",
            "output_dir": f"experiments/formal/{slug}_{search_seed}",
            "overwrite": False,
        }
    )

    if method == "RS-SA-RE":
        config["stability"]["repeat_seed"] = REPEAT_SEED_OFFSET + search_seed

    return config


def build_matched_seed_manifest() -> dict[str, Any]:
    runs = []
    for search_seed in SEARCH_SEEDS:
        runs.append(
            {
                "search_seed": search_seed,
                "surrogate_seed": SURROGATE_SEED_OFFSET + search_seed,
                "repeat_seed": REPEAT_SEED_OFFSET + search_seed,
            }
        )
    return {
        "schema_version": 1,
        "status": "frozen",
        "freeze_date": FREEZE_DATE,
        "matched_seed_policy": "same_search_seed_across_RE_SA_RE_RS_SA_RE",
        "methods": list(METHODS),
        "search_seeds": list(SEARCH_SEEDS),
        "expected_runs_per_method": len(SEARCH_SEEDS),
        "expected_total_runs": len(SEARCH_SEEDS) * len(METHODS),
        "rng_namespaces": {
            "search": {
                "derivation": "search_seed",
                "shared_across_methods": True,
            },
            "training": {
                "base": TRAINING_SEED_BASE,
                "derivation": "existing_evaluator_replica_schedule",
                "replica_one_preserves_first_training_schedule": True,
            },
            "surrogate": {
                "offset": SURROGATE_SEED_OFFSET,
                "derivation": "surrogate_seed_offset + search_seed",
                "advances_search_rng": False,
            },
            "repeat": {
                "offset": REPEAT_SEED_OFFSET,
                "derivation": "repeat_seed_offset + search_seed",
                "advances_search_rng": False,
            },
        },
        "runs": runs,
    }


def build_run_manifest_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for search_seed in SEARCH_SEEDS:
        for method, values in METHODS.items():
            slug = values["slug"]
            rows.append(
                {
                    "method": method,
                    "search_seed": str(search_seed),
                    "config_path": f"configs/formal/{slug}/{slug}_{search_seed}.yaml",
                    "status": "pending",
                    "start_time": "",
                    "end_time": "",
                    "output_directory": f"experiments/formal/{slug}_{search_seed}",
                    "real_training_runs": "",
                    "first_evaluations": "",
                    "repeat_evaluations": "",
                    "exit_code": "",
                    "audit_status": "pending",
                    "notes": "",
                }
            )
    return rows


def _yaml_text(value: Mapping[str, Any]) -> str:
    return yaml.safe_dump(
        dict(value),
        sort_keys=False,
        allow_unicode=True,
        width=88,
    )


def _csv_text(rows: list[dict[str, str]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=RUN_MANIFEST_FIELDS,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def expected_artifacts() -> dict[Path, str]:
    templates = _load_source_templates()
    artifacts: dict[Path, str] = {
        MATCHED_SEED_MANIFEST_PATH: _yaml_text(build_matched_seed_manifest()),
        RUN_MANIFEST_PATH: _csv_text(build_run_manifest_rows()),
    }
    for method, values in METHODS.items():
        slug = values["slug"]
        for search_seed in SEARCH_SEEDS:
            path = FORMAL_DIR / slug / f"{slug}_{search_seed}.yaml"
            artifacts[path] = _yaml_text(
                build_formal_config(method, search_seed, templates[method])
            )
    return artifacts


def _write_without_drift(path: Path, text: str) -> str:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(
                f"refusing to overwrite differing frozen artifact: {path}"
            )
        return "unchanged"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")
    return "created"


def write_artifacts() -> None:
    counts = {"created": 0, "unchanged": 0}
    for path, text in expected_artifacts().items():
        counts[_write_without_drift(path, text)] += 1
    print(
        "formal artifacts: PASS "
        f"created={counts['created']} unchanged={counts['unchanged']} total={sum(counts.values())}"
    )


def check_artifacts() -> None:
    missing: list[Path] = []
    changed: list[Path] = []
    for path, expected in expected_artifacts().items():
        if not path.exists():
            missing.append(path)
        elif path.read_text(encoding="utf-8") != expected:
            changed.append(path)
    if missing or changed:
        details = [f"missing: {path}" for path in missing]
        details.extend(f"changed: {path}" for path in changed)
        raise RuntimeError("formal artifact check failed\n" + "\n".join(details))
    print("formal artifacts: PASS checked=32 configs=30")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or verify the frozen A1 formal artifacts."
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--write",
        action="store_true",
        help="Create missing artifacts; refuse to overwrite any differing file.",
    )
    action.add_argument(
        "--check",
        action="store_true",
        help="Verify all generated artifacts (default).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write:
        write_artifacts()
    else:
        check_artifacts()


if __name__ == "__main__":
    main()
