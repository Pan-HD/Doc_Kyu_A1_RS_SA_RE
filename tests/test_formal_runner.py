from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import yaml

from scripts.run_formal import MANIFEST_FIELDS, execute_formal_run, read_manifest


METHODS = {"RE": "re", "SA-RE": "sa_re", "RS-SA-RE": "rs_sa_re"}


def _write_manifest(root: Path) -> None:
    path = root / "experiments" / "formal" / "manifest.csv"
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=MANIFEST_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        for seed in range(1001, 1011):
            for method, slug in METHODS.items():
                writer.writerow(
                    {
                        "method": method,
                        "search_seed": seed,
                        "config_path": f"configs/formal/{slug}/{slug}_{seed}.yaml",
                        "status": "pending",
                        "start_time": "",
                        "end_time": "",
                        "output_directory": f"experiments/formal/{slug}_{seed}",
                        "real_training_runs": "",
                        "first_evaluations": "",
                        "repeat_evaluations": "",
                        "exit_code": "",
                        "audit_status": "pending",
                        "notes": "",
                    }
                )


def _write_config(root: Path, method: str, seed: int, evaluations: int = 2) -> Path:
    slug = METHODS[method]
    path = root / "configs" / "formal" / slug / f"{slug}_{seed}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": {
            "name": f"test_{slug}_{seed}",
            "method": method,
            "mode": "formal",
            "status": "frozen",
            "do_not_run": False,
            "search_seed": seed,
            "output_dir": f"experiments/formal/{slug}_{seed}",
            "overwrite": False,
        },
        "evolution": {"budget": evaluations},
        "audit_expectations": {
            "real_training_runs": evaluations,
            "first_evaluations": evaluations,
            "repeat_evaluations": 0,
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _write_success_runner(path: Path) -> None:
    path.write_text(
        """from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
output.mkdir(parents=True)
with (output / 'evaluations.csv').open('w', encoding='utf-8', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=('evaluation_index', 'fitness'), lineterminator='\\n')
    writer.writeheader()
    writer.writerow({'evaluation_index': 1, 'fitness': 0.7})
    stream.flush()
    writer.writerow({'evaluation_index': 2, 'fitness': 0.8})
    stream.flush()
(output / 'history.json').write_text(json.dumps({'final_population': [{}, {}]}), encoding='utf-8')
print('evaluation 1 persisted', flush=True)
print('evaluation 2 persisted', flush=True)
""",
        encoding="utf-8",
    )


def _row(root: Path, method: str, seed: int) -> dict[str, str]:
    return next(
        row
        for row in read_manifest(root / "experiments" / "formal" / "manifest.csv")
        if row["method"] == method and int(row["search_seed"]) == seed
    )


def test_formal_launcher_flushes_log_materializes_contract_and_audits(
    tmp_path: Path,
) -> None:
    _write_manifest(tmp_path)
    config_path = _write_config(tmp_path, "RE", 1001)
    fake_runner = tmp_path / "fake_success.py"
    _write_success_runner(fake_runner)
    output = tmp_path / "experiments" / "formal" / "re_1001"

    exit_code = execute_formal_run(
        config_path,
        project_root=tmp_path,
        runner_commands={
            "RE": (sys.executable, str(fake_runner), str(output)),
        },
    )

    assert exit_code == 0
    row = _row(tmp_path, "RE", 1001)
    assert row["status"] == "audited"
    assert row["audit_status"] == "passed"
    assert row["real_training_runs"] == "2"
    assert row["first_evaluations"] == "2"
    assert row["repeat_evaluations"] == "0"
    assert row["exit_code"] == "0"
    assert (output / "config.yaml").is_file()
    assert (output / "best.json").is_file()
    log_text = (output / "run.log").read_text(encoding="utf-8")
    assert "evaluation 1 persisted" in log_text
    assert "evaluation 2 persisted" in log_text
    best = json.loads((output / "best.json").read_text(encoding="utf-8"))
    assert best["value"] == 0.8


def test_formal_launcher_records_nonzero_runner_exit_as_failed(
    tmp_path: Path,
) -> None:
    _write_manifest(tmp_path)
    config_path = _write_config(tmp_path, "SA-RE", 1001)
    fake_runner = tmp_path / "fake_failure.py"
    fake_runner.write_text(
        "import sys\nprint('synthetic failure', flush=True)\nraise SystemExit(3)\n",
        encoding="utf-8",
    )

    exit_code = execute_formal_run(
        config_path,
        project_root=tmp_path,
        runner_commands={"SA-RE": (sys.executable, str(fake_runner))},
    )

    assert exit_code == 3
    row = _row(tmp_path, "SA-RE", 1001)
    assert row["status"] == "failed"
    assert row["exit_code"] == "3"
    assert row["audit_status"] == "not_run"
    assert "non-zero" in row["notes"]


def test_formal_dry_run_does_not_mutate_manifest(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    config_path = _write_config(tmp_path, "RE", 1001)
    manifest = tmp_path / "experiments" / "formal" / "manifest.csv"
    before = manifest.read_bytes()

    assert execute_formal_run(
        config_path,
        project_root=tmp_path,
        runner_commands={"RE": (sys.executable, "unused.py")},
        dry_run=True,
    ) == 0
    assert manifest.read_bytes() == before
    assert _row(tmp_path, "RE", 1001)["status"] == "pending"
