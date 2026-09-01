"""Launch one immutable formal run and maintain the formal run manifest.

This wrapper never changes a formal YAML file and never supplies seed
overrides.  It records lifecycle state atomically, captures line-flushed
stdout/stderr, verifies the output contract, and records audited counts.
"""

from __future__ import annotations

import math
import argparse
import contextlib
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_RELATIVE_PATH = Path("experiments/formal/manifest.csv")
VALID_STATUSES = {"pending", "running", "completed", "failed", "audited"}
MANIFEST_FIELDS = (
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
RUNNERS = {
    "RE": Path("scripts/run_re.py"),
    "SA-RE": Path("scripts/run_sa_re.py"),
    "RS-SA-RE": Path("scripts/run_rs_sa_re.py"),
}
REQUIRED_OUTPUTS = {
    "RE": {
        "config.yaml",
        "evaluations.csv",
        "history.json",
        "run.log",
        "best.json",
    },
    "SA-RE": {
        "config.yaml",
        "evaluations.csv",
        "history.json",
        "run.log",
        "best.json",
        "candidate_predictions.csv",
    },
    "RS-SA-RE": {
        "config.yaml",
        "evaluations.csv",
        "history.json",
        "run.log",
        "best.json",
        "candidate_predictions.csv",
        "repeat_evaluations.csv",
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextlib.contextmanager
def _manifest_lock(path: Path, timeout_seconds: float = 10.0) -> Iterator[None]:
    lock_path = path.with_suffix(path.suffix + ".lock")
    deadline = time.monotonic() + timeout_seconds
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"timed out waiting for manifest lock: {lock_path}")
            time.sleep(0.05)
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise ValueError("formal manifest has an unexpected schema")
        rows = [dict(row) for row in reader]
    if len(rows) != 30:
        raise ValueError("formal manifest must contain exactly 30 rows")
    keys = [(row["method"], row["search_seed"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("formal manifest contains duplicate run keys")
    if any(row["status"] not in VALID_STATUSES for row in rows):
        raise ValueError("formal manifest contains an invalid status")
    return rows


def _manifest_text(rows: list[dict[str, str]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=MANIFEST_FIELDS,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def update_manifest_row(
    path: Path,
    *,
    method: str,
    search_seed: int,
    expected_status: str | None,
    changes: Mapping[str, Any],
) -> dict[str, str]:
    unknown = set(changes) - set(MANIFEST_FIELDS)
    if unknown:
        raise ValueError(f"unknown manifest fields: {sorted(unknown)}")
    with _manifest_lock(path):
        rows = read_manifest(path)
        matches = [
            row
            for row in rows
            if row["method"] == method
            and int(row["search_seed"]) == search_seed
        ]
        if len(matches) != 1:
            raise ValueError("formal run key is absent or duplicated in manifest")
        row = matches[0]
        if expected_status is not None and row["status"] != expected_status:
            raise RuntimeError(
                f"manifest status must be {expected_status}, observed {row['status']}"
            )
        for name, value in changes.items():
            row[name] = "" if value is None else str(value)
        if row["status"] not in VALID_STATUSES:
            raise ValueError(f"invalid manifest status: {row['status']}")
        _atomic_write_text(path, _manifest_text(rows))
        return dict(row)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no CSV header")
        return [dict(row) for row in reader]


def _write_best_if_missing(output_dir: Path) -> None:
    best_path = output_dir / "best.json"
    if best_path.exists():
        return

    rows = _read_csv_rows(output_dir / "evaluations.csv")
    if not rows:
        raise ValueError("cannot derive best.json from an empty evaluations.csv")

    metric_names = (
        "fitness",
        "accuracy",
        "validation_accuracy",
        "final_validation_accuracy",
        "accuracy_epoch_5",
    )
    metric_name = next(
        (name for name in metric_names if name in rows[0]),
        None,
    )
    if metric_name is None:
        raise ValueError("evaluations.csv has no recognized fitness column")

    scored_rows: list[tuple[float, dict[str, str]]] = []
    for row in rows:
        # RS-SA-RE repeat evaluations consume budget but must never replace
        # the first-seed population fitness or become the reported best.
        if str(row.get("event_type", "")).strip() == "repeat_evaluation":
            continue

        try:
            metric_value = float(row[metric_name])
        except (TypeError, ValueError) as error:
            raise ValueError(
                "a non-repeat evaluation contains an invalid fitness value"
            ) from error

        if not math.isfinite(metric_value):
            raise ValueError(
                "a non-repeat evaluation contains a non-finite fitness value"
            )

        scored_rows.append((metric_value, row))

    if not scored_rows:
        raise ValueError(
            "evaluations.csv contains no valid non-repeat fitness values"
        )

    metric_value, best_row = max(scored_rows, key=lambda item: item[0])
    payload = {
        "metric": metric_name,
        "value": metric_value,
        "evaluation": best_row,
    }
    _atomic_write_text(
        best_path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def _materialize_config_if_missing(
    output_dir: Path,
    config: Mapping[str, Any],
) -> None:
    path = output_dir / "config.yaml"
    if path.exists():
        observed = _load_yaml(path)
        if observed != dict(config):
            raise ValueError("output config.yaml differs from the frozen config")
        return
    _atomic_write_text(
        path,
        yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True),
    )


def validate_output_contract(output_dir: Path, method: str) -> None:
    if method not in REQUIRED_OUTPUTS:
        raise ValueError(f"unsupported formal method: {method}")
    missing = sorted(
        name
        for name in REQUIRED_OUTPUTS[method]
        if not (output_dir / name).is_file()
    )
    if missing:
        raise ValueError(f"formal output contract missing: {', '.join(missing)}")
    for name in REQUIRED_OUTPUTS[method]:
        if (output_dir / name).stat().st_size == 0:
            raise ValueError(f"formal output artifact is empty: {name}")
    _load_yaml(output_dir / "config.yaml")
    _read_csv_rows(output_dir / "evaluations.csv")
    json.loads((output_dir / "history.json").read_text(encoding="utf-8"))
    json.loads((output_dir / "best.json").read_text(encoding="utf-8"))
    if method in {"SA-RE", "RS-SA-RE"}:
        _read_csv_rows(output_dir / "candidate_predictions.csv")
    if method == "RS-SA-RE":
        _read_csv_rows(output_dir / "repeat_evaluations.csv")


def audit_output_directory(
    output_dir: Path,
    method: str,
    config: Mapping[str, Any],
) -> dict[str, int]:
    expectations = config["audit_expectations"]
    if method == "RS-SA-RE":
        from scripts.check_rs_sa_re_smoke import audit_rs_sa_re_formal

        audit = audit_rs_sa_re_formal(output_dir)
        return {
            "real_training_runs": audit.real_training_runs,
            "first_evaluations": audit.first_evaluations,
            "repeat_evaluations": audit.repeat_evaluations,
        }

    first_evaluations = len(_read_csv_rows(output_dir / "evaluations.csv"))
    expected_first = int(expectations["first_evaluations"])
    expected_real = int(expectations["real_training_runs"])
    expected_repeats = int(expectations["repeat_evaluations"])
    if first_evaluations != expected_first:
        raise ValueError(
            f"first evaluations: expected {expected_first}, observed {first_evaluations}"
        )
    if expected_repeats != 0 or expected_real != first_evaluations:
        raise ValueError("RE/SA-RE budget expectations are inconsistent")
    if method == "SA-RE":
        candidate_rows = len(
            _read_csv_rows(output_dir / "candidate_predictions.csv")
        )
        if candidate_rows != int(expectations["candidate_rows"]):
            raise ValueError("SA-RE candidate row count is incorrect")
    return {
        "real_training_runs": expected_real,
        "first_evaluations": first_evaluations,
        "repeat_evaluations": 0,
    }


def _append_log_destination(temporary_log: Path, output_dir: Path) -> None:
    if not output_dir.is_dir():
        return
    destination = output_dir / "run.log"
    temporary_text = temporary_log.read_text(encoding="utf-8")
    existing_text = (
        destination.read_text(encoding="utf-8")
        if destination.exists()
        else ""
    )
    separator = "" if not existing_text or existing_text.endswith("\n") else "\n"
    _atomic_write_text(destination, existing_text + separator + temporary_text)
    temporary_log.unlink(missing_ok=True)


def _validate_formal_identity(
    config_path: Path,
    config: Mapping[str, Any],
    project_root: Path,
) -> tuple[str, int, Path, str]:
    experiment = config["experiment"]
    method = str(experiment["method"])
    if method not in RUNNERS:
        raise ValueError(f"unsupported formal method: {method}")
    if str(experiment["mode"]).lower() != "formal":
        raise ValueError("run_formal.py accepts only mode=formal configs")
    if str(experiment.get("status", "")).lower() != "frozen":
        raise ValueError("formal config status must be frozen")
    if bool(experiment.get("do_not_run", True)):
        raise ValueError("formal config is marked do_not_run")
    if bool(experiment.get("overwrite", True)):
        raise ValueError("formal overwrite must be false")
    search_seed = int(experiment["search_seed"])
    if search_seed not in range(1001, 1011):
        raise ValueError("formal search_seed must be in 1001..1010")
    output_relative = Path(str(experiment["output_dir"]))
    if output_relative.is_absolute() or "{" in output_relative.as_posix():
        raise ValueError("formal output_dir must be a concrete relative path")
    output_dir = (project_root / output_relative).resolve()
    formal_root = (project_root / "experiments" / "formal").resolve()
    if formal_root not in output_dir.parents:
        raise ValueError("formal output_dir must be under experiments/formal")
    config_relative = config_path.resolve().relative_to(project_root.resolve()).as_posix()
    return method, search_seed, output_dir, config_relative


def execute_formal_run(
    config_path: Path,
    *,
    project_root: Path = PROJECT_ROOT,
    runner_commands: Mapping[str, Sequence[str]] | None = None,
    dry_run: bool = False,
) -> int:
    project_root = project_root.resolve()
    config_path = config_path.resolve()
    config = _load_yaml(config_path)
    method, search_seed, output_dir, config_relative = _validate_formal_identity(
        config_path,
        config,
        project_root,
    )
    manifest_path = project_root / MANIFEST_RELATIVE_PATH
    rows = read_manifest(manifest_path)
    row = next(
        (
            item
            for item in rows
            if item["method"] == method
            and int(item["search_seed"]) == search_seed
        ),
        None,
    )
    if row is None:
        raise ValueError("formal config has no manifest row")
    if row["config_path"] != config_relative:
        raise ValueError("formal config path differs from manifest")
    if row["output_directory"] != config["experiment"]["output_dir"]:
        raise ValueError("formal output directory differs from manifest")
    if row["status"] != "pending":
        raise RuntimeError(f"formal manifest row is not pending: {row['status']}")
    if output_dir.exists():
        raise FileExistsError(
            f"formal output directory already exists; refusing overwrite: {output_dir}"
        )

    command = list(
        (runner_commands or {}).get(
            method,
            (
                sys.executable,
                str(project_root / RUNNERS[method]),
                "--config",
                config_relative,
            ),
        )
    )
    if dry_run:
        print("DRY RUN " + subprocess.list2cmdline(command))
        return 0

    start_time = _utc_now()
    update_manifest_row(
        manifest_path,
        method=method,
        search_seed=search_seed,
        expected_status="pending",
        changes={
            "status": "running",
            "start_time": start_time,
            "end_time": "",
            "exit_code": "",
            "audit_status": "pending",
            "notes": "",
        },
    )

    log_dir = project_root / "experiments" / "formal" / ".logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    temporary_log = log_dir / f"{method.lower().replace('-', '_')}_{search_seed}.log"
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    exit_code: int | None = None
    try:
        with temporary_log.open("w", encoding="utf-8", newline="") as log:
            log.write(
                f"FORMAL START method={method} search_seed={search_seed} "
                f"time={start_time}\n"
            )
            log.flush()
            os.fsync(log.fileno())
            process = subprocess.Popen(
                command,
                cwd=project_root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                os.fsync(log.fileno())
            exit_code = process.wait()
            log.write(
                f"FORMAL END method={method} search_seed={search_seed} "
                f"exit_code={exit_code} time={_utc_now()}\n"
            )
            log.flush()
            os.fsync(log.fileno())
    except Exception as error:
        _append_log_destination(temporary_log, output_dir)
        update_manifest_row(
            manifest_path,
            method=method,
            search_seed=search_seed,
            expected_status="running",
            changes={
                "status": "failed",
                "end_time": _utc_now(),
                "exit_code": exit_code,
                "audit_status": "not_run",
                "notes": f"launcher error: {type(error).__name__}: {error}",
            },
        )
        raise

    _append_log_destination(temporary_log, output_dir)
    if exit_code != 0:
        update_manifest_row(
            manifest_path,
            method=method,
            search_seed=search_seed,
            expected_status="running",
            changes={
                "status": "failed",
                "end_time": _utc_now(),
                "exit_code": exit_code,
                "audit_status": "not_run",
                "notes": "runner returned a non-zero exit code",
            },
        )
        return int(exit_code)

    try:
        if not output_dir.is_dir():
            raise FileNotFoundError("runner did not create its formal output directory")
        _materialize_config_if_missing(output_dir, config)
        _write_best_if_missing(output_dir)
        validate_output_contract(output_dir, method)
        counts = audit_output_directory(output_dir, method, config)
    except Exception as error:
        update_manifest_row(
            manifest_path,
            method=method,
            search_seed=search_seed,
            expected_status="running",
            changes={
                "status": "completed",
                "end_time": _utc_now(),
                "exit_code": 0,
                "audit_status": "failed",
                "notes": f"audit error: {type(error).__name__}: {error}",
            },
        )
        raise

    update_manifest_row(
        manifest_path,
        method=method,
        search_seed=search_seed,
        expected_status="running",
        changes={
            "status": "audited",
            "end_time": _utc_now(),
            "real_training_runs": counts["real_training_runs"],
            "first_evaluations": counts["first_evaluations"],
            "repeat_evaluations": counts["repeat_evaluations"],
            "exit_code": 0,
            "audit_status": "passed",
            "notes": "",
        },
    )
    print(
        "FORMAL RUN: AUDITED "
        f"method={method} search_seed={search_seed} output={output_dir}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch exactly one immutable manifest-backed formal run."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate identity and print the command without changing state.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        exit_code = execute_formal_run(
            args.config,
            project_root=PROJECT_ROOT,
            dry_run=args.dry_run,
        )
    except Exception as error:
        print(f"FORMAL RUN: FAILED {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
