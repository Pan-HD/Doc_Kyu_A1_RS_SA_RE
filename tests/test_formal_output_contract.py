from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from scripts.check_rs_sa_re_smoke import audit_rs_sa_re_formal
from scripts.run_formal import REQUIRED_OUTPUTS, validate_output_contract


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_contract_directory(path: Path, method: str) -> None:
    path.mkdir()
    (path / "config.yaml").write_text(
        yaml.safe_dump({"experiment": {"method": method}}),
        encoding="utf-8",
    )
    _write_csv(
        path / "evaluations.csv",
        ("evaluation_index", "fitness"),
        [{"evaluation_index": 1, "fitness": 0.8}],
    )
    (path / "history.json").write_text(
        json.dumps({"final_population": [{}]}),
        encoding="utf-8",
    )
    (path / "run.log").write_text("evaluation 1 complete\n", encoding="utf-8")
    (path / "best.json").write_text(
        json.dumps({"metric": "fitness", "value": 0.8}),
        encoding="utf-8",
    )
    if method in {"SA-RE", "RS-SA-RE"}:
        _write_csv(
            path / "candidate_predictions.csv",
            ("candidate_index", "selected"),
            [{"candidate_index": 0, "selected": True}],
        )
    if method == "RS-SA-RE":
        _write_csv(
            path / "repeat_evaluations.csv",
            ("base_evaluation_index", "repeat_phase"),
            [{"base_evaluation_index": 1, "repeat_phase": "warmup"}],
        )


@pytest.mark.parametrize("method", ("RE", "SA-RE", "RS-SA-RE"))
def test_method_output_contract_accepts_exact_required_files(
    tmp_path: Path,
    method: str,
) -> None:
    output = tmp_path / method.lower().replace("-", "_")
    _write_contract_directory(output, method)
    validate_output_contract(output, method)
    assert REQUIRED_OUTPUTS[method].issubset(
        {item.name for item in output.iterdir()}
    )


def test_output_contract_rejects_a_missing_method_specific_file(
    tmp_path: Path,
) -> None:
    output = tmp_path / "rs_sa_re_1001"
    _write_contract_directory(output, "RS-SA-RE")
    (output / "repeat_evaluations.csv").unlink()
    with pytest.raises(ValueError, match="repeat_evaluations.csv"):
        validate_output_contract(output, "RS-SA-RE")


def test_formal_rs_audit_enforces_49_first_plus_11_repeats(
    tmp_path: Path,
) -> None:
    output = tmp_path / "rs_sa_re_1001"
    output.mkdir()
    config = {
        "experiment": {"method": "RS-SA-RE", "mode": "formal"},
        "audit_expectations": {
            "real_training_runs": 60,
            "first_evaluations": 49,
            "repeat_evaluations": 11,
            "candidate_rows": 145,
            "selected_rows": 29,
            "final_population": 20,
        },
    }
    (output / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    _write_csv(
        output / "evaluations.csv",
        ("evaluation_index", "fitness"),
        [
            {"evaluation_index": index, "fitness": 0.7 + index / 1000}
            for index in range(1, 50)
        ],
    )
    _write_csv(
        output / "repeat_evaluations.csv",
        ("base_evaluation_index", "repeat_phase"),
        [
            {
                "base_evaluation_index": index,
                "repeat_phase": "warmup" if index <= 4 else "periodic",
            }
            for index in range(1, 12)
        ],
    )
    _write_csv(
        output / "candidate_predictions.csv",
        ("evolution_step", "candidate_index", "selected"),
        [
            {
                "evolution_step": step,
                "candidate_index": candidate,
                "selected": candidate == 0,
            }
            for step in range(1, 30)
            for candidate in range(5)
        ],
    )
    (output / "history.json").write_text(
        json.dumps({"final_population": [{} for _ in range(20)]}),
        encoding="utf-8",
    )

    audit = audit_rs_sa_re_formal(output)
    assert audit.real_training_runs == 60
    assert audit.first_evaluations == 49
    assert audit.repeat_evaluations == 11
    assert audit.warmup_repeats == 4
    assert audit.periodic_repeats == 7
    assert audit.candidate_rows == 145
    assert audit.selected_rows == 29
    assert audit.final_population_size == 20
