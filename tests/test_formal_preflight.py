from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from scripts.check_formal_preflight import PROJECT_ROOT, run_preflight


def test_no_gpu_formal_preflight_passes_the_frozen_artifact_set() -> None:
    report = run_preflight(
        PROJECT_ROOT,
        validate_runner_imports=False,
        write_report=False,
    )
    assert report["status"] == "PASS"
    assert report["formal_config_count"] == 30
    assert all(value == "PASS" for value in report["checks"].values())


def test_rs_runner_accepts_formal_b60_config_when_full_source_is_available() -> None:
    pytest.importorskip("src.search.nasnet_rs_sa_re")
    from scripts.run_rs_sa_re import validate_rs_sa_re_config

    path = PROJECT_ROOT / "configs" / "formal" / "rs_sa_re" / "rs_sa_re_1001.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_rs_sa_re_config(config)

    drifted = copy.deepcopy(config)
    drifted["evolution"]["budget"] = 59
    with pytest.raises(ValueError, match="budget"):
        validate_rs_sa_re_config(drifted)
