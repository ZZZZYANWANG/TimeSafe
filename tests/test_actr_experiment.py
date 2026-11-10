from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


EXPERIMENT_DIR = Path(__file__).parents[1] / "experiments" / "actr20_public"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

common = importlib.import_module("common")
audit_plan = importlib.import_module("audit_plan")
run_audits = importlib.import_module("run_audits")
assemble_report = importlib.import_module("assemble_report")
run_all = importlib.import_module("run_all")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_manifest_fixture(root: Path) -> None:
    (root / "src").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "LICENSE_RESEARCH_ONLY.md").write_text("research only\n", encoding="utf-8")
    (root / "data" / "actr_deidentified_benchmark.npz").write_bytes(b"fixture")
    verification = {
        "status": "PASS",
        "package": {"benchmark_sha256": common.EXPECTED_BENCHMARK_SHA256},
    }
    (root / "PACKAGE_VERIFICATION.json").write_text(
        json.dumps(verification), encoding="utf-8"
    )
    listed = [
        "LICENSE_RESEARCH_ONLY.md",
        "PACKAGE_VERIFICATION.json",
        "data/actr_deidentified_benchmark.npz",
    ]
    lines = [f"{_sha(root / relative)}  ./{relative}" for relative in listed]
    (root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_external_manifest_verification_detects_tampering(tmp_path):
    _make_manifest_fixture(tmp_path)
    result = common.verify_sha256_manifest(tmp_path)
    assert result["status"] == "PASS"
    assert result["checked_files"] == 3

    (tmp_path / "LICENSE_RESEARCH_ONLY.md").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        common.verify_sha256_manifest(tmp_path)


def test_manifest_rejects_path_traversal(tmp_path):
    _make_manifest_fixture(tmp_path)
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    with (tmp_path / "MANIFEST.sha256").open("a", encoding="utf-8") as handle:
        handle.write(f"{_sha(outside)}  ./../outside.txt\n")
    with pytest.raises(ValueError, match="escapes package root"):
        common.verify_sha256_manifest(tmp_path)


def test_combine_worker_artifacts_enforces_expected_verdicts(tmp_path):
    worker_dir = tmp_path / "audit_workers"
    worker_dir.mkdir()
    for index, name in enumerate(audit_plan.AUDIT_NAMES):
        status = sorted(audit_plan.EXPECTED[name])[0]
        payload = {
            "audit": name,
            "audit_asset_count": 24,
            "audit_rows": 1000,
            "checkpoints": [1000, 1200],
            "result": {"status": status},
        }
        (worker_dir / f"{name}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    combined = run_audits.combine_worker_artifacts(tmp_path)
    assert combined["all_expected"] is True
    assert set(combined["observed_verdicts"]) == set(audit_plan.AUDIT_NAMES)

    bad = json.loads((worker_dir / "full_sample_negative_control.json").read_text())
    bad["result"]["status"] = "PASS"
    (worker_dir / "full_sample_negative_control.json").write_text(
        json.dumps(bad), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="Unexpected ACTR TimeSafe verdicts"):
        run_audits.combine_worker_artifacts(tmp_path)


def test_dependency_light_report_assembly(tmp_path):
    pd.DataFrame(
        {
            "model": ["Strong CS", "ACTR benchmark"],
            "rankic": [0.05, 0.07],
            "hac_p": [0.04, 0.002],
        }
    ).to_csv(tmp_path / "benchmark_summary.csv", index=False)
    pd.DataFrame(
        {
            "leader": ["ACTR benchmark"],
            "other": ["Strong CS"],
            "mean_difference": [0.02],
        }
    ).to_csv(tmp_path / "benchmark_pairs.csv", index=False)
    config = {
        "selected_anchor_alpha": 0.45,
        "selected_residual_gamma": 0.10,
        "data_sha256": "abc",
        "rows": 109499,
        "sessions": 1810,
        "external_manifest_verification": {
            "status": "PASS",
            "checked_files": 75,
        },
    }
    (tmp_path / "benchmark_config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    verification = {
        "status": "PASS",
        "max_abs_rankic_difference": 0.0,
        "actr_hac_p_difference": 0.0,
    }
    (tmp_path / "benchmark_verification.json").write_text(
        json.dumps(verification), encoding="utf-8"
    )
    audits = {}
    for name in audit_plan.AUDIT_NAMES:
        audits[name] = {
            "status": sorted(audit_plan.EXPECTED[name])[0],
            "checked_checkpoints": 4,
            "eligible_checkpoints": 900,
            "total_changed_cells": 0,
            "total_structural_differences": 0,
        }
    (tmp_path / "timesafe_audits.json").write_text(
        json.dumps(audits), encoding="utf-8"
    )

    output = assemble_report.assemble(tmp_path, tmp_path / "report.md")
    text = output.read_text(encoding="utf-8")
    assert "External manifest status: `PASS`" in text
    assert "ACTR mature-feedback prefix audit" in text
    assert "ACTR benchmark" in text


def test_orchestrator_refuses_stale_output_without_overwrite(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "stale.txt").write_text("stale", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--overwrite"):
        run_all._prepare_output(output, overwrite=False)
    run_all._prepare_output(output, overwrite=True)
    assert (output / "logs").is_dir()
    assert (output / "audit_workers").is_dir()
    assert not (output / "stale.txt").exists()
