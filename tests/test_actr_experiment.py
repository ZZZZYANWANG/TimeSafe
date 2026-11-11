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


def _worker_identity() -> dict:
    return {
        "artifact_schema_version": common.ARTIFACT_SCHEMA_VERSION,
        "experiment_run_id": "test-run-001",
        "implementation_sha256": "c" * 64,
        "audit_contract_version": common.AUDIT_CONTRACT_VERSION,
        "benchmark_sha256": common.EXPECTED_BENCHMARK_SHA256,
        "actr_manifest_sha256": common.EXPECTED_MANIFEST_SHA256,
        "benchmark_rows": 109499,
        "benchmark_sessions": 1810,
        "timesafe_version": "0.3.1",
        "audit_asset_ids_sha256": "a" * 64,
        "audit_asset_count": 24,
        "audit_rows": 1000,
        "audit_sessions": 100,
        "audit_session_min": 1,
        "audit_session_max": 100,
        "planned_checkpoints": [1000, 1200],
    }


def _write_benchmark_config(root: Path, identity: dict | None = None) -> None:
    identity = identity or _worker_identity()
    payload = {
        "experiment_run_id": identity["experiment_run_id"],
        "implementation_sha256": identity["implementation_sha256"],
        "experiment_contract_version": identity["audit_contract_version"],
        "timesafe_version": identity["timesafe_version"],
        "data_sha256": identity["benchmark_sha256"],
        "actr_manifest_sha256": identity["actr_manifest_sha256"],
        "rows": identity["benchmark_rows"],
        "sessions": identity["benchmark_sessions"],
    }
    (root / "benchmark_config.json").write_text(json.dumps(payload), encoding="utf-8")


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
    result = common.verify_sha256_manifest(
        tmp_path, expected_manifest_sha256=_sha(tmp_path / "MANIFEST.sha256")
    )
    assert result["status"] == "PASS"
    assert result["checked_files"] == 3

    (tmp_path / "LICENSE_RESEARCH_ONLY.md").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        common.verify_sha256_manifest(
            tmp_path, expected_manifest_sha256=_sha(tmp_path / "MANIFEST.sha256")
        )


def test_manifest_rejects_path_traversal(tmp_path):
    _make_manifest_fixture(tmp_path)
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    with (tmp_path / "MANIFEST.sha256").open("a", encoding="utf-8") as handle:
        handle.write(f"{_sha(outside)}  ./../outside.txt\n")
    with pytest.raises(ValueError, match="escapes package root"):
        common.verify_sha256_manifest(
            tmp_path, expected_manifest_sha256=_sha(tmp_path / "MANIFEST.sha256")
        )


def _write_worker_artifacts(root: Path, identity: dict | None = None) -> Path:
    identity = identity or _worker_identity()
    worker_dir = root / "audit_workers"
    worker_dir.mkdir()
    for name in audit_plan.AUDIT_NAMES:
        status = sorted(audit_plan.EXPECTED[name])[0]
        payload = {
            "audit": name,
            "identity": identity,
            "audit_asset_count": identity["audit_asset_count"],
            "audit_rows": identity["audit_rows"],
            "checkpoints": [1000, 1200],
            "result": {"status": status},
        }
        (worker_dir / f"{name}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    return worker_dir


def test_manifest_identity_is_pinned_before_file_validation(tmp_path):
    _make_manifest_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="manifest identity mismatch"):
        common.verify_sha256_manifest(tmp_path)


def test_combine_worker_artifacts_enforces_expected_verdicts(tmp_path):
    identity = _worker_identity()
    _write_benchmark_config(tmp_path, identity)
    worker_dir = _write_worker_artifacts(tmp_path, identity)
    combined = run_audits.combine_worker_artifacts(tmp_path)
    assert combined["all_expected"] is True
    assert combined["identity"] == identity
    assert set(combined["worker_artifact_sha256"]) == set(audit_plan.AUDIT_NAMES)
    assert set(combined["observed_verdicts"]) == set(audit_plan.AUDIT_NAMES)

    bad = json.loads((worker_dir / "full_sample_negative_control.json").read_text())
    bad["result"]["status"] = "PASS"
    (worker_dir / "full_sample_negative_control.json").write_text(
        json.dumps(bad), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="Unexpected ACTR TimeSafe verdicts"):
        run_audits.combine_worker_artifacts(tmp_path)


def test_combine_rejects_mixed_or_stale_worker_identity(tmp_path):
    identity = _worker_identity()
    _write_benchmark_config(tmp_path, identity)
    worker_dir = _write_worker_artifacts(tmp_path, identity)
    path = worker_dir / "legal_refit.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["identity"]["benchmark_sha256"] = "b" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Mixed or stale worker identity"):
        run_audits.combine_worker_artifacts(tmp_path)


def test_combine_rejects_worker_identity_from_different_benchmark_config(tmp_path):
    identity = _worker_identity()
    _write_benchmark_config(tmp_path, {**identity, "benchmark_rows": 1})
    _write_worker_artifacts(tmp_path, identity)
    with pytest.raises(RuntimeError, match="Benchmark config and worker identity disagree"):
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
    identity = _worker_identity()
    config = {
        "selected_anchor_alpha": 0.45,
        "selected_residual_gamma": 0.10,
        "experiment_run_id": identity["experiment_run_id"],
        "implementation_sha256": identity["implementation_sha256"],
        "experiment_contract_version": identity["audit_contract_version"],
        "timesafe_version": identity["timesafe_version"],
        "data_sha256": identity["benchmark_sha256"],
        "actr_manifest_sha256": identity["actr_manifest_sha256"],
        "rows": identity["benchmark_rows"],
        "sessions": identity["benchmark_sessions"],
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
    audits = {"identity": identity, "all_expected": True}
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


def test_report_assembly_rejects_mismatched_benchmark_and_audit_identity(tmp_path):
    pd.DataFrame({"model": ["ACTR benchmark"], "rankic": [0.07], "hac_p": [0.002]}).to_csv(
        tmp_path / "benchmark_summary.csv", index=False
    )
    pd.DataFrame({"leader": [], "other": [], "mean_difference": []}).to_csv(
        tmp_path / "benchmark_pairs.csv", index=False
    )
    identity = _worker_identity()
    config = {
        "selected_anchor_alpha": 0.45,
        "selected_residual_gamma": 0.10,
        "experiment_run_id": identity["experiment_run_id"],
        "implementation_sha256": identity["implementation_sha256"],
        "experiment_contract_version": identity["audit_contract_version"],
        "timesafe_version": identity["timesafe_version"],
        "data_sha256": "b" * 64,
        "rows": identity["benchmark_rows"],
        "sessions": identity["benchmark_sessions"],
        "external_manifest_verification": {"status": "PASS", "checked_files": 75},
    }
    (tmp_path / "benchmark_config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "benchmark_verification.json").write_text(
        json.dumps({"status": "PASS", "max_abs_rankic_difference": 0.0, "actr_hac_p_difference": 0.0}),
        encoding="utf-8",
    )
    audits = {"identity": identity, "all_expected": True}
    for name in audit_plan.AUDIT_NAMES:
        audits[name] = {"status": sorted(audit_plan.EXPECTED[name])[0]}
    (tmp_path / "timesafe_audits.json").write_text(json.dumps(audits), encoding="utf-8")
    with pytest.raises(RuntimeError, match="evidence identities are inconsistent"):
        assemble_report.assemble(tmp_path, tmp_path / "report.md")


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


def test_committed_reference_evidence_is_self_consistent():
    results = EXPERIMENT_DIR / "results" / "reference_v031"
    assert results.is_dir()
    manifest = results / "MANIFEST.sha256"
    listed = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        listed[name] = digest
    assert "ACTR20_TIMESAFE_EXPERIMENT.md" in listed
    assert "timesafe_audits.json" in listed
    for name, expected in listed.items():
        assert _sha(results / name) == expected

    metadata = json.loads((results / "RESULTS_METADATA.json").read_text(encoding="utf-8"))
    verification = json.loads((results / "benchmark_verification.json").read_text(encoding="utf-8"))
    audits = json.loads((results / "timesafe_audits.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "PASS"
    assert metadata["timesafe_version"] == "0.3.1"
    assert metadata["experiment_run_id"]
    assert metadata["implementation_sha256"] == common.build_implementation_sha256()
    assert metadata["rows"] == 109499
    assert metadata["sessions"] == 1810
    assert verification["status"] == "PASS"
    assert verification["max_abs_rankic_difference"] < 1e-12
    assert audits["all_expected"] is True
    assert audits["full_sample_negative_control"]["status"] == "FAIL"
    assert audits["immature_label_negative_control"]["status"] == "FAIL"
    assert audits["label_maturity_and_purge"]["status"] == "PASS"


def test_actr_fit_model_routes_weights_without_swallowing_value_errors():
    import importlib.util
    from pathlib import Path

    import numpy as np
    import pytest
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    common_path = (
        Path(__file__).parents[1]
        / "experiments"
        / "actr20_public"
        / "common.py"
    )
    spec = importlib.util.spec_from_file_location("timesafe_actr_common_fit_test", common_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    x = np.arange(40, dtype=float).reshape(20, 2)
    y = np.linspace(-1.0, 1.0, 20)
    weights = np.linspace(0.5, 1.5, 20)
    pipeline = make_pipeline(
        SimpleImputer(),
        GradientBoostingRegressor(n_estimators=2, random_state=1),
    )
    fitted = module._fit_model(pipeline, x, y, weights)
    assert fitted is pipeline

    class BrokenEstimator:
        def fit(self, x, y, sample_weight=None):
            raise ValueError("genuine data failure")

    with pytest.raises(ValueError, match="genuine data failure"):
        module._fit_model(BrokenEstimator(), x, y, weights)


def test_actr_adapter_rejects_preimported_module_from_unverified_path(tmp_path, monkeypatch):
    import importlib.util
    import sys
    import types
    from pathlib import Path

    import pytest

    common_path = (
        Path(__file__).parents[1]
        / "experiments"
        / "actr20_public"
        / "common.py"
    )
    spec = importlib.util.spec_from_file_location("timesafe_actr_common_origin_test", common_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fake = types.ModuleType("actr_public")
    fake.__file__ = str(tmp_path / "unverified" / "actr_public" / "__init__.py")
    monkeypatch.setitem(sys.modules, "actr_public", fake)

    with pytest.raises(RuntimeError, match="unverified path"):
        module._assert_actr_module_origin(tmp_path / "verified" / "src")
