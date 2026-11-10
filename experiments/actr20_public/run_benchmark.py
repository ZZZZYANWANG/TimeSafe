"""Stage 1: reproduce and verify the external ACTR privacy-hardened benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import file_sha256, register_actr_package, verify_sha256_manifest


METRIC_NAME_MAP = {
    "strong_cs_rankic": "Strong CS",
    "strong_ts_rankic": "Strong TS",
    "stable_fusion_rankic": "Stable fusion",
    "cross_anchor_rankic": "Cross anchor",
    "actr_rankic": "ACTR benchmark",
}


def _verify_against_external_package(
    *,
    root: Path,
    summary,
    selected_alpha: float,
    selected_gamma: float,
    data_sha256: str,
    rows: int,
    sessions: int,
    atol: float = 1e-9,
) -> dict:
    verification_path = root / "PACKAGE_VERIFICATION.json"
    config_path = root / "results" / "reference" / "benchmark_config.json"
    if not verification_path.exists() or not config_path.exists():
        raise FileNotFoundError(
            "The ACTR package must include PACKAGE_VERIFICATION.json and "
            "results/reference/benchmark_config.json."
        )
    external = json.loads(verification_path.read_text(encoding="utf-8"))
    expected_metrics = external["privacy_hardened_runnable_benchmark"]
    expected_config = json.loads(config_path.read_text(encoding="utf-8"))
    observed_by_model = dict(zip(summary["model"], summary["rankic"]))

    metric_differences: dict[str, float] = {}
    for key, model in METRIC_NAME_MAP.items():
        if model not in observed_by_model:
            raise RuntimeError(f"ACTR output is missing expected model {model!r}.")
        difference = float(observed_by_model[model]) - float(expected_metrics[key])
        metric_differences[key] = difference
        if not np.isclose(difference, 0.0, atol=atol, rtol=0.0):
            raise RuntimeError(
                f"ACTR benchmark drift for {key}: observed={observed_by_model[model]}, "
                f"expected={expected_metrics[key]}, difference={difference}."
            )

    actr_p = float(summary.loc[summary["model"] == "ACTR benchmark", "hac_p"].iloc[0])
    p_difference = actr_p - float(expected_metrics["actr_hac_p"])
    if not np.isclose(p_difference, 0.0, atol=atol, rtol=0.0):
        raise RuntimeError(
            f"ACTR HAC p-value drift: observed={actr_p}, "
            f"expected={expected_metrics['actr_hac_p']}."
        )
    if data_sha256 != external["package"]["benchmark_sha256"]:
        raise RuntimeError("ACTR benchmark data SHA-256 does not match package verification.")
    if not np.isclose(selected_alpha, expected_config["selected_anchor_alpha"], atol=atol):
        raise RuntimeError("Selected ACTR anchor alpha does not match the external reference.")
    if not np.isclose(selected_gamma, expected_config["selected_residual_gamma"], atol=atol):
        raise RuntimeError("Selected ACTR residual gamma does not match the external reference.")
    metadata_path = root / "data" / "benchmark_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_rows = int(metadata["rows"])
    expected_sessions = int(metadata["anonymous_sessions"])
    if rows != expected_rows or sessions != expected_sessions:
        raise RuntimeError(
            "Unexpected ACTR benchmark shape: "
            f"rows={rows} (expected {expected_rows}), "
            f"sessions={sessions} (expected {expected_sessions})."
        )

    return {
        "status": "PASS",
        "reference": str(verification_path.name),
        "atol": atol,
        "max_abs_rankic_difference": max(abs(value) for value in metric_differences.values()),
        "actr_hac_p_difference": p_difference,
        "metric_differences": metric_differences,
        "data_sha256_match": True,
        "selected_config_match": True,
        "shape_match": True,
        "expected_rows": expected_rows,
        "expected_sessions": expected_sessions,
    }


def run(actr_root: str | Path, output_dir: str | Path) -> dict:
    root = Path(actr_root).expanduser().resolve()
    manifest_verification = verify_sha256_manifest(root)
    root = register_actr_package(root)
    from actr_public import ACTRPublicFramework, load_benchmark

    data_path = root / "data" / "actr_deidentified_benchmark.npz"
    data = load_benchmark(data_path)
    result = ACTRPublicFramework().fit_predict(data)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.summary.to_csv(output / "benchmark_summary.csv", index=False)
    result.annual.to_csv(output / "benchmark_annual.csv", index=False)
    result.daily_ic.to_csv(output / "benchmark_daily_ic.csv", index=False)
    result.pairs.to_csv(output / "benchmark_pairs.csv", index=False)
    result.bootstrap.to_csv(output / "benchmark_block_bootstrap.csv", index=False)

    data_hash = file_sha256(data_path)
    rows = int(len(data.date_id))
    sessions = int(len(set(data.date_id.tolist())))
    verification = _verify_against_external_package(
        root=root,
        summary=result.summary,
        selected_alpha=float(result.selected_alpha),
        selected_gamma=float(result.selected_gamma),
        data_sha256=data_hash,
        rows=rows,
        sessions=sessions,
    )
    config = {
        "selected_anchor_alpha": result.selected_alpha,
        "selected_residual_gamma": result.selected_gamma,
        "data_sha256": data_hash,
        "rows": rows,
        "sessions": sessions,
        "external_reference_verification": verification,
        "external_manifest_verification": manifest_verification,
    }
    (output / "benchmark_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (output / "benchmark_verification.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actr-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run(args.actr_root, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
