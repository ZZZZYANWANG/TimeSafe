"""Stage 3: assemble a dependency-light ACTR/TimeSafe evidence report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_plan import AUDIT_NAMES


def _format_value(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.8g}"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small DataFrame without requiring the optional tabulate package."""

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format_value(value) for value in row) + " |")
    return "\n".join(lines)


def _validate_evidence_identity(audits: dict, config: dict) -> dict:
    """Tie the aggregate audit evidence to the benchmark reproduced in stage 1."""

    identity = audits.get("identity")
    if not isinstance(identity, dict):
        raise RuntimeError("timesafe_audits.json is missing its shared identity block.")
    if audits.get("all_expected") is not True:
        raise RuntimeError("TimeSafe audit verdicts do not match the pre-registered plan.")
    expected = {
        "experiment_run_id": config.get("experiment_run_id"),
        "implementation_sha256": config.get("implementation_sha256"),
        "audit_contract_version": config.get("experiment_contract_version"),
        "timesafe_version": config.get("timesafe_version"),
        "benchmark_sha256": config.get("data_sha256"),
        "actr_manifest_sha256": config.get("actr_manifest_sha256"),
        "benchmark_rows": config.get("rows"),
        "benchmark_sessions": config.get("sessions"),
    }
    mismatches = {
        key: {"audit": identity.get(key), "benchmark": value}
        for key, value in expected.items()
        if identity.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Benchmark and audit evidence identities are inconsistent: {mismatches}."
        )
    return identity


def assemble(results_dir: str | Path, output_path: str | Path) -> Path:
    root = Path(results_dir)
    summary = pd.read_csv(root / "benchmark_summary.csv")
    pairs = pd.read_csv(root / "benchmark_pairs.csv")
    audits = json.loads((root / "timesafe_audits.json").read_text(encoding="utf-8"))
    config = json.loads((root / "benchmark_config.json").read_text(encoding="utf-8"))
    verification = json.loads(
        (root / "benchmark_verification.json").read_text(encoding="utf-8")
    )
    identity = _validate_evidence_identity(audits, config)

    rows = []
    purposes = {
        "mature_feedback_prefix": "Exact ACTR mature-feedback function must remain prefix invariant",
        "legal_refit": "Same legal learner on prefix and full input with an explicit cutoff",
        "full_sample_negative_control": "Must detect fitting on future rows",
        "legal_availability": "Mask not-yet-mature targets without changing legal predictions",
        "immature_label_negative_control": "Must detect materialized future targets used in training",
        "refit_determinism": "Fresh refits with fixed seeds must reproduce predictions",
        "label_maturity_and_purge": "Validate close-to-next-open 20-session labels plus a 4-session gap",
    }
    display_names = {
        "mature_feedback_prefix": "ACTR mature-feedback prefix audit",
        "legal_refit": "Legal fresh-state refit",
        "full_sample_negative_control": "Full-sample training negative control",
        "legal_availability": "Legal availability filtering",
        "immature_label_negative_control": "Immature-label negative control",
        "refit_determinism": "Fresh-state determinism",
        "label_maturity_and_purge": "Label maturity and purge",
    }
    for key in AUDIT_NAMES:
        result = audits[key]
        rows.append(
            {
                "Audit": display_names[key],
                "Verdict": result["status"],
                "Checked / eligible": (
                    f"{result.get('checked_checkpoints', '')} / "
                    f"{result.get('eligible_checkpoints', '')}"
                    if "checked_checkpoints" in result
                    else "n/a"
                ),
                "Changed values": result.get("total_changed_cells", ""),
                "Structural changes": result.get("total_structural_differences", ""),
                "Purpose": purposes[key],
            }
        )
    audit_table = pd.DataFrame(rows)

    lines = [
        "# TimeSafe × ACTR-20 public model experiment",
        "",
        "This experiment combines the external ACTR privacy-hardened public benchmark "
        "with TimeSafe temporal-integrity audits. The anonymous benchmark preserves "
        "the public modeling task but is block-shuffled; it is evidence of structural "
        "reproducibility, not a substitute for the original chronological market study.",
        "",
        "## External reference verification",
        "",
        f"- Reference-metric status: `{verification['status']}`",
        f"- External manifest status: `{config['external_manifest_verification']['status']}` "
        f"({config['external_manifest_verification']['checked_files']} files verified)",
        f"- Benchmark data SHA-256: `{config['data_sha256']}`",
        f"- External manifest SHA-256: `{config['actr_manifest_sha256']}`",
        f"- Rows / sessions: `{config['rows']}` / `{config['sessions']}`",
        f"- TimeSafe / experiment contract: `{identity['timesafe_version']}` / "
        f"`{identity['audit_contract_version']}`",
        f"- Experiment run / implementation identity: `{identity['experiment_run_id']}` / "
        f"`{identity['implementation_sha256']}`",
        f"- Audit subset identity: `{identity['audit_asset_ids_sha256']}` "
        f"({identity['audit_asset_count']} assets, {identity['audit_rows']} rows)",
        f"- Maximum absolute RankIC difference versus the ACTR package reference: "
        f"`{verification['max_abs_rankic_difference']:.3g}`",
        f"- ACTR HAC p-value difference versus reference: "
        f"`{verification['actr_hac_p_difference']:.3g}`",
        "",
        "## Benchmark reproduction",
        "",
        _markdown_table(summary),
        "",
        "## Paired comparisons",
        "",
        _markdown_table(pairs),
        "",
        f"Selected anchor alpha: `{config['selected_anchor_alpha']}`",
        f"Selected residual gamma: `{config['selected_residual_gamma']}`",
        "",
        "## TimeSafe verdicts",
        "",
        _markdown_table(audit_table),
        "",
        "## Interpretation",
        "",
        "The full benchmark is recomputed on all 109,499 anonymous rows. The exact "
        "ACTR mature-feedback transform is checked exhaustively at all 21 eligible "
        "checkpoints in a pre-registered bounded window. The expensive fresh-state "
        "model audits use four pre-registered checkpoints and the first 24 anonymous "
        "assets across all 1,810 sessions. A clean sampled model audit therefore "
        "returns `WARN`, not `PASS`; the exact coverage is retained in JSON. Both "
        "deliberately leaking negative controls must return `FAIL`, while the event-aware "
        "label contract and purge audit must return `PASS`.",
        "",
        "Every model audit is executed in a fresh interpreter and exchanges only JSON "
        "artifacts. This avoids process-local native state from one scikit-learn "
        "histogram-boosting workload affecting a later audit.",
        "",
        "The TimeSafe repository does not redistribute the ACTR benchmark array, private "
        "feature definitions, strong-expert checkpoints, instruments, positions, or "
        "orders. The external ACTR research-only notice remains controlling.",
    ]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(assemble(args.results_dir, args.output))


if __name__ == "__main__":
    main()
