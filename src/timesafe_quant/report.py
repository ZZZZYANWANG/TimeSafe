"""Markdown and JSON evidence reports."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Iterable

from .types import AuditStatus, Finding, TimeSafeReport

_STATUS_ORDER = {
    AuditStatus.PASS: 0,
    AuditStatus.WARN: 1,
    AuditStatus.INCONCLUSIVE: 2,
    AuditStatus.FAIL: 3,
    AuditStatus.ERROR: 4,
}


def combine_status(statuses: Iterable[AuditStatus]) -> AuditStatus:
    """Return the most severe status."""

    values = list(statuses)
    if not values:
        return AuditStatus.INCONCLUSIVE
    return max(values, key=_STATUS_ORDER.__getitem__)


def make_report(
    *,
    title: str = "TimeSafe temporal integrity report",
    prefix=None,
    pipeline=None,
    determinism=None,
    maturity=None,
    findings: list[Finding] | None = None,
    metadata: dict | None = None,
) -> TimeSafeReport:
    """Create one combined report from independently runnable audits."""

    sections = [item for item in (prefix, pipeline, determinism, maturity) if item is not None]
    overall = combine_status(item.status for item in sections)
    combined_findings: list[Finding] = []
    for section in sections:
        combined_findings.extend(section.findings)
    if findings:
        combined_findings.extend(findings)
        overall = combine_status([overall, *(item.status for item in findings)])
    return TimeSafeReport(
        title=title,
        overall_status=overall,
        prefix=prefix,
        pipeline=pipeline,
        determinism=determinism,
        maturity=maturity,
        findings=combined_findings,
        metadata=metadata or {},
    )


def render_markdown(report: TimeSafeReport) -> str:
    """Render a compact, publication-friendly Markdown evidence card."""

    lines = [
        f"# {report.title}",
        "",
        f"**Overall status: `{report.overall_status.value}`**",
        "",
        "## Evidence matrix",
        "",
        "| Audit | Status | Key evidence |",
        "|---|---|---|",
    ]
    if report.prefix is not None:
        audit_mode = report.prefix.metadata.get("audit_mode", "transform")
        audit_label = {
            "transform": "Prefix invariance",
            "refit": "Fresh-state refit invariance",
            "availability_transform": "Availability perturbation",
            "availability_refit": "Fresh-state availability perturbation",
            "refit_determinism": "Fresh-state refit determinism",
        }.get(audit_mode, "Temporal invariance")
        lines.append(
            f"| {audit_label} | "
            f"{report.prefix.status.value} | "
            f"{report.prefix.total_changed_cells} changed values; "
            f"{report.prefix.total_structural_differences} structural differences; "
            f"coverage={report.prefix.checked_checkpoints}/"
            f"{report.prefix.eligible_checkpoints} "
            f"({report.prefix.coverage:.1%}); "
            f"first violation: {report.prefix.first_violation or 'none'} |"
        )
    if report.pipeline is not None:
        coverage = report.pipeline.metadata.get("coverage", 0.0)
        lines.append(
            "| Stage localization | "
            f"{report.pipeline.status.value} | "
            f"first failing stage: {report.pipeline.first_failing_stage or 'none'}; "
            f"coverage={coverage:.1%} |"
        )
    if report.determinism is not None:
        lines.append(
            "| Determinism | "
            f"{report.determinism.status.value} | "
            f"{report.determinism.changed_cells} changed values; "
            f"{report.determinism.structural_differences} structural differences across "
            f"{report.determinism.repetitions} runs |"
        )
    if report.maturity is not None:
        lines.append(
            "| Label maturity / purge | "
            f"{report.maturity.status.value} | "
            f"{report.maturity.immature_rows} immature; "
            f"{report.maturity.split_overlap_rows} violating train rows; "
            f"recommended purge={report.maturity.contract.recommended_purge_sessions} sessions |"
        )

    lines.extend(["", "## Findings", ""])
    if not report.findings:
        lines.append("No findings.")
    else:
        for finding in report.findings:
            lines.append(f"- **{finding.status.value} — {finding.code}:** {finding.message}")
            if finding.details:
                details = ", ".join(
                    f"{key}={value}" for key, value in finding.details.items()
                )
                lines.append(f"  - {details}")

    if report.prefix is not None:
        lines.extend(
            [
                "",
                "## Prefix/refit checkpoints",
                "",
                "| Checkpoint | Status | Compared | Changed | Missing left | Missing right | Max abs diff | Context | Error |",
                "|---|---:|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for item in report.prefix.checkpoints:
            lines.append(
                f"| {item.checkpoint} | {item.status.value} | {item.compared_cells} | "
                f"{item.changed_cells} | {item.missing_left} | {item.missing_right} | "
                f"{item.max_abs_diff if item.max_abs_diff is not None else ''} | "
                f"{item.metadata or ''} | {item.error or ''} |"
            )

    if report.pipeline is not None:
        lines.extend(
            [
                "",
                "## Pipeline stages",
                "",
                "| Stage | Status | Violating | Errors | Inconclusive | Changed | Structural | First issue |",
                "|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for stage in report.pipeline.stages:
            lines.append(
                f"| {stage.stage} | {stage.status.value} | "
                f"{stage.violating_checkpoints} | {stage.error_checkpoints} | "
                f"{stage.inconclusive_checkpoints} | {stage.changed_cells} | "
                f"{stage.structural_differences} | {stage.first_violation or ''} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "`PASS` is reserved for conclusive, exhaustive evidence under the supplied "
            "contract. `WARN` means no violation was found but coverage or another "
            "non-fatal limitation remains. `INCONCLUSIVE` means the audit produced no "
            "comparable evidence. None of these statuses proves that an upstream vendor "
            "dataset was historically point-in-time correct. Trainable pipelines should "
            "use the refit audit with fresh estimator state.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_report(report: TimeSafeReport, path: str | Path) -> Path:
    """Write JSON or Markdown based on file suffix."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".json":
        content = (
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2, default=str)
            + "\n"
        )
    elif output.suffix.lower() in {".md", ".markdown"}:
        content = render_markdown(report)
    else:
        raise ValueError("Report path must end with .json, .md, or .markdown.")

    temporary = output.with_name(
        f".{output.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
