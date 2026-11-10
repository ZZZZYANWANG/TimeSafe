"""Markdown and JSON evidence reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .types import AuditStatus, Finding, TimeSafeReport

_STATUS_ORDER = {
    AuditStatus.PASS: 0,
    AuditStatus.WARN: 1,
    AuditStatus.FAIL: 2,
    AuditStatus.ERROR: 3,
}


def combine_status(statuses: Iterable[AuditStatus]) -> AuditStatus:
    """Return the most severe status."""

    values = list(statuses)
    if not values:
        return AuditStatus.PASS
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
        lines.append(
            "| Prefix invariance | "
            f"{report.prefix.status.value} | "
            f"{report.prefix.total_changed_cells} changed cells; "
            f"first violation: {report.prefix.first_violation or 'none'} |"
        )
    if report.pipeline is not None:
        lines.append(
            "| Stage localization | "
            f"{report.pipeline.status.value} | "
            f"first failing stage: {report.pipeline.first_failing_stage or 'none'} |"
        )
    if report.determinism is not None:
        lines.append(
            "| Determinism | "
            f"{report.determinism.status.value} | "
            f"{report.determinism.changed_cells} changed cells across "
            f"{report.determinism.repetitions} runs |"
        )
    if report.maturity is not None:
        lines.append(
            "| Label maturity / purge | "
            f"{report.maturity.status.value} | "
            f"{report.maturity.immature_rows} immature; "
            f"{report.maturity.split_overlap_rows} overlapping train rows; "
            f"recommended purge={report.maturity.contract.recommended_purge_sessions} sessions |"
        )

    lines.extend(["", "## Findings", ""])
    if not report.findings:
        lines.append("No findings.")
    else:
        for finding in report.findings:
            lines.append(f"- **{finding.status.value} — {finding.code}:** {finding.message}")
            if finding.details:
                details = ", ".join(f"{key}={value}" for key, value in finding.details.items())
                lines.append(f"  - {details}")

    if report.prefix is not None:
        lines.extend(
            [
                "",
                "## Prefix checkpoints",
                "",
                "| Checkpoint | Status | Compared | Changed | Max abs diff | Error |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for item in report.prefix.checkpoints:
            lines.append(
                f"| {item.checkpoint} | {item.status.value} | {item.compared_cells} | "
                f"{item.changed_cells} | {item.max_abs_diff if item.max_abs_diff is not None else ''} | "
                f"{item.error or ''} |"
            )

    if report.pipeline is not None:
        lines.extend(
            [
                "",
                "## Pipeline stages",
                "",
                "| Stage | Status | Violating checkpoints | Changed cells | First violation |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for stage in report.pipeline.stages:
            lines.append(
                f"| {stage.stage} | {stage.status.value} | {stage.violating_checkpoints} | "
                f"{stage.changed_cells} | {stage.first_violation or ''} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A `PASS` means the tested implementation satisfied this audit under the supplied "
            "checkpoints, tolerances, calendar, and timing contract. It is not a proof that every "
            "upstream data source was point-in-time correct. Pair TimeSafe with data-vintage controls, "
            "purged model selection, and cost-aware backtesting.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_report(report: TimeSafeReport, path: str | Path) -> Path:
    """Write JSON or Markdown based on file suffix."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".json":
        output.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    elif output.suffix.lower() in {".md", ".markdown"}:
        output.write_text(render_markdown(report), encoding="utf-8")
    else:
        raise ValueError("Report path must end with .json, .md, or .markdown.")
    return output
