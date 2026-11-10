"""Typed audit results used across TimeSafe."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class AuditStatus(str, Enum):
    """Normalized audit state.

    ``PASS`` is reserved for conclusive, exhaustive evidence under the supplied
    contract. ``WARN`` means no violation was found but coverage or another
    non-fatal limitation remains. ``INCONCLUSIVE`` means the audit produced no
    comparable evidence.
    """

    PASS = "PASS"
    WARN = "WARN"
    INCONCLUSIVE = "INCONCLUSIVE"
    FAIL = "FAIL"
    ERROR = "ERROR"


@dataclass(frozen=True)
class Finding:
    """One human-readable audit finding."""

    code: str
    status: AuditStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


@dataclass(frozen=True)
class CheckpointComparison:
    """Comparison at a single historical checkpoint."""

    checkpoint: str
    status: AuditStatus
    compared_cells: int
    changed_cells: int
    changed_fraction: float | None
    missing_left: int = 0
    missing_right: int = 0
    max_abs_diff: float | None = None
    max_rel_diff: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def structural_differences(self) -> int:
        return self.missing_left + self.missing_right

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["structural_differences"] = self.structural_differences
        return payload


@dataclass
class PrefixAuditResult:
    """Result of black-box prefix-invariance or refit testing."""

    status: AuditStatus
    transform_name: str
    checkpoints: list[CheckpointComparison]
    first_violation: str | None
    total_changed_cells: int
    total_compared_cells: int
    total_missing_left: int = 0
    total_missing_right: int = 0
    eligible_checkpoints: int = 0
    checked_checkpoints: int = 0
    checkpoint_policy: str = "uniform"
    exhaustive: bool = False
    findings: list[Finding] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def changed_fraction(self) -> float | None:
        if self.total_compared_cells == 0:
            return None
        return self.total_changed_cells / self.total_compared_cells

    @property
    def coverage(self) -> float:
        if self.eligible_checkpoints == 0:
            return 0.0
        return self.checked_checkpoints / self.eligible_checkpoints

    @property
    def total_structural_differences(self) -> int:
        return self.total_missing_left + self.total_missing_right

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "transform_name": self.transform_name,
            "checkpoints": [item.to_dict() for item in self.checkpoints],
            "first_violation": self.first_violation,
            "total_changed_cells": self.total_changed_cells,
            "total_compared_cells": self.total_compared_cells,
            "total_missing_left": self.total_missing_left,
            "total_missing_right": self.total_missing_right,
            "total_structural_differences": self.total_structural_differences,
            "changed_fraction": self.changed_fraction,
            "eligible_checkpoints": self.eligible_checkpoints,
            "checked_checkpoints": self.checked_checkpoints,
            "checkpoint_policy": self.checkpoint_policy,
            "exhaustive": self.exhaustive,
            "coverage": self.coverage,
            "findings": [item.to_dict() for item in self.findings],
            "metadata": self.metadata,
        }


@dataclass
class PipelineStageSummary:
    """Aggregated prefix-audit result for a named pipeline stage."""

    stage: str
    status: AuditStatus
    violating_checkpoints: int
    changed_cells: int
    compared_cells: int
    first_violation: str | None
    missing_left: int = 0
    missing_right: int = 0
    error_checkpoints: int = 0
    inconclusive_checkpoints: int = 0

    @property
    def structural_differences(self) -> int:
        return self.missing_left + self.missing_right

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["structural_differences"] = self.structural_differences
        return payload


@dataclass
class PipelineAuditResult:
    """Prefix-invariance result for a stage-aware pipeline."""

    status: AuditStatus
    first_failing_stage: str | None
    stages: list[PipelineStageSummary]
    findings: list[Finding] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "first_failing_stage": self.first_failing_stage,
            "stages": [stage.to_dict() for stage in self.stages],
            "findings": [item.to_dict() for item in self.findings],
            "metadata": self.metadata,
        }


@dataclass
class DeterminismAuditResult:
    """Repeated-run determinism result."""

    status: AuditStatus
    repetitions: int
    changed_cells: int
    compared_cells: int
    missing_left: int = 0
    missing_right: int = 0
    findings: list[Finding] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def structural_differences(self) -> int:
        return self.missing_left + self.missing_right

    @property
    def changed_fraction(self) -> float | None:
        if self.compared_cells == 0:
            return None
        return self.changed_cells / self.compared_cells

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "repetitions": self.repetitions,
            "changed_cells": self.changed_cells,
            "compared_cells": self.compared_cells,
            "missing_left": self.missing_left,
            "missing_right": self.missing_right,
            "structural_differences": self.structural_differences,
            "changed_fraction": self.changed_fraction,
            "findings": [item.to_dict() for item in self.findings],
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class LabelContractSummary:
    """Serializable summary of a label timing contract."""

    execution_offset_sessions: int
    holding_period_sessions: int
    label_publication_lag_sessions: int
    safety_gap_sessions: int
    decision_event: str
    execution_event: str
    exit_event: str
    label_available_event: str
    label_start_offset_sessions: int
    label_end_offset_sessions: int
    label_available_offset_sessions: int
    recommended_purge_sessions: int
    contract_schema_version: int
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LabelMaturityAuditResult:
    """Label maturity and split-overlap audit result."""

    status: AuditStatus
    contract: LabelContractSummary
    rows: int
    mature_rows: int
    immature_rows: int
    split_overlap_rows: int
    findings: list[Finding] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "contract": self.contract.to_dict(),
            "rows": self.rows,
            "mature_rows": self.mature_rows,
            "immature_rows": self.immature_rows,
            "split_overlap_rows": self.split_overlap_rows,
            "findings": [item.to_dict() for item in self.findings],
            "metadata": self.metadata,
        }


@dataclass
class TimeSafeReport:
    """Combined evidence report."""

    title: str
    overall_status: AuditStatus
    prefix: PrefixAuditResult | None = None
    pipeline: PipelineAuditResult | None = None
    determinism: DeterminismAuditResult | None = None
    maturity: LabelMaturityAuditResult | None = None
    findings: list[Finding] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "overall_status": self.overall_status.value,
            "prefix": None if self.prefix is None else self.prefix.to_dict(),
            "pipeline": None if self.pipeline is None else self.pipeline.to_dict(),
            "determinism": None if self.determinism is None else self.determinism.to_dict(),
            "maturity": None if self.maturity is None else self.maturity.to_dict(),
            "findings": [item.to_dict() for item in self.findings],
            "metadata": self.metadata,
        }
