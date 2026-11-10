"""TimeSafe: temporal integrity auditing for quantitative research."""

from .maturity import (
    LabelContract,
    audit_label_maturity,
    build_label_intervals,
    purged_walk_forward_masks,
)
from .pipeline import CausalPipeline, PipelineStep, audit_pipeline_prefix_invariance
from .prefix import audit_determinism, audit_prefix_invariance, choose_checkpoints
from .report import make_report, render_markdown, write_report
from .types import AuditStatus, TimeSafeReport

__all__ = [
    "AuditStatus",
    "CausalPipeline",
    "LabelContract",
    "PipelineStep",
    "TimeSafeReport",
    "audit_determinism",
    "audit_label_maturity",
    "audit_pipeline_prefix_invariance",
    "audit_prefix_invariance",
    "build_label_intervals",
    "choose_checkpoints",
    "make_report",
    "purged_walk_forward_masks",
    "render_markdown",
    "write_report",
]

__version__ = "0.1.0"
