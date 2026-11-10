"""TimeSafe: temporal integrity auditing for quantitative research."""

from .availability import (
    AvailabilityRule,
    audit_availability_invariance,
    audit_refit_availability_invariance,
    mask_unavailable_fields,
)
from .maturity import (
    IntervalContractValidation,
    LabelContract,
    audit_label_maturity,
    build_label_intervals,
    purged_train_test_masks,
    purged_walk_forward_masks,
    validate_label_intervals,
)
from .pipeline import CausalPipeline, PipelineStep, audit_pipeline_prefix_invariance
from .prefix import (
    audit_determinism,
    audit_prefix_invariance,
    choose_checkpoints,
    plan_checkpoints,
)
from .refit import (
    audit_refit_determinism,
    audit_refit_prefix_invariance,
    sklearn_estimator_factory,
)
from .report import make_report, render_markdown, write_report
from .timing import SessionEvent
from .types import AuditStatus, TimeSafeReport

__all__ = [
    "AuditStatus",
    "AvailabilityRule",
    "CausalPipeline",
    "IntervalContractValidation",
    "LabelContract",
    "PipelineStep",
    "SessionEvent",
    "TimeSafeReport",
    "audit_availability_invariance",
    "audit_determinism",
    "audit_label_maturity",
    "audit_pipeline_prefix_invariance",
    "audit_prefix_invariance",
    "audit_refit_availability_invariance",
    "audit_refit_determinism",
    "audit_refit_prefix_invariance",
    "build_label_intervals",
    "choose_checkpoints",
    "make_report",
    "mask_unavailable_fields",
    "plan_checkpoints",
    "purged_train_test_masks",
    "purged_walk_forward_masks",
    "render_markdown",
    "sklearn_estimator_factory",
    "validate_label_intervals",
    "write_report",
]

__version__ = "0.3.0"
