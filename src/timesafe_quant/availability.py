"""Audits for values materialized before their declared availability time."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from ._pandas import (
    PandasObject,
    TimeLevel,
    ensure_pandas_output,
    point_slice,
    prefix_slice,
    stable_checkpoint_repr,
    time_values,
    validate_temporal_order,
)
from .compare import compare_labeled_outputs
from .refit import _new_distinct_estimator
from .prefix import (
    CheckpointPolicy,
    _build_prefix_findings,
    _checkpoint_status,
    _overall_prefix_status,
    plan_checkpoints,
)
from .timing import EVENT_ORDER, SessionEvent, normalize_event
from .types import AuditStatus, CheckpointComparison, Finding, PrefixAuditResult

Transform = Callable[[pd.DataFrame], PandasObject]
EstimatorFactory = Callable[[], Any]
FitPredict = Callable[[Any, pd.DataFrame, Any], PandasObject]


@dataclass(frozen=True)
class AvailabilityRule:
    """Declare when one materialized data column becomes legally observable.

    Parameters
    ----------
    value_column:
        Column whose pre-availability values must be masked.
    available_at_column:
        Column containing the session at which each row's value becomes available.
    available_event:
        Constant within-session availability event when no event column is supplied.
    available_event_column:
        Optional row-level event column.
    mask_value:
        Replacement used for unavailable values. ``NaN`` is the default because most
        model pipelines already treat it as missing information.
    """

    value_column: str
    available_at_column: str
    available_event: str | SessionEvent = SessionEvent.AFTER_CLOSE
    available_event_column: str | None = None
    mask_value: Any = np.nan
    reference: Literal["checkpoint", "row"] = "checkpoint"

    def __post_init__(self) -> None:
        object.__setattr__(self, "available_event", normalize_event(self.available_event))
        if self.reference not in {"checkpoint", "row"}:
            raise ValueError("reference must be 'checkpoint' or 'row'.")
        if not pd.api.types.is_scalar(self.mask_value):
            raise TypeError("mask_value must be a scalar replacement value.")


def _availability_mask(
    frame: pd.DataFrame,
    *,
    checkpoint: Any,
    decision_event: str | SessionEvent,
    time_level: TimeLevel,
    rule: AvailabilityRule,
) -> np.ndarray:
    if rule.value_column not in frame.columns:
        raise ValueError(f"Missing value column {rule.value_column!r}.")
    if rule.available_at_column not in frame.columns:
        raise ValueError(f"Missing availability column {rule.available_at_column!r}.")

    decision_name = normalize_event(decision_event)
    available_at = frame[rule.available_at_column]
    if rule.available_event_column is None:
        event_rank = np.full(len(frame), EVENT_ORDER[str(rule.available_event)], dtype=np.int8)
    else:
        if rule.available_event_column not in frame.columns:
            raise ValueError(
                f"Missing availability event column {rule.available_event_column!r}."
            )
        event_rank = frame[rule.available_event_column].map(
            lambda value: EVENT_ORDER[normalize_event(value)]
        ).to_numpy(dtype=np.int8)

    if rule.reference == "checkpoint":
        reference_session: Any = checkpoint
        session_available = available_at.notna() & (
            (available_at < reference_session)
            | (
                (available_at == reference_session)
                & (event_rank <= EVENT_ORDER[decision_name])
            )
        )
    else:
        reference_session = pd.Series(
            time_values(frame.index, time_level).to_numpy(),
            index=frame.index,
        )
        session_available = available_at.notna() & (
            (available_at < reference_session)
            | (
                (available_at == reference_session)
                & (event_rank <= EVENT_ORDER[decision_name])
            )
        )
    return ~session_available.to_numpy(dtype=bool)


def mask_unavailable_fields(
    data: pd.DataFrame,
    *,
    checkpoint: Any,
    rules: Sequence[AvailabilityRule],
    decision_event: str | SessionEvent = SessionEvent.CLOSE,
    time_level: TimeLevel = 0,
) -> tuple[pd.DataFrame, int, dict[str, int]]:
    """Return a copy with values hidden until their declared availability point."""

    if not rules:
        raise ValueError("rules must contain at least one AvailabilityRule.")
    value_columns = [rule.value_column for rule in rules]
    if len(value_columns) != len(set(value_columns)):
        raise ValueError("Each availability rule must target a unique value_column.")
    metadata_columns = {rule.available_at_column for rule in rules}
    metadata_columns.update(
        rule.available_event_column
        for rule in rules
        if rule.available_event_column is not None
    )
    overlap = sorted(set(value_columns).intersection(metadata_columns))
    if overlap:
        raise ValueError(
            "Availability metadata columns cannot also be masked value columns: "
            f"{overlap}."
        )
    validate_temporal_order(data, time_level)
    masked = data.copy(deep=True)
    counts: dict[str, int] = {}
    total = 0
    for rule in rules:
        unavailable = _availability_mask(
            data,
            checkpoint=checkpoint,
            decision_event=decision_event,
            time_level=time_level,
            rule=rule,
        )
        values = data[rule.value_column]
        if bool(pd.isna(rule.mask_value)):
            actually_changed = unavailable & values.notna().to_numpy(dtype=bool)
        else:
            equal_to_mask = values.eq(rule.mask_value).fillna(False).to_numpy(dtype=bool)
            actually_changed = unavailable & ~equal_to_mask
        count = int(actually_changed.sum())
        counts[rule.value_column] = count
        total += count
        if unavailable.any():
            try:
                masked.loc[unavailable, rule.value_column] = rule.mask_value
            except (TypeError, ValueError):
                masked[rule.value_column] = masked[rule.value_column].astype(object)
                masked.loc[unavailable, rule.value_column] = rule.mask_value
    return masked, total, counts


def _audit_availability(
    *,
    runner: Callable[[pd.DataFrame, Any, str], PandasObject],
    data: pd.DataFrame,
    rules: Sequence[AvailabilityRule],
    time_level: TimeLevel,
    checkpoints: Sequence[Any] | None,
    checkpoint_policy: CheckpointPolicy,
    n_checkpoints: int,
    min_history: int,
    decision_event: str | SessionEvent,
    atol: float,
    rtol: float,
    audit_name: str,
    audit_mode: str,
) -> PrefixAuditResult:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas.DataFrame.")
    if data.empty:
        raise ValueError("data must not be empty.")
    if not data.index.is_unique:
        raise ValueError("data index must be unique.")
    validate_temporal_order(data, time_level)

    plan = plan_checkpoints(
        data,
        time_level=time_level,
        checkpoints=checkpoints,
        checkpoint_policy=checkpoint_policy,
        n_checkpoints=n_checkpoints,
        min_history=min_history,
    )

    comparisons: list[CheckpointComparison] = []
    total_changed = 0
    total_compared = 0
    total_missing_left = 0
    total_missing_right = 0
    total_masked = 0
    first_violation: str | None = None

    for checkpoint in plan.selected:
        checkpoint_text = stable_checkpoint_repr(checkpoint)
        try:
            prefix = prefix_slice(data, checkpoint, time_level)
            safe_prefix, masked_cells, per_column = mask_unavailable_fields(
                prefix,
                checkpoint=checkpoint,
                rules=rules,
                decision_event=decision_event,
                time_level=time_level,
            )
            total_masked += masked_cells
            materialized_output = ensure_pandas_output(
                runner(prefix.copy(deep=True), checkpoint, "materialized"),
                name=f"{audit_name}.materialized",
            )
            safe_output = ensure_pandas_output(
                runner(safe_prefix, checkpoint, "availability_safe"),
                name=f"{audit_name}.availability_safe",
            )
            left = point_slice(safe_output, checkpoint, time_level)
            right = point_slice(materialized_output, checkpoint, time_level)
            stats = compare_labeled_outputs(left, right, atol=atol, rtol=rtol)
            status = _checkpoint_status(stats)
            if masked_cells == 0 and status is AuditStatus.PASS:
                status = AuditStatus.INCONCLUSIVE
            if status is AuditStatus.FAIL and first_violation is None:
                first_violation = checkpoint_text
            total_changed += stats.changed_cells
            total_compared += stats.compared_cells
            total_missing_left += stats.missing_left
            total_missing_right += stats.missing_right
            comparisons.append(
                CheckpointComparison(
                    checkpoint=checkpoint_text,
                    status=status,
                    compared_cells=stats.compared_cells,
                    changed_cells=stats.changed_cells,
                    changed_fraction=stats.changed_fraction,
                    missing_left=stats.missing_left,
                    missing_right=stats.missing_right,
                    max_abs_diff=stats.max_abs_diff,
                    max_rel_diff=stats.max_rel_diff,
                    metadata={
                        "masked_cells": masked_cells,
                        "masked_by_column": per_column,
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001
            if first_violation is None:
                first_violation = checkpoint_text
            comparisons.append(
                CheckpointComparison(
                    checkpoint=checkpoint_text,
                    status=AuditStatus.ERROR,
                    compared_cells=0,
                    changed_cells=0,
                    changed_fraction=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    status = _overall_prefix_status(comparisons, exhaustive=plan.exhaustive)
    findings = _build_prefix_findings(
        status=status,
        plan=plan,
        first_violation=first_violation,
        total_changed=total_changed,
        total_missing_left=total_missing_left,
        total_missing_right=total_missing_right,
        atol=atol,
        rtol=rtol,
        audit_mode=audit_mode,
    )

    if total_masked == 0 and status not in {AuditStatus.FAIL, AuditStatus.ERROR}:
        status = AuditStatus.INCONCLUSIVE
        findings = [
            Finding(
                code="TS-AVAILABILITY-021",
                status=status,
                message=(
                    "No declared value was masked at the tested checkpoints; the audit "
                    "did not exercise an availability boundary."
                ),
                details={
                    "checked_checkpoints": len(plan.selected),
                    "rules": [rule.value_column for rule in rules],
                },
            )
        ]
    elif status in {AuditStatus.PASS, AuditStatus.WARN}:
        findings.append(
            Finding(
                code="TS-AVAILABILITY-001",
                status=AuditStatus.PASS,
                message=(
                    "Predictions were invariant to masking values that were not yet "
                    "available under the supplied rules."
                ),
                details={"masked_cells": total_masked},
            )
        )
    elif status is AuditStatus.FAIL:
        findings.append(
            Finding(
                code="TS-AVAILABILITY-101",
                status=status,
                message=(
                    "Predictions changed when not-yet-available values were masked; "
                    "the pipeline depends on materialized future information."
                ),
                details={
                    "masked_cells": total_masked,
                    "first_violation": first_violation,
                },
            )
        )

    return PrefixAuditResult(
        status=status,
        transform_name=audit_name,
        checkpoints=comparisons,
        first_violation=first_violation,
        total_changed_cells=total_changed,
        total_compared_cells=total_compared,
        total_missing_left=total_missing_left,
        total_missing_right=total_missing_right,
        eligible_checkpoints=len(plan.eligible),
        checked_checkpoints=len(plan.selected),
        checkpoint_policy=plan.policy,
        exhaustive=plan.exhaustive,
        findings=findings,
        metadata={
            "audit_mode": audit_mode,
            "coverage": plan.coverage,
            "decision_event": normalize_event(decision_event),
            "masked_cells": total_masked,
            "availability_rules": [
                {
                    "value_column": rule.value_column,
                    "available_at_column": rule.available_at_column,
                    "available_event": str(rule.available_event),
                    "reference": rule.reference,
                }
                for rule in rules
            ],
        },
    )


def audit_availability_invariance(
    transform: Transform,
    data: pd.DataFrame,
    *,
    rules: Sequence[AvailabilityRule],
    time_level: TimeLevel = 0,
    checkpoints: Sequence[Any] | None = None,
    checkpoint_policy: CheckpointPolicy = "uniform",
    n_checkpoints: int = 12,
    min_history: int = 30,
    decision_event: str | SessionEvent = SessionEvent.CLOSE,
    atol: float = 1e-10,
    rtol: float = 1e-8,
    audit_name: str | None = None,
) -> PrefixAuditResult:
    """Test whether a transform depends on values unavailable at each checkpoint."""

    name = audit_name or getattr(transform, "__name__", transform.__class__.__name__)

    def runner(frame: pd.DataFrame, checkpoint: Any, path: str) -> PandasObject:
        _ = checkpoint, path
        return transform(frame)

    return _audit_availability(
        runner=runner,
        data=data,
        rules=rules,
        time_level=time_level,
        checkpoints=checkpoints,
        checkpoint_policy=checkpoint_policy,
        n_checkpoints=n_checkpoints,
        min_history=min_history,
        decision_event=decision_event,
        atol=atol,
        rtol=rtol,
        audit_name=name,
        audit_mode="availability_transform",
    )


def audit_refit_availability_invariance(
    estimator_factory: EstimatorFactory,
    fit_predict: FitPredict,
    data: pd.DataFrame,
    *,
    rules: Sequence[AvailabilityRule],
    time_level: TimeLevel = 0,
    checkpoints: Sequence[Any] | None = None,
    checkpoint_policy: CheckpointPolicy = "uniform",
    n_checkpoints: int = 12,
    min_history: int = 30,
    decision_event: str | SessionEvent = SessionEvent.CLOSE,
    atol: float = 1e-10,
    rtol: float = 1e-8,
    audit_name: str = "availability_refit_pipeline",
) -> PrefixAuditResult:
    """Fresh-state availability audit for a trainable model pipeline."""

    retained_estimators: list[Any] = []

    def runner(frame: pd.DataFrame, checkpoint: Any, path: str) -> PandasObject:
        estimator = _new_distinct_estimator(
            estimator_factory,
            retained_estimators,
            context=f"availability checkpoint={checkpoint!r} path={path}",
        )
        return fit_predict(estimator, frame, checkpoint)

    return _audit_availability(
        runner=runner,
        data=data,
        rules=rules,
        time_level=time_level,
        checkpoints=checkpoints,
        checkpoint_policy=checkpoint_policy,
        n_checkpoints=n_checkpoints,
        min_history=min_history,
        decision_event=decision_event,
        atol=atol,
        rtol=rtol,
        audit_name=audit_name,
        audit_mode="availability_refit",
    )
