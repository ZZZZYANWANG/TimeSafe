"""Black-box temporal prefix-invariance auditing."""

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
    sorted_unique_times,
    stable_checkpoint_repr,
    validate_temporal_order,
)
from .compare import ComparisonStats, compare_labeled_outputs
from .types import (
    AuditStatus,
    CheckpointComparison,
    DeterminismAuditResult,
    Finding,
    PrefixAuditResult,
)

Transform = Callable[[pd.DataFrame], PandasObject]
CheckpointPolicy = Literal["uniform", "all", "explicit"]


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer.")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive.")
    return normalized


@dataclass(frozen=True)
class CheckpointPlan:
    """Resolved checkpoint sampling plan."""

    selected: list[Any]
    eligible: list[Any]
    policy: CheckpointPolicy

    @property
    def exhaustive(self) -> bool:
        return len(self.eligible) > 0 and set(self.selected) == set(self.eligible)

    @property
    def coverage(self) -> float:
        if not self.eligible:
            return 0.0
        return len(self.selected) / len(self.eligible)


def plan_checkpoints(
    data: pd.DataFrame,
    *,
    time_level: TimeLevel = 0,
    checkpoints: Sequence[Any] | None = None,
    checkpoint_policy: CheckpointPolicy = "uniform",
    n_checkpoints: int = 12,
    min_history: int = 30,
) -> CheckpointPlan:
    """Resolve historical checkpoints and their evidence coverage.

    The final timestamp is excluded because no future row exists beyond it; comparing
    a full sample with itself provides no look-ahead evidence.
    """

    validate_temporal_order(data, time_level)
    min_history = _positive_integer(min_history, name="min_history")
    n_checkpoints = _positive_integer(n_checkpoints, name="n_checkpoints")
    times = sorted_unique_times(data.index, time_level)
    if len(times) <= min_history:
        raise ValueError(
            f"Need more than min_history={min_history} unique times; received {len(times)}."
        )

    eligible = list(times[min_history - 1 : -1])
    if not eligible:
        raise ValueError("No eligible checkpoint has at least one future observation.")

    if checkpoints is not None:
        checkpoint_policy = "explicit"
        unknown = pd.Index(checkpoints).difference(pd.Index(eligible))
        if len(unknown):
            raise ValueError(
                "Explicit checkpoints must be eligible historical timestamps with at "
                f"least one future row. Unknown/ineligible: {list(unknown[:5])}"
            )
        selected = sorted(set(checkpoints))
        if not selected:
            raise ValueError("checkpoints must not be empty.")
        return CheckpointPlan(selected=selected, eligible=eligible, policy="explicit")

    if checkpoint_policy not in {"uniform", "all", "explicit"}:
        raise ValueError("checkpoint_policy must be 'uniform', 'all', or 'explicit'.")
    if checkpoint_policy == "explicit":
        raise ValueError("checkpoint_policy='explicit' requires checkpoints=...")
    if checkpoint_policy == "all":
        return CheckpointPlan(selected=eligible, eligible=eligible, policy="all")
    count = min(n_checkpoints, len(eligible))
    chosen_positions = np.linspace(0, len(eligible) - 1, num=count, dtype=int)
    selected = [eligible[position] for position in np.unique(chosen_positions)]
    return CheckpointPlan(selected=selected, eligible=eligible, policy="uniform")


def choose_checkpoints(
    data: pd.DataFrame,
    *,
    time_level: TimeLevel = 0,
    checkpoints: Sequence[Any] | None = None,
    checkpoint_policy: CheckpointPolicy = "uniform",
    n_checkpoints: int = 12,
    min_history: int = 30,
) -> list[Any]:
    """Backward-compatible convenience wrapper returning selected checkpoints only."""

    return plan_checkpoints(
        data,
        time_level=time_level,
        checkpoints=checkpoints,
        checkpoint_policy=checkpoint_policy,
        n_checkpoints=n_checkpoints,
        min_history=min_history,
    ).selected


def _checkpoint_status(stats: ComparisonStats) -> AuditStatus:
    if stats.changed_cells or stats.structural_differences:
        return AuditStatus.FAIL
    if stats.compared_cells == 0:
        return AuditStatus.INCONCLUSIVE
    return AuditStatus.PASS


def _overall_prefix_status(
    comparisons: list[CheckpointComparison],
    *,
    exhaustive: bool,
) -> AuditStatus:
    if any(item.status is AuditStatus.ERROR for item in comparisons):
        return AuditStatus.ERROR
    if any(item.status is AuditStatus.FAIL for item in comparisons):
        return AuditStatus.FAIL
    conclusive = [item for item in comparisons if item.status is AuditStatus.PASS]
    if not conclusive:
        return AuditStatus.INCONCLUSIVE
    if any(item.status is AuditStatus.INCONCLUSIVE for item in comparisons):
        return AuditStatus.WARN
    if not exhaustive:
        return AuditStatus.WARN
    return AuditStatus.PASS


def _build_prefix_findings(
    *,
    status: AuditStatus,
    plan: CheckpointPlan,
    first_violation: str | None,
    total_changed: int,
    total_missing_left: int,
    total_missing_right: int,
    atol: float,
    rtol: float,
    audit_mode: str,
) -> list[Finding]:
    details = {
        "audit_mode": audit_mode,
        "checkpoint_policy": plan.policy,
        "eligible_checkpoints": len(plan.eligible),
        "checked_checkpoints": len(plan.selected),
        "coverage": plan.coverage,
        "exhaustive": plan.exhaustive,
        "atol": atol,
        "rtol": rtol,
    }
    if status is AuditStatus.PASS:
        return [
            Finding(
                code="TS-PREFIX-001",
                status=status,
                message=(
                    "Historical outputs are invariant at every eligible checkpoint "
                    "under the supplied audit contract."
                ),
                details=details,
            )
        ]
    if status is AuditStatus.WARN:
        return [
            Finding(
                code="TS-PREFIX-011",
                status=status,
                message=(
                    "No violation was found in the tested checkpoints, but the evidence "
                    "is sampled or contains unresolved checkpoints. Use "
                    "checkpoint_policy='all' for an exhaustive verdict."
                ),
                details=details,
            )
        ]
    if status is AuditStatus.INCONCLUSIVE:
        return [
            Finding(
                code="TS-PREFIX-021",
                status=status,
                message=(
                    "The audit produced no comparable labeled cells. Absence of a detected "
                    "difference is not evidence of temporal integrity."
                ),
                details=details,
            )
        ]
    if status is AuditStatus.FAIL:
        details.update(
            {
                "first_violation": first_violation,
                "changed_cells": total_changed,
                "missing_left": total_missing_left,
                "missing_right": total_missing_right,
            }
        )
        return [
            Finding(
                code="TS-PREFIX-101",
                status=status,
                message=(
                    "At least one historical output changed or changed structure when "
                    "future rows were added."
                ),
                details=details,
            )
        ]
    details["first_error"] = first_violation
    return [
        Finding(
            code="TS-PREFIX-201",
            status=status,
            message="The audit could not complete because at least one checkpoint errored.",
            details=details,
        )
    ]


def audit_prefix_invariance(
    transform: Transform,
    data: pd.DataFrame,
    *,
    time_level: TimeLevel = 0,
    checkpoints: Sequence[Any] | None = None,
    checkpoint_policy: CheckpointPolicy = "uniform",
    n_checkpoints: int = 12,
    min_history: int = 30,
    atol: float = 1e-10,
    rtol: float = 1e-8,
    copy_input: bool = True,
    transform_name: str | None = None,
) -> PrefixAuditResult:
    """Test whether future rows change outputs already published in the past.

    ``PASS`` is reserved for exhaustive checks. A clean sampled audit returns ``WARN``
    with explicit coverage metadata.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas.DataFrame.")
    if data.empty:
        raise ValueError("data must not be empty.")
    if not data.index.is_unique:
        raise ValueError("data index must be unique; use a (time, asset) MultiIndex for panels.")

    plan = plan_checkpoints(
        data,
        time_level=time_level,
        checkpoints=checkpoints,
        checkpoint_policy=checkpoint_policy,
        n_checkpoints=n_checkpoints,
        min_history=min_history,
    )
    name = transform_name or getattr(transform, "__name__", transform.__class__.__name__)

    try:
        # Always isolate the caller's frame. ``copy_input`` is retained for API
        # compatibility; temporal evidence must never depend on transform mutation.
        full_input = data.copy(deep=True)
        full_output = ensure_pandas_output(transform(full_input), name=name)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        findings = [
            Finding(
                code="TS-PREFIX-201",
                status=AuditStatus.ERROR,
                message="The transform failed on the full reference dataset.",
                details={"error": error},
            )
        ]
        return PrefixAuditResult(
            status=AuditStatus.ERROR,
            transform_name=name,
            checkpoints=[],
            first_violation=None,
            total_changed_cells=0,
            total_compared_cells=0,
            eligible_checkpoints=len(plan.eligible),
            checked_checkpoints=0,
            checkpoint_policy=plan.policy,
            exhaustive=plan.exhaustive,
            findings=findings,
            metadata={"audit_mode": "transform", "full_reference_error": error},
        )

    comparisons: list[CheckpointComparison] = []
    total_changed = 0
    total_compared = 0
    total_missing_left = 0
    total_missing_right = 0
    first_violation: str | None = None

    for checkpoint in plan.selected:
        checkpoint_text = stable_checkpoint_repr(checkpoint)
        try:
            prefix = prefix_slice(data, checkpoint, time_level)
            prefix_output = ensure_pandas_output(transform(prefix), name=name)
            left = point_slice(prefix_output, checkpoint, time_level)
            right = point_slice(full_output, checkpoint, time_level)
            stats = compare_labeled_outputs(left, right, atol=atol, rtol=rtol)

            checkpoint_status = _checkpoint_status(stats)
            if checkpoint_status is AuditStatus.FAIL and first_violation is None:
                first_violation = checkpoint_text
            total_changed += stats.changed_cells
            total_compared += stats.compared_cells
            total_missing_left += stats.missing_left
            total_missing_right += stats.missing_right
            comparisons.append(
                CheckpointComparison(
                    checkpoint=checkpoint_text,
                    status=checkpoint_status,
                    compared_cells=stats.compared_cells,
                    changed_cells=stats.changed_cells,
                    changed_fraction=stats.changed_fraction,
                    missing_left=stats.missing_left,
                    missing_right=stats.missing_right,
                    max_abs_diff=stats.max_abs_diff,
                    max_rel_diff=stats.max_rel_diff,
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
        audit_mode="transform",
    )
    return PrefixAuditResult(
        status=status,
        transform_name=name,
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
            "audit_mode": "transform",
            "time_level": time_level,
            "minimum_history": min_history,
            "atol": atol,
            "rtol": rtol,
            "coverage": plan.coverage,
            "input_isolation": True,
            "copy_input_argument": bool(copy_input),
        },
    )


def audit_determinism(
    transform: Transform,
    data: pd.DataFrame,
    *,
    repetitions: int = 3,
    atol: float = 1e-10,
    rtol: float = 1e-8,
    transform_name: str | None = None,
) -> DeterminismAuditResult:
    """Run the same transform repeatedly and verify identical labeled outputs."""

    repetitions = _positive_integer(repetitions, name="repetitions")
    if repetitions < 2:
        raise ValueError("repetitions must be at least 2.")
    name = transform_name or getattr(transform, "__name__", transform.__class__.__name__)
    try:
        reference = ensure_pandas_output(transform(data.copy(deep=True)), name=name)
    except Exception as exc:  # noqa: BLE001
        finding = Finding(
            code="TS-DETERMINISM-201",
            status=AuditStatus.ERROR,
            message="The reference run failed; determinism could not be evaluated.",
            details={"error": f"{type(exc).__name__}: {exc}"},
        )
        return DeterminismAuditResult(
            status=AuditStatus.ERROR,
            repetitions=repetitions,
            changed_cells=0,
            compared_cells=0,
            findings=[finding],
            metadata={"transform_name": name, "error_run": 0},
        )

    total_changed = 0
    total_compared = 0
    total_missing_left = 0
    total_missing_right = 0
    errors: list[str] = []

    for repetition in range(1, repetitions):
        try:
            candidate = ensure_pandas_output(
                transform(data.copy(deep=True)), name=name
            )
            stats = compare_labeled_outputs(reference, candidate, atol=atol, rtol=rtol)
            total_changed += stats.changed_cells
            total_compared += stats.compared_cells
            total_missing_left += stats.missing_left
            total_missing_right += stats.missing_right
        except Exception as exc:  # noqa: BLE001
            errors.append(f"run {repetition}: {type(exc).__name__}: {exc}")

    structural = total_missing_left + total_missing_right
    if errors:
        status = AuditStatus.ERROR
        code = "TS-DETERMINISM-201"
        message = "At least one repeated run failed; determinism is unresolved."
    elif total_changed or structural:
        status = AuditStatus.FAIL
        code = "TS-DETERMINISM-101"
        message = "Repeated runs produced different values or labeled output structure."
    elif total_compared == 0:
        status = AuditStatus.INCONCLUSIVE
        code = "TS-DETERMINISM-021"
        message = "Repeated runs produced no comparable labeled cells."
    else:
        status = AuditStatus.PASS
        code = "TS-DETERMINISM-001"
        message = "Repeated runs are deterministic."

    findings = [
        Finding(
            code=code,
            status=status,
            message=message,
            details={
                "repetitions": repetitions,
                "changed_cells": total_changed,
                "missing_left": total_missing_left,
                "missing_right": total_missing_right,
                "errors": errors,
            },
        )
    ]
    return DeterminismAuditResult(
        status=status,
        repetitions=repetitions,
        changed_cells=total_changed,
        compared_cells=total_compared,
        missing_left=total_missing_left,
        missing_right=total_missing_right,
        findings=findings,
        metadata={"transform_name": name, "errors": errors},
    )

