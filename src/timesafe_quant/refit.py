"""Refit-aware temporal integrity auditing for trainable estimators."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd

from ._pandas import (
    PandasObject,
    TimeLevel,
    ensure_pandas_output,
    point_slice,
    prefix_slice,
    stable_checkpoint_repr,
    validate_temporal_order,
)
from .compare import compare_labeled_outputs
from .prefix import (
    CheckpointPolicy,
    _build_prefix_findings,
    _checkpoint_status,
    _overall_prefix_status,
    plan_checkpoints,
)
from .types import AuditStatus, CheckpointComparison, Finding, PrefixAuditResult

EstimatorFactory = Callable[[], Any]
FitPredict = Callable[[Any, pd.DataFrame, Any], PandasObject]




def _new_distinct_estimator(
    estimator_factory: EstimatorFactory,
    retained_estimators: list[Any],
    *,
    context: str,
) -> Any:
    """Create an estimator whose object identity has not appeared in this audit.

    Strong references are retained for the duration of the audit so Python cannot
    recycle object ids and conceal a factory that cycles previously returned state.
    This proves object separation, not that an arbitrary custom estimator is unfitted;
    use :func:`sklearn_estimator_factory` when scikit-learn clone semantics apply.
    """

    estimator = estimator_factory()
    if any(estimator is prior for prior in retained_estimators):
        raise RuntimeError(
            "estimator_factory reused an estimator object across audit calls "
            f"({context})."
        )
    retained_estimators.append(estimator)
    return estimator


def sklearn_estimator_factory(estimator: Any) -> EstimatorFactory:
    """Return a factory backed by :func:`sklearn.base.clone`.

    scikit-learn is an optional dependency. Install ``timesafe-quant[model]``.
    """

    try:
        from sklearn.base import clone
    except ImportError as exc:  # pragma: no cover - exercised without optional extra
        raise ImportError(
            "scikit-learn is required for sklearn_estimator_factory; "
            "install timesafe-quant[model]."
        ) from exc

    def factory() -> Any:
        return clone(estimator)

    return factory


def audit_refit_prefix_invariance(
    estimator_factory: EstimatorFactory,
    fit_predict: FitPredict,
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
    audit_name: str = "refit_pipeline",
) -> PrefixAuditResult:
    """Audit a trainable pipeline by rebuilding fresh estimator state.

    For each checkpoint, ``fit_predict`` is invoked twice with two fresh estimators:
    once on the historical prefix and once on the full dataset. A correct implementation
    must internally restrict fitting and prediction to information legal at the checkpoint,
    so both outputs agree.

    All trainable state must be created by ``estimator_factory``. External pre-fitted
    objects, global caches, or vendor data vintages remain outside the black-box guarantee.
    """

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
    first_violation: str | None = None
    retained_estimators: list[Any] = []

    for checkpoint in plan.selected:
        checkpoint_text = stable_checkpoint_repr(checkpoint)
        try:
            prefix_input = prefix_slice(data, checkpoint, time_level)
            # Full and prefix paths always receive private frames.  The legacy
            # copy_input argument is retained but cannot disable isolation.
            full_input = data.copy(deep=True)

            prefix_estimator = _new_distinct_estimator(
                estimator_factory,
                retained_estimators,
                context=f"checkpoint={checkpoint!r} path=prefix",
            )
            full_estimator = _new_distinct_estimator(
                estimator_factory,
                retained_estimators,
                context=f"checkpoint={checkpoint!r} path=full",
            )

            prefix_output = ensure_pandas_output(
                fit_predict(prefix_estimator, prefix_input, checkpoint),
                name=f"{audit_name}.prefix",
            )
            full_output = ensure_pandas_output(
                fit_predict(full_estimator, full_input, checkpoint),
                name=f"{audit_name}.full",
            )
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
        audit_mode="refit",
    )
    factory_reuse_detected = any(
        "estimator_factory reused an estimator object" in (item.error or "")
        for item in comparisons
    )
    expected_instances = 2 * len(plan.selected)
    if factory_reuse_detected:
        factory_status = AuditStatus.ERROR
        factory_code = "TS-REFIT-201"
        factory_message = "The estimator factory reused an object during the audit."
    elif len(retained_estimators) == expected_instances:
        factory_status = AuditStatus.PASS
        factory_code = "TS-REFIT-001"
        factory_message = (
            "Distinct estimator objects were created independently for every prefix "
            "and full-data run. For arbitrary custom factories, object separation "
            "does not by itself prove that returned objects are unfitted."
        )
    else:
        factory_status = AuditStatus.INCONCLUSIVE
        factory_code = "TS-REFIT-021"
        factory_message = (
            "Not every planned estimator instance was created because an upstream "
            "audit step failed; complete factory-isolation evidence is unavailable."
        )
    findings.append(
        Finding(
            code=factory_code,
            status=factory_status,
            message=factory_message,
            details={
                "factory_isolation": (
                    True if factory_status is AuditStatus.PASS else False if factory_reuse_detected else None
                ),
                "retained_estimator_instances": len(retained_estimators),
                "expected_estimator_instances": expected_instances,
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
            "audit_mode": "refit",
            "factory_isolation": not factory_reuse_detected,
            "time_level": time_level,
            "minimum_history": min_history,
            "coverage": plan.coverage,
            "input_isolation": True,
            "copy_input_argument": bool(copy_input),
            "retained_estimator_instances": len(retained_estimators),
            "custom_factory_unfitted_state_not_introspected": True,
        },
    )


def audit_refit_determinism(
    estimator_factory: EstimatorFactory,
    fit_predict: FitPredict,
    data: pd.DataFrame,
    *,
    time_level: TimeLevel = 0,
    checkpoints: Sequence[Any] | None = None,
    checkpoint_policy: CheckpointPolicy = "uniform",
    n_checkpoints: int = 12,
    min_history: int = 30,
    repetitions: int = 3,
    atol: float = 1e-10,
    rtol: float = 1e-8,
    audit_name: str = "refit_determinism",
) -> PrefixAuditResult:
    """Audit training reproducibility with fresh estimator state at each checkpoint."""

    if repetitions < 2:
        raise ValueError("repetitions must be at least 2.")
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("data must be a non-empty pandas.DataFrame.")
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
    first_violation: str | None = None
    retained_estimators: list[Any] = []

    for checkpoint in plan.selected:
        checkpoint_text = stable_checkpoint_repr(checkpoint)
        try:
            prefix_data = prefix_slice(data, checkpoint, time_level)
            outputs: list[PandasObject] = []
            for repetition in range(repetitions):
                estimator = _new_distinct_estimator(
                    estimator_factory,
                    retained_estimators,
                    context=(
                        f"determinism checkpoint={checkpoint!r} "
                        f"repetition={repetition}"
                    ),
                )
                output = ensure_pandas_output(
                    fit_predict(estimator, prefix_data.copy(deep=True), checkpoint),
                    name=f"{audit_name}.run_{repetition}",
                )
                outputs.append(point_slice(output, checkpoint, time_level))

            checkpoint_changed = 0
            checkpoint_compared = 0
            checkpoint_missing_left = 0
            checkpoint_missing_right = 0
            max_abs: float | None = None
            max_rel: float | None = None
            reference = outputs[0]
            for candidate in outputs[1:]:
                stats = compare_labeled_outputs(reference, candidate, atol=atol, rtol=rtol)
                checkpoint_changed += stats.changed_cells
                checkpoint_compared += stats.compared_cells
                checkpoint_missing_left += stats.missing_left
                checkpoint_missing_right += stats.missing_right
                if stats.max_abs_diff is not None:
                    max_abs = max(max_abs or 0.0, stats.max_abs_diff)
                if stats.max_rel_diff is not None:
                    max_rel = max(max_rel or 0.0, stats.max_rel_diff)

            if checkpoint_changed or checkpoint_missing_left or checkpoint_missing_right:
                checkpoint_status = AuditStatus.FAIL
            elif checkpoint_compared == 0:
                checkpoint_status = AuditStatus.INCONCLUSIVE
            else:
                checkpoint_status = AuditStatus.PASS
            if checkpoint_status is AuditStatus.FAIL and first_violation is None:
                first_violation = checkpoint_text

            total_changed += checkpoint_changed
            total_compared += checkpoint_compared
            total_missing_left += checkpoint_missing_left
            total_missing_right += checkpoint_missing_right
            comparisons.append(
                CheckpointComparison(
                    checkpoint=checkpoint_text,
                    status=checkpoint_status,
                    compared_cells=checkpoint_compared,
                    changed_cells=checkpoint_changed,
                    changed_fraction=(
                        checkpoint_changed / checkpoint_compared
                        if checkpoint_compared
                        else None
                    ),
                    missing_left=checkpoint_missing_left,
                    missing_right=checkpoint_missing_right,
                    max_abs_diff=max_abs,
                    max_rel_diff=max_rel,
                    metadata={"repetitions": repetitions},
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
                    metadata={"repetitions": repetitions},
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
        audit_mode="refit_determinism",
    )
    if status in {AuditStatus.PASS, AuditStatus.WARN}:
        findings.append(
            Finding(
                code="TS-REFIT-DETERMINISM-001",
                status=AuditStatus.PASS,
                message="Fresh-state refits were reproducible at the tested checkpoints.",
                details={"repetitions": repetitions},
            )
        )
    elif status is AuditStatus.FAIL:
        findings.append(
            Finding(
                code="TS-REFIT-DETERMINISM-101",
                status=status,
                message="Fresh-state refits produced different predictions.",
                details={"repetitions": repetitions},
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
            "audit_mode": "refit_determinism",
            "repetitions": repetitions,
            "coverage": plan.coverage,
            "retained_estimator_instances": len(retained_estimators),
            "custom_factory_unfitted_state_not_introspected": True,
        },
    )
