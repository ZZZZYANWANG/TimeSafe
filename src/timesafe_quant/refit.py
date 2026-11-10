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

    plan = plan_checkpoints(
        data,
        time_level=time_level,
        checkpoints=checkpoints,
        checkpoint_policy=checkpoint_policy,
        n_checkpoints=n_checkpoints,
        min_history=min_history,
    )

    probe_left = estimator_factory()
    probe_right = estimator_factory()
    if probe_left is probe_right:
        finding = Finding(
            code="TS-REFIT-201",
            status=AuditStatus.ERROR,
            message="estimator_factory returned the same object twice; fresh state is required.",
            details={"audit_name": audit_name},
        )
        return PrefixAuditResult(
            status=AuditStatus.ERROR,
            transform_name=audit_name,
            checkpoints=[],
            first_violation=None,
            total_changed_cells=0,
            total_compared_cells=0,
            eligible_checkpoints=len(plan.eligible),
            checked_checkpoints=0,
            checkpoint_policy=plan.policy,
            exhaustive=plan.exhaustive,
            findings=[finding],
            metadata={"audit_mode": "refit", "factory_isolation": False},
        )
    del probe_left, probe_right

    comparisons: list[CheckpointComparison] = []
    total_changed = 0
    total_compared = 0
    total_missing_left = 0
    total_missing_right = 0
    first_violation: str | None = None

    for checkpoint in plan.selected:
        checkpoint_text = stable_checkpoint_repr(checkpoint)
        try:
            prefix_data = prefix_slice(data, checkpoint, time_level)
            prefix_input = prefix_data.copy(deep=True) if copy_input else prefix_data
            full_input = data.copy(deep=True) if copy_input else data

            prefix_estimator = estimator_factory()
            full_estimator = estimator_factory()
            if prefix_estimator is full_estimator:
                raise RuntimeError(
                    "estimator_factory returned shared state inside a checkpoint."
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
    findings.append(
        Finding(
            code="TS-REFIT-001",
            status=AuditStatus.PASS
            if status in {AuditStatus.PASS, AuditStatus.WARN}
            else status,
            message=(
                "Fresh estimator state was created independently for prefix and full-data runs."
            ),
            details={"factory_isolation": True},
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
            "factory_isolation": True,
            "time_level": time_level,
            "minimum_history": min_history,
            "coverage": plan.coverage,
        },
    )
