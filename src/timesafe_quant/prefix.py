"""Black-box temporal prefix-invariance auditing."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

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
)
from .compare import compare_labeled_outputs
from .types import (
    AuditStatus,
    CheckpointComparison,
    DeterminismAuditResult,
    Finding,
    PrefixAuditResult,
)

Transform = Callable[[pd.DataFrame], PandasObject]


def choose_checkpoints(
    data: pd.DataFrame,
    *,
    time_level: TimeLevel = 0,
    checkpoints: Sequence[Any] | None = None,
    n_checkpoints: int = 12,
    min_history: int = 30,
) -> list[Any]:
    """Choose evenly spaced historical checkpoints after a minimum history."""

    times = sorted_unique_times(data.index, time_level)
    if checkpoints is not None:
        unknown = pd.Index(checkpoints).difference(times)
        if len(unknown):
            raise ValueError(f"Unknown checkpoints: {list(unknown[:5])}")
        return sorted(set(checkpoints))

    if n_checkpoints <= 0:
        raise ValueError("n_checkpoints must be positive.")
    if min_history < 1:
        raise ValueError("min_history must be at least 1.")
    if len(times) <= min_history:
        raise ValueError(
            f"Need more than min_history={min_history} unique times; received {len(times)}."
        )

    eligible_positions = np.arange(min_history - 1, len(times))
    count = min(n_checkpoints, len(eligible_positions))
    chosen_positions = np.linspace(
        eligible_positions[0], eligible_positions[-1], num=count, dtype=int
    )
    return list(times[np.unique(chosen_positions)])


def audit_prefix_invariance(
    transform: Transform,
    data: pd.DataFrame,
    *,
    time_level: TimeLevel = 0,
    checkpoints: Sequence[Any] | None = None,
    n_checkpoints: int = 12,
    min_history: int = 30,
    atol: float = 1e-10,
    rtol: float = 1e-8,
    copy_input: bool = True,
    transform_name: str | None = None,
) -> PrefixAuditResult:
    """Test whether future rows change outputs that were already published in the past.

    The transform is run once on the full dataset and again on every historical prefix.
    For each checkpoint ``t``, TimeSafe compares ``F(D[:t])`` with ``F(D[:T]).loc[t]``.
    A causal, deterministic pipeline should produce identical values within tolerance.

    Notes
    -----
    The input and output must use a unique pandas index. For panel data, use a MultiIndex
    such as ``(date, asset)`` and pass ``time_level='date'``.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas.DataFrame.")
    if data.empty:
        raise ValueError("data must not be empty.")
    if not data.index.is_unique:
        raise ValueError("data index must be unique; use a (time, asset) MultiIndex for panels.")

    selected = choose_checkpoints(
        data,
        time_level=time_level,
        checkpoints=checkpoints,
        n_checkpoints=n_checkpoints,
        min_history=min_history,
    )
    name = transform_name or getattr(transform, "__name__", transform.__class__.__name__)

    full_input = data.copy(deep=True) if copy_input else data
    full_output = ensure_pandas_output(transform(full_input), name=name)

    comparisons: list[CheckpointComparison] = []
    total_changed = 0
    total_compared = 0
    first_violation: str | None = None
    findings: list[Finding] = []

    for checkpoint in selected:
        checkpoint_text = stable_checkpoint_repr(checkpoint)
        try:
            prefix = prefix_slice(data, checkpoint, time_level)
            prefix_input = prefix.copy(deep=True) if copy_input else prefix
            prefix_output = ensure_pandas_output(transform(prefix_input), name=name)
            left = point_slice(prefix_output, checkpoint, time_level)
            right = point_slice(full_output, checkpoint, time_level)
            stats = compare_labeled_outputs(left, right, atol=atol, rtol=rtol)

            changed = stats.changed_cells + stats.missing_left + stats.missing_right
            status = AuditStatus.PASS if changed == 0 else AuditStatus.FAIL
            if status is AuditStatus.FAIL and first_violation is None:
                first_violation = checkpoint_text
            total_changed += changed
            total_compared += stats.compared_cells
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
                )
            )
        except Exception as exc:  # noqa: BLE001 - audit must preserve the failing checkpoint
            if first_violation is None:
                first_violation = checkpoint_text
            comparisons.append(
                CheckpointComparison(
                    checkpoint=checkpoint_text,
                    status=AuditStatus.ERROR,
                    compared_cells=0,
                    changed_cells=0,
                    changed_fraction=0.0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    has_error = any(item.status is AuditStatus.ERROR for item in comparisons)
    has_failure = any(item.status is AuditStatus.FAIL for item in comparisons)
    if has_error:
        status = AuditStatus.ERROR
    elif has_failure:
        status = AuditStatus.FAIL
    else:
        status = AuditStatus.PASS

    if status is AuditStatus.PASS:
        findings.append(
            Finding(
                code="TS-PREFIX-001",
                status=AuditStatus.PASS,
                message="Historical outputs are invariant to the addition of future rows.",
                details={"checkpoints": len(selected), "atol": atol, "rtol": rtol},
            )
        )
    else:
        findings.append(
            Finding(
                code="TS-PREFIX-101",
                status=status,
                message=(
                    "At least one historical output changed when future rows were added. "
                    "The pipeline may contain look-ahead leakage, full-sample fitting, "
                    "centered windows, retrospective universe construction, or nondeterminism."
                ),
                details={
                    "first_violation": first_violation,
                    "changed_cells": total_changed,
                    "checkpoints": len(selected),
                },
            )
        )

    return PrefixAuditResult(
        status=status,
        transform_name=name,
        checkpoints=comparisons,
        first_violation=first_violation,
        total_changed_cells=total_changed,
        total_compared_cells=total_compared,
        findings=findings,
        metadata={
            "time_level": time_level,
            "checkpoints_requested": len(selected),
            "minimum_history": min_history,
            "atol": atol,
            "rtol": rtol,
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

    if repetitions < 2:
        raise ValueError("repetitions must be at least 2.")
    name = transform_name or getattr(transform, "__name__", transform.__class__.__name__)
    reference = ensure_pandas_output(transform(data.copy(deep=True)), name=name)
    total_changed = 0
    total_compared = 0
    findings: list[Finding] = []

    for _ in range(repetitions - 1):
        candidate = ensure_pandas_output(transform(data.copy(deep=True)), name=name)
        stats = compare_labeled_outputs(reference, candidate, atol=atol, rtol=rtol)
        total_changed += stats.changed_cells + stats.missing_left + stats.missing_right
        total_compared += stats.compared_cells

    status = AuditStatus.PASS if total_changed == 0 else AuditStatus.FAIL
    findings.append(
        Finding(
            code="TS-DETERMINISM-001" if status is AuditStatus.PASS else "TS-DETERMINISM-101",
            status=status,
            message=(
                "Repeated runs are deterministic."
                if status is AuditStatus.PASS
                else "Repeated runs produced different outputs; prefix failures may reflect randomness."
            ),
            details={"repetitions": repetitions, "changed_cells": total_changed},
        )
    )
    return DeterminismAuditResult(
        status=status,
        repetitions=repetitions,
        changed_cells=total_changed,
        compared_cells=total_compared,
        findings=findings,
    )
