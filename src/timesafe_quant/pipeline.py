"""Stage-aware causal pipeline diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
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
from .prefix import choose_checkpoints
from .types import (
    AuditStatus,
    Finding,
    PipelineAuditResult,
    PipelineStageSummary,
)


@dataclass(frozen=True)
class PipelineStep:
    """One named pandas-to-pandas transformation."""

    name: str
    function: Callable[[PandasObject], PandasObject]


class CausalPipeline:
    """Minimal named pipeline used to localize the first non-causal stage."""

    def __init__(self, steps: Sequence[PipelineStep]):
        if not steps:
            raise ValueError("A pipeline must contain at least one step.")
        names = [step.name for step in steps]
        if len(names) != len(set(names)):
            raise ValueError("Pipeline step names must be unique.")
        self.steps = list(steps)

    def run(self, data: pd.DataFrame) -> dict[str, PandasObject]:
        current: PandasObject = data
        outputs: dict[str, PandasObject] = {}
        for step in self.steps:
            current = ensure_pandas_output(step.function(current), name=step.name)
            outputs[step.name] = current
        return outputs


def audit_pipeline_prefix_invariance(
    pipeline: CausalPipeline,
    data: pd.DataFrame,
    *,
    time_level: TimeLevel = 0,
    checkpoints: Sequence[Any] | None = None,
    n_checkpoints: int = 12,
    min_history: int = 30,
    atol: float = 1e-10,
    rtol: float = 1e-8,
) -> PipelineAuditResult:
    """Audit every stage and identify the earliest stage affected by future data."""

    selected = choose_checkpoints(
        data,
        time_level=time_level,
        checkpoints=checkpoints,
        n_checkpoints=n_checkpoints,
        min_history=min_history,
    )
    full_outputs = pipeline.run(data.copy(deep=True))

    state: dict[str, dict[str, Any]] = {
        step.name: {
            "violations": 0,
            "changed": 0,
            "compared": 0,
            "first": None,
            "errors": 0,
        }
        for step in pipeline.steps
    }

    for checkpoint in selected:
        checkpoint_text = stable_checkpoint_repr(checkpoint)
        prefix = prefix_slice(data, checkpoint, time_level)
        try:
            prefix_outputs = pipeline.run(prefix.copy(deep=True))
        except Exception:  # noqa: BLE001 - stage-specific errors are counted below when possible
            for step in pipeline.steps:
                state[step.name]["errors"] += 1
                if state[step.name]["first"] is None:
                    state[step.name]["first"] = checkpoint_text
            continue

        upstream_failed = False
        for step in pipeline.steps:
            stage = state[step.name]
            try:
                left = point_slice(prefix_outputs[step.name], checkpoint, time_level)
                right = point_slice(full_outputs[step.name], checkpoint, time_level)
                stats = compare_labeled_outputs(left, right, atol=atol, rtol=rtol)
                changed = stats.changed_cells + stats.missing_left + stats.missing_right
                stage["changed"] += changed
                stage["compared"] += stats.compared_cells
                if changed:
                    stage["violations"] += 1
                    if stage["first"] is None:
                        stage["first"] = checkpoint_text
                    upstream_failed = True
                elif upstream_failed:
                    pass
            except Exception:  # noqa: BLE001
                stage["errors"] += 1
                if stage["first"] is None:
                    stage["first"] = checkpoint_text
                upstream_failed = True

    summaries: list[PipelineStageSummary] = []
    first_failing_stage: str | None = None
    overall = AuditStatus.PASS
    for step in pipeline.steps:
        stage = state[step.name]
        if stage["errors"]:
            status = AuditStatus.ERROR
        elif stage["violations"]:
            status = AuditStatus.FAIL
        else:
            status = AuditStatus.PASS
        if status is not AuditStatus.PASS and first_failing_stage is None:
            first_failing_stage = step.name
        if status is AuditStatus.ERROR:
            overall = AuditStatus.ERROR
        elif status is AuditStatus.FAIL and overall is AuditStatus.PASS:
            overall = AuditStatus.FAIL
        summaries.append(
            PipelineStageSummary(
                stage=step.name,
                status=status,
                violating_checkpoints=stage["violations"] + stage["errors"],
                changed_cells=stage["changed"],
                compared_cells=stage["compared"],
                first_violation=stage["first"],
            )
        )

    findings = [
        Finding(
            code="TS-PIPELINE-001" if overall is AuditStatus.PASS else "TS-PIPELINE-101",
            status=overall,
            message=(
                "All named pipeline stages are prefix invariant."
                if overall is AuditStatus.PASS
                else f"The first stage affected by future rows is {first_failing_stage!r}."
            ),
            details={"checkpoints": len(selected), "first_failing_stage": first_failing_stage},
        )
    ]
    return PipelineAuditResult(
        status=overall,
        first_failing_stage=first_failing_stage,
        stages=summaries,
        findings=findings,
    )
