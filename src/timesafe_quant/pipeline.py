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
from .prefix import CheckpointPolicy, plan_checkpoints
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


@dataclass
class PipelineRunResult:
    """Outputs captured as immutable stage snapshots until the first error."""

    outputs: dict[str, PandasObject]
    error_stage: str | None = None
    error: str | None = None


class PipelineStageError(RuntimeError):
    """Raised by :meth:`CausalPipeline.run` with the true failing stage."""

    def __init__(self, stage: str, message: str):
        super().__init__(f"Pipeline stage {stage!r} failed: {message}")
        self.stage = stage


class CausalPipeline:
    """Minimal named pipeline used to localize the first non-causal stage."""

    def __init__(self, steps: Sequence[PipelineStep]):
        if not steps:
            raise ValueError("A pipeline must contain at least one step.")
        names = [step.name for step in steps]
        if len(names) != len(set(names)):
            raise ValueError("Pipeline step names must be unique.")
        self.steps = list(steps)

    def run_detailed(self, data: pd.DataFrame) -> PipelineRunResult:
        """Run stage by stage, snapshotting outputs and preserving the failing stage."""

        current: PandasObject = data.copy(deep=True)
        outputs: dict[str, PandasObject] = {}
        for step in self.steps:
            try:
                stage_input = current.copy(deep=True)
                current = ensure_pandas_output(step.function(stage_input), name=step.name)
                snapshot = current.copy(deep=True)
                outputs[step.name] = snapshot
                current = snapshot.copy(deep=True)
            except Exception as exc:  # noqa: BLE001
                return PipelineRunResult(
                    outputs=outputs,
                    error_stage=step.name,
                    error=f"{type(exc).__name__}: {exc}",
                )
        return PipelineRunResult(outputs=outputs)

    def run(self, data: pd.DataFrame) -> dict[str, PandasObject]:
        result = self.run_detailed(data)
        if result.error_stage is not None:
            raise PipelineStageError(result.error_stage, result.error or "unknown error")
        return result.outputs


def audit_pipeline_prefix_invariance(
    pipeline: CausalPipeline,
    data: pd.DataFrame,
    *,
    time_level: TimeLevel = 0,
    checkpoints: Sequence[Any] | None = None,
    checkpoint_policy: CheckpointPolicy = "uniform",
    n_checkpoints: int = 12,
    min_history: int = 30,
    atol: float = 1e-10,
    rtol: float = 1e-8,
) -> PipelineAuditResult:
    """Audit every stage and identify the earliest stage affected by future data."""

    plan = plan_checkpoints(
        data,
        time_level=time_level,
        checkpoints=checkpoints,
        checkpoint_policy=checkpoint_policy,
        n_checkpoints=n_checkpoints,
        min_history=min_history,
    )
    full_run = pipeline.run_detailed(data.copy(deep=True))

    state: dict[str, dict[str, Any]] = {
        step.name: {
            "violations": 0,
            "changed": 0,
            "compared": 0,
            "missing_left": 0,
            "missing_right": 0,
            "first": None,
            "errors": 0,
            "inconclusive": 0,
        }
        for step in pipeline.steps
    }

    for checkpoint in plan.selected:
        checkpoint_text = stable_checkpoint_repr(checkpoint)
        prefix = prefix_slice(data, checkpoint, time_level)
        prefix_run = pipeline.run_detailed(prefix.copy(deep=True))

        for step in pipeline.steps:
            stage = state[step.name]
            if step.name not in full_run.outputs:
                if full_run.error_stage == step.name:
                    stage["errors"] += 1
                    if stage["first"] is None:
                        stage["first"] = checkpoint_text
                else:
                    stage["inconclusive"] += 1
                continue
            if step.name not in prefix_run.outputs:
                if prefix_run.error_stage == step.name:
                    stage["errors"] += 1
                    if stage["first"] is None:
                        stage["first"] = checkpoint_text
                else:
                    stage["inconclusive"] += 1
                continue

            try:
                left = point_slice(prefix_run.outputs[step.name], checkpoint, time_level)
                right = point_slice(full_run.outputs[step.name], checkpoint, time_level)
                stats = compare_labeled_outputs(left, right, atol=atol, rtol=rtol)
                stage["changed"] += stats.changed_cells
                stage["compared"] += stats.compared_cells
                stage["missing_left"] += stats.missing_left
                stage["missing_right"] += stats.missing_right
                if stats.changed_cells or stats.structural_differences:
                    stage["violations"] += 1
                    if stage["first"] is None:
                        stage["first"] = checkpoint_text
                elif stats.compared_cells == 0:
                    stage["inconclusive"] += 1
            except Exception:  # noqa: BLE001
                stage["errors"] += 1
                if stage["first"] is None:
                    stage["first"] = checkpoint_text

    summaries: list[PipelineStageSummary] = []
    first_failing_stage: str | None = None
    first_error_stage: str | None = None
    statuses: list[AuditStatus] = []

    for step in pipeline.steps:
        stage = state[step.name]
        if stage["errors"]:
            status = AuditStatus.ERROR
        elif stage["violations"]:
            status = AuditStatus.FAIL
        elif stage["compared"] == 0:
            status = AuditStatus.INCONCLUSIVE
        elif stage["inconclusive"] or not plan.exhaustive:
            status = AuditStatus.WARN
        else:
            status = AuditStatus.PASS

        if status in {AuditStatus.FAIL, AuditStatus.ERROR} and first_failing_stage is None:
            first_failing_stage = step.name
        if status is AuditStatus.ERROR and first_error_stage is None:
            first_error_stage = step.name
        statuses.append(status)
        summaries.append(
            PipelineStageSummary(
                stage=step.name,
                status=status,
                violating_checkpoints=stage["violations"],
                changed_cells=stage["changed"],
                compared_cells=stage["compared"],
                first_violation=stage["first"],
                missing_left=stage["missing_left"],
                missing_right=stage["missing_right"],
                error_checkpoints=stage["errors"],
                inconclusive_checkpoints=stage["inconclusive"],
            )
        )

    if AuditStatus.ERROR in statuses:
        overall = AuditStatus.ERROR
    elif AuditStatus.FAIL in statuses:
        overall = AuditStatus.FAIL
    elif all(status is AuditStatus.INCONCLUSIVE for status in statuses):
        overall = AuditStatus.INCONCLUSIVE
    elif any(status in {AuditStatus.WARN, AuditStatus.INCONCLUSIVE} for status in statuses):
        overall = AuditStatus.WARN
    else:
        overall = AuditStatus.PASS

    if overall is AuditStatus.PASS:
        message = "All named pipeline stages are exhaustively prefix invariant."
        code = "TS-PIPELINE-001"
    elif overall is AuditStatus.WARN:
        message = (
            "No failing stage was found, but the audit is sampled or contains "
            "inconclusive stage checkpoints."
        )
        code = "TS-PIPELINE-011"
    elif overall is AuditStatus.INCONCLUSIVE:
        message = "The pipeline produced no comparable stage evidence."
        code = "TS-PIPELINE-021"
    elif overall is AuditStatus.FAIL:
        message = f"The first stage affected by future rows is {first_failing_stage!r}."
        code = "TS-PIPELINE-101"
    else:
        message = (
            f"An audit error first occurred at stage {first_error_stage!r}; "
            f"the first non-passing stage is {first_failing_stage!r}."
        )
        code = "TS-PIPELINE-201"

    findings = [
        Finding(
            code=code,
            status=overall,
            message=message,
            details={
                "checkpoint_policy": plan.policy,
                "eligible_checkpoints": len(plan.eligible),
                "checked_checkpoints": len(plan.selected),
                "coverage": plan.coverage,
                "exhaustive": plan.exhaustive,
                "first_failing_stage": first_failing_stage,
                "first_error_stage": first_error_stage,
                "full_reference_error_stage": full_run.error_stage,
                "full_reference_error": full_run.error,
            },
        )
    ]
    return PipelineAuditResult(
        status=overall,
        first_failing_stage=first_failing_stage,
        stages=summaries,
        first_error_stage=first_error_stage,
        findings=findings,
        metadata={
            "checkpoint_policy": plan.policy,
            "eligible_checkpoints": len(plan.eligible),
            "checked_checkpoints": len(plan.selected),
            "coverage": plan.coverage,
            "exhaustive": plan.exhaustive,
            "first_error_stage": first_error_stage,
        },
    )
