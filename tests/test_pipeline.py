import pandas as pd

from timesafe_quant import (
    AuditStatus,
    CausalPipeline,
    PipelineStep,
    audit_pipeline_prefix_invariance,
)
from timesafe_quant.demo import make_demo_panel


def test_pipeline_localizes_first_leaking_stage():
    panel = make_demo_panel(sessions=90, assets=4)
    pipeline = CausalPipeline(
        [
            PipelineStep("raw", lambda frame: frame[["return"]]),
            PipelineStep("leaking_scale", lambda frame: (frame - frame.mean()) / frame.std()),
            PipelineStep("causal_ema", lambda frame: frame.ewm(span=10, adjust=False).mean()),
        ]
    )
    result = audit_pipeline_prefix_invariance(
        pipeline,
        panel,
        time_level="date",
        n_checkpoints=7,
        min_history=25,
    )
    assert result.status is AuditStatus.FAIL
    assert result.first_failing_stage == "leaking_scale"
    assert result.stages[0].status is AuditStatus.PASS
    assert result.stages[1].status is AuditStatus.FAIL


def test_duplicate_step_names_rejected():
    with pd.option_context("mode.copy_on_write", True):
        try:
            CausalPipeline(
                [
                    PipelineStep("x", lambda frame: frame),
                    PipelineStep("x", lambda frame: frame),
                ]
            )
        except ValueError:
            pass
        else:
            raise AssertionError("Expected duplicate names to be rejected")
