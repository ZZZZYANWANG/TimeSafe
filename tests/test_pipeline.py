
from timesafe_quant import (
    AuditStatus,
    CausalPipeline,
    PipelineStep,
    audit_pipeline_prefix_invariance,
)
from timesafe_quant.demo import make_demo_panel


def test_pipeline_localizes_first_leaking_stage():
    panel = make_demo_panel(sessions=60, assets=3)
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
        checkpoint_policy="all",
        min_history=25,
    )
    assert result.status is AuditStatus.FAIL
    assert result.first_failing_stage == "leaking_scale"
    assert result.stages[0].status is AuditStatus.PASS
    assert result.stages[1].status is AuditStatus.FAIL


def test_inplace_mutation_does_not_rewrite_prior_stage_snapshot():
    panel = make_demo_panel(sessions=50, assets=2)

    def leak_inplace(frame):
        frame.iloc[:, 0] = (frame.iloc[:, 0] - frame.iloc[:, 0].mean()) / frame.iloc[:, 0].std()
        return frame

    pipeline = CausalPipeline(
        [
            PipelineStep("clean", lambda frame: frame[["return"]]),
            PipelineStep("leak_inplace", leak_inplace),
        ]
    )
    result = audit_pipeline_prefix_invariance(
        pipeline,
        panel,
        time_level="date",
        checkpoint_policy="all",
        min_history=20,
    )
    assert result.first_failing_stage == "leak_inplace"
    assert result.stages[0].status is AuditStatus.PASS


def test_stage_exception_is_localized():
    panel = make_demo_panel(sessions=45, assets=2)
    cutoff = panel.index.get_level_values("date").unique()[25]

    def fragile(frame):
        if frame.index.get_level_values("date").max() <= cutoff:
            raise RuntimeError("short prefix")
        return frame

    pipeline = CausalPipeline(
        [
            PipelineStep("clean", lambda frame: frame[["return"]]),
            PipelineStep("fragile", fragile),
            PipelineStep("downstream", lambda frame: frame),
        ]
    )
    result = audit_pipeline_prefix_invariance(
        pipeline,
        panel,
        time_level="date",
        checkpoint_policy="all",
        min_history=15,
    )
    assert result.first_failing_stage == "fragile"
    assert result.stages[0].status is AuditStatus.PASS
    assert result.stages[1].status is AuditStatus.ERROR
    assert result.stages[2].status in {AuditStatus.WARN, AuditStatus.INCONCLUSIVE}


def test_duplicate_step_names_rejected():
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
