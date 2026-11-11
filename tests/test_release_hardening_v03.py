import numpy as np
import pandas as pd
import pytest

from timesafe_quant import (
    AuditStatus,
    AvailabilityRule,
    CausalPipeline,
    LabelContract,
    PipelineStep,
    audit_availability_invariance,
    audit_label_maturity,
    audit_pipeline_prefix_invariance,
    audit_refit_availability_invariance,
    audit_refit_determinism,
    audit_refit_prefix_invariance,
    build_label_intervals,
)
from timesafe_quant.compare import compare_labeled_outputs
from timesafe_quant.demo import make_demo_panel
from timesafe_quant.timing import canonical_calendar, shift_sessions


def test_shift_sessions_never_revives_missing_source_position():
    calendar = pd.bdate_range("2024-01-02", periods=6)
    shifted = shift_sessions(calendar, np.array([-1, 0, 5]), 3)
    assert pd.isna(shifted[0])
    assert shifted[1] == calendar[3]
    assert pd.isna(shifted[2])


def test_calendar_rejects_missing_session():
    with pytest.raises(ValueError, match="missing sessions"):
        canonical_calendar(pd.DatetimeIndex([pd.Timestamp("2024-01-02"), pd.NaT]))


def test_complex_numeric_comparison_preserves_imaginary_component():
    left = pd.Series([1 + 1j], index=["x"])
    right = pd.Series([1 + 2j], index=["x"])
    result = compare_labeled_outputs(left, right)
    assert result.compared_cells == 1
    assert result.changed_cells == 1
    assert result.max_abs_diff == pytest.approx(1.0)


class _Dummy:
    pass


def _constant_prediction(estimator, frame: pd.DataFrame, checkpoint):
    _ = estimator
    current = frame.index.get_level_values("date") == checkpoint
    return pd.Series(0.0, index=frame.index[current], name="prediction")


def _cycling_factory():
    objects = [_Dummy(), _Dummy(), _Dummy()]
    state = {"i": 0}

    def factory():
        item = objects[state["i"] % len(objects)]
        state["i"] += 1
        return item

    return factory


def test_refit_audit_rejects_object_reuse_across_checkpoints():
    panel = make_demo_panel(sessions=24, assets=2)
    result = audit_refit_prefix_invariance(
        _cycling_factory(),
        _constant_prediction,
        panel,
        time_level="date",
        checkpoint_policy="all",
        min_history=8,
    )
    assert result.status is AuditStatus.ERROR
    assert any("reused an estimator object" in (item.error or "") for item in result.checkpoints)
    isolation = next(item for item in result.findings if item.code == "TS-REFIT-201")
    assert isolation.status is AuditStatus.ERROR
    assert result.metadata["factory_isolation"] is False


def test_refit_determinism_rejects_cross_checkpoint_object_reuse():
    panel = make_demo_panel(sessions=20, assets=2)
    result = audit_refit_determinism(
        _cycling_factory(),
        _constant_prediction,
        panel,
        time_level="date",
        checkpoint_policy="all",
        min_history=8,
        repetitions=2,
    )
    assert result.status is AuditStatus.ERROR


def test_refit_availability_rejects_cross_checkpoint_object_reuse():
    panel = make_demo_panel(sessions=20, assets=2).copy()
    panel["future"] = panel["return"]
    dates = panel.index.get_level_values("date")
    panel["available_at"] = dates
    result = audit_refit_availability_invariance(
        _cycling_factory(),
        _constant_prediction,
        panel,
        rules=[AvailabilityRule("future", "available_at", available_event="after_close")],
        time_level="date",
        checkpoint_policy="all",
        min_history=8,
    )
    assert result.status is AuditStatus.ERROR


def test_availability_rule_rejects_non_scalar_mask_value():
    with pytest.raises(TypeError, match="scalar"):
        AvailabilityRule("x", "available_at", mask_value=[0.0])


def test_unexercised_availability_checkpoint_is_not_pass():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    panel = pd.DataFrame(
        {"x": np.arange(10.0), "available_at": dates},
        index=pd.Index(dates, name="date"),
    )
    result = audit_availability_invariance(
        lambda frame: frame["x"],
        panel,
        rules=[AvailabilityRule("x", "available_at", available_event="open")],
        time_level="date",
        checkpoints=[dates[5]],
        min_history=3,
        decision_event="close",
    )
    assert result.status is AuditStatus.INCONCLUSIVE
    assert result.checkpoints[0].status is AuditStatus.INCONCLUSIVE
    assert result.checkpoints[0].metadata["masked_cells"] == 0


def test_maturity_without_as_of_is_warn_not_pass():
    calendar = pd.bdate_range("2024-01-02", periods=20)
    contract = LabelContract(holding_period_sessions=3)
    intervals = build_label_intervals(calendar[:10], calendar=calendar, contract=contract)
    result = audit_label_maturity(intervals, contract=contract, calendar=calendar)
    assert result.status is AuditStatus.WARN
    codes = {item.code for item in result.findings}
    assert "TS-MATURITY-011" in codes
    assert "TS-MATURITY-012" in codes
    assert "TS-MATURITY-001" not in codes
    assert not any(
        "mature at the declared audit time" in item.message
        for item in result.findings
    )


def test_pipeline_reports_true_error_stage_after_earlier_leak():
    panel = make_demo_panel(sessions=24, assets=2)

    def fail_only_on_prefix(frame):
        dates = frame.index.get_level_values("date")
        if len(pd.Index(dates).unique()) < 24:
            raise RuntimeError("prefix-only error")
        return frame

    pipeline = CausalPipeline(
        [
            PipelineStep("leak", lambda frame: (frame - frame.mean()) / frame.std()),
            PipelineStep("later_error", fail_only_on_prefix),
        ]
    )
    result = audit_pipeline_prefix_invariance(
        pipeline,
        panel[["return"]],
        time_level="date",
        checkpoint_policy="all",
        min_history=8,
    )
    assert result.status is AuditStatus.ERROR
    assert result.first_failing_stage == "leak"
    assert result.first_error_stage == "later_error"
    assert result.findings[0].details["first_error_stage"] == "later_error"


def test_object_cells_with_list_values_compare_without_missingness_error():
    left = pd.Series([[1, 2], [3, 4]], index=["a", "b"], dtype=object)
    same = pd.Series([[1, 2], [3, 4]], index=["a", "b"], dtype=object)
    changed = pd.Series([[1, 2], [3, 5]], index=["a", "b"], dtype=object)

    from timesafe_quant.compare import compare_labeled_outputs

    same_stats = compare_labeled_outputs(left, same)
    changed_stats = compare_labeled_outputs(left, changed)
    assert same_stats.changed_cells == 0
    assert changed_stats.changed_cells == 1


def test_maturity_without_as_of_describes_incomplete_tail_as_structural_failure():
    calendar = pd.bdate_range("2024-01-02", periods=20)
    contract = LabelContract(holding_period_sessions=5)
    intervals = build_label_intervals(calendar, calendar=calendar, contract=contract)

    result = audit_label_maturity(
        intervals,
        contract=contract,
        calendar=calendar,
    )

    assert result.status is AuditStatus.FAIL
    finding = next(item for item in result.findings if item.code == "TS-MATURITY-102")
    assert "structurally incomplete" in finding.message
    assert finding.details["as_of_declared"] is False


def test_zero_atol_keeps_relative_difference_finite_for_equal_zeros():
    from timesafe_quant.compare import compare_labeled_outputs

    left = pd.Series([0.0], index=["a"])
    right = pd.Series([0.0], index=["a"])
    stats = compare_labeled_outputs(left, right, atol=0.0, rtol=0.0)

    assert stats.changed_cells == 0
    assert stats.max_rel_diff == 0.0


def test_object_cells_with_equal_ndarrays_compare_equal():
    left = pd.Series([np.array([1.0, np.nan])], index=["a"], dtype=object)
    right = pd.Series([np.array([1.0, np.nan])], index=["a"], dtype=object)
    changed = pd.Series([np.array([1.0, 2.0])], index=["a"], dtype=object)

    assert compare_labeled_outputs(left, right).changed_cells == 0
    assert compare_labeled_outputs(left, changed).changed_cells == 1


def test_maturity_rejects_negative_embargo_even_without_masks():
    calendar = pd.bdate_range("2024-01-02", periods=20)
    contract = LabelContract(holding_period_sessions=3)
    intervals = build_label_intervals(calendar[:10], calendar=calendar, contract=contract)
    with pytest.raises(ValueError, match="embargo_sessions must be non-negative"):
        audit_label_maturity(
            intervals,
            contract=contract,
            calendar=calendar,
            embargo_sessions=-1,
        )


def test_availability_metadata_columns_cannot_be_mask_targets():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    panel = pd.DataFrame(
        {"x": np.arange(10.0), "available_at": dates},
        index=pd.Index(dates, name="date"),
    )
    rules = [
        AvailabilityRule("available_at", "available_at"),
        AvailabilityRule("x", "available_at"),
    ]
    with pytest.raises(ValueError, match="metadata columns cannot also be masked"):
        from timesafe_quant import mask_unavailable_fields

        mask_unavailable_fields(
            panel,
            checkpoint=dates[5],
            rules=rules,
            time_level="date",
        )


def test_refit_factory_isolation_is_inconclusive_when_factory_stops_early():
    panel = make_demo_panel(sessions=20, assets=2)

    class Dummy:
        pass

    calls = {"count": 0}

    def incomplete_factory():
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("factory stopped")
        return Dummy()

    result = audit_refit_prefix_invariance(
        incomplete_factory,
        _constant_prediction,
        panel,
        time_level="date",
        checkpoints=[panel.index.get_level_values("date").unique()[10]],
        min_history=5,
    )
    finding = next(item for item in result.findings if item.code == "TS-REFIT-021")
    assert finding.status is AuditStatus.INCONCLUSIVE
    assert finding.details["factory_isolation"] is None
