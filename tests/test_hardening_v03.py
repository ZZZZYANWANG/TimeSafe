import numpy as np
import pandas as pd
import pytest

from timesafe_quant import (
    AuditStatus,
    AvailabilityRule,
    LabelContract,
    audit_availability_invariance,
    audit_label_maturity,
    audit_prefix_invariance,
    audit_refit_prefix_invariance,
    build_label_intervals,
)
from timesafe_quant.compare import compare_labeled_outputs
from timesafe_quant.demo import make_demo_panel


def test_unsorted_temporal_input_is_rejected():
    panel = make_demo_panel(sessions=40, assets=2).sort_index(ascending=False)
    with pytest.raises(ValueError, match="sorted"):
        audit_prefix_invariance(
            lambda frame: frame["return"],
            panel,
            time_level="date",
            checkpoint_policy="all",
            min_history=10,
        )


def test_transform_cannot_mutate_callers_input_even_when_legacy_copy_flag_is_false():
    panel = make_demo_panel(sessions=35, assets=2)
    original = panel.copy(deep=True)

    def mutating_transform(frame: pd.DataFrame) -> pd.Series:
        frame.loc[:, "return"] = 0.0
        return frame["return"].rename("mutated")

    result = audit_prefix_invariance(
        mutating_transform,
        panel,
        time_level="date",
        checkpoint_policy="all",
        min_history=10,
        copy_input=False,
    )
    pd.testing.assert_frame_equal(panel, original)
    assert result.status is AuditStatus.PASS
    assert result.metadata["input_isolation"] is True


def test_refit_cannot_mutate_callers_input_when_legacy_copy_flag_is_false():
    panel = make_demo_panel(sessions=35, assets=2)
    panel["target"] = panel["return"]
    original = panel.copy(deep=True)

    class Dummy:
        pass

    def fit_predict(estimator, frame: pd.DataFrame, checkpoint):
        _ = estimator
        frame.loc[:, "return"] = 0.0
        current = frame.index.get_level_values("date") == checkpoint
        return pd.Series(0.0, index=frame.index[current], name="prediction")

    result = audit_refit_prefix_invariance(
        Dummy,
        fit_predict,
        panel,
        time_level="date",
        checkpoint_policy="all",
        min_history=10,
        copy_input=False,
    )
    pd.testing.assert_frame_equal(panel, original)
    assert result.status is AuditStatus.PASS


def test_compare_rejects_duplicate_labels_and_negative_tolerance():
    duplicate = pd.Series([1.0, 2.0], index=["x", "x"])
    with pytest.raises(ValueError, match="unique"):
        compare_labeled_outputs(duplicate, duplicate)
    clean = pd.Series([1.0], index=["x"])
    with pytest.raises(ValueError, match="non-negative"):
        compare_labeled_outputs(clean, clean, atol=-1.0)


def test_availability_counts_only_actual_perturbations():
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    panel = pd.DataFrame(
        {
            "value": np.nan,
            "available_at": dates + pd.Timedelta(days=2),
        },
        index=pd.Index(dates, name="date"),
    )
    result = audit_availability_invariance(
        lambda frame: frame["value"],
        panel,
        rules=[AvailabilityRule("value", "available_at")],
        time_level="date",
        checkpoint_policy="all",
        min_history=2,
    )
    assert result.status is AuditStatus.INCONCLUSIVE
    assert result.metadata["masked_cells"] == 0


def test_row_reference_detects_historical_feature_published_after_its_row():
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    panel = pd.DataFrame(
        {
            "feature": np.arange(12, dtype=float),
            "available_at": dates + pd.Timedelta(days=2),
        },
        index=pd.Index(dates, name="date"),
    )

    def expanding_signal(frame: pd.DataFrame) -> pd.Series:
        return frame["feature"].expanding().mean().rename("signal")

    checkpoint_basis = audit_availability_invariance(
        expanding_signal,
        panel,
        rules=[AvailabilityRule("feature", "available_at", reference="checkpoint")],
        time_level="date",
        checkpoint_policy="all",
        min_history=4,
    )
    row_basis = audit_availability_invariance(
        expanding_signal,
        panel,
        rules=[AvailabilityRule("feature", "available_at", reference="row")],
        time_level="date",
        checkpoint_policy="all",
        min_history=4,
    )
    assert checkpoint_basis.status is AuditStatus.FAIL  # recent rows are still hidden
    assert row_basis.status is AuditStatus.FAIL
    assert row_basis.metadata["masked_cells"] > checkpoint_basis.metadata["masked_cells"]


def test_duplicate_availability_targets_are_rejected():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    panel = pd.DataFrame(
        {"x": range(6), "available_at": dates},
        index=pd.Index(dates, name="date"),
    )
    rules = [
        AvailabilityRule("x", "available_at"),
        AvailabilityRule("x", "available_at"),
    ]
    result = audit_availability_invariance(
        lambda frame: frame["x"],
        panel,
        rules=rules,
        time_level="date",
        checkpoint_policy="all",
        min_history=2,
    )
    assert result.status is AuditStatus.ERROR


def test_empty_train_or_test_masks_do_not_receive_purge_pass():
    calendar = pd.bdate_range("2024-01-02", periods=50)
    contract = LabelContract(execution_offset_sessions=1, holding_period_sessions=5)
    intervals = build_label_intervals(calendar, calendar=calendar, contract=contract)

    valid = intervals["interval_valid"].to_numpy(dtype=bool)
    no_test = audit_label_maturity(
        intervals,
        contract=contract,
        as_of=calendar[-1],
        train_mask=valid,
        test_mask=np.zeros(len(intervals), dtype=bool),
        calendar=calendar,
    )
    assert no_test.status is AuditStatus.INCONCLUSIVE
    assert any(f.code == "TS-PURGE-021" for f in no_test.findings)

    no_train = audit_label_maturity(
        intervals,
        contract=contract,
        as_of=calendar[-1],
        train_mask=np.zeros(len(intervals), dtype=bool),
        test_mask=((intervals["decision_time"] < calendar[20]) & intervals["interval_valid"]).to_numpy(),
        calendar=calendar,
    )
    assert no_train.status is AuditStatus.INCONCLUSIVE
    assert any(f.code == "TS-PURGE-022" for f in no_train.findings)


def test_factory_isolation_finding_is_independently_pass_even_when_model_leaks():
    panel = make_demo_panel(sessions=40, assets=2)
    panel["target"] = panel["return"]

    class Dummy:
        pass

    def leaking_fit_predict(estimator, frame: pd.DataFrame, checkpoint):
        _ = estimator
        current = frame.index.get_level_values("date") == checkpoint
        value = float(frame["target"].mean())
        return pd.Series(value, index=frame.index[current], name="prediction")

    result = audit_refit_prefix_invariance(
        Dummy,
        leaking_fit_predict,
        panel,
        time_level="date",
        checkpoints=[panel.index.get_level_values("date").unique()[20]],
        min_history=10,
    )
    assert result.status is AuditStatus.FAIL
    isolation = next(f for f in result.findings if f.code == "TS-REFIT-001")
    assert isolation.status is AuditStatus.PASS


def test_clean_sampled_availability_has_pass_component_finding_and_warn_overall():
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    panel = pd.DataFrame(
        {
            "x": np.arange(12, dtype=float),
            "available_at": dates,
        },
        index=pd.Index(dates, name="date"),
    )

    def lagged(frame: pd.DataFrame) -> pd.Series:
        return frame["x"].shift(1).rename("signal")

    result = audit_availability_invariance(
        lagged,
        panel,
        rules=[AvailabilityRule("x", "available_at", available_event="after_close")],
        time_level="date",
        checkpoints=[dates[5], dates[8]],
        min_history=3,
        decision_event="close",
    )
    assert result.status is AuditStatus.WARN
    finding = next(f for f in result.findings if f.code == "TS-AVAILABILITY-001")
    assert finding.status is AuditStatus.PASS
