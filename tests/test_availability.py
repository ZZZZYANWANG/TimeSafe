import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from timesafe_quant import (
    AuditStatus,
    AvailabilityRule,
    audit_availability_invariance,
    audit_refit_availability_invariance,
    sklearn_estimator_factory,
)
from timesafe_quant.demo import make_demo_panel


def _panel_with_materialized_target(sessions: int = 45, assets: int = 4) -> pd.DataFrame:
    panel = make_demo_panel(sessions=sessions, assets=assets).copy()
    dates = pd.Index(panel.index.get_level_values("date").unique()).sort_values()
    date_to_position = {date: pos for pos, date in enumerate(dates)}
    target = (
        panel["return"]
        .groupby(level="asset")
        .shift(-3)
        .fillna(0.0)
        .rename("future_target")
    )
    panel["future_target"] = target
    availability = []
    for date in panel.index.get_level_values("date"):
        pos = date_to_position[date] + 3
        availability.append(dates[pos] if pos < len(dates) else pd.NaT)
    panel["target_available"] = availability
    return panel


def test_materialized_future_target_fails_availability_audit():
    panel = _panel_with_materialized_target()
    rule = AvailabilityRule("future_target", "target_available")
    result = audit_availability_invariance(
        lambda frame: frame["future_target"],
        panel,
        rules=[rule],
        time_level="date",
        checkpoint_policy="all",
        min_history=10,
    )
    assert result.status is AuditStatus.FAIL
    assert result.total_changed_cells > 0
    assert result.metadata["masked_cells"] > 0


def test_causal_transform_is_invariant_to_unavailable_target_masking():
    panel = _panel_with_materialized_target()
    rule = AvailabilityRule("future_target", "target_available")

    def causal(frame: pd.DataFrame) -> pd.Series:
        return (
            frame["return"]
            .groupby(level="asset", group_keys=False)
            .ewm(span=5, adjust=False)
            .mean()
        )

    result = audit_availability_invariance(
        causal,
        panel,
        rules=[rule],
        time_level="date",
        checkpoint_policy="all",
        min_history=10,
    )
    assert result.status is AuditStatus.PASS
    assert result.metadata["masked_cells"] > 0


def test_availability_audit_without_perturbation_is_inconclusive():
    panel = _panel_with_materialized_target()
    panel["target_available"] = panel.index.get_level_values("date")
    rule = AvailabilityRule(
        "future_target",
        "target_available",
        available_event="close",
    )
    result = audit_availability_invariance(
        lambda frame: frame["return"],
        panel,
        rules=[rule],
        time_level="date",
        checkpoint_policy="all",
        min_history=10,
        decision_event="close",
    )
    assert result.status is AuditStatus.INCONCLUSIVE
    assert result.metadata["masked_cells"] == 0


def _fit_predict_bad(estimator, frame: pd.DataFrame, checkpoint):
    dates = frame.index.get_level_values("date")
    train = (dates < checkpoint) & frame["future_target"].notna().to_numpy()
    current = dates == checkpoint
    estimator.fit(frame.loc[train, ["return", "volume"]], frame.loc[train, "future_target"])
    prediction = estimator.predict(frame.loc[current, ["return", "volume"]])
    return pd.Series(prediction, index=frame.index[current], name="prediction")


def _fit_predict_good(estimator, frame: pd.DataFrame, checkpoint):
    dates = frame.index.get_level_values("date")
    legal = frame["target_available"].notna() & (frame["target_available"] < checkpoint)
    train = (dates < checkpoint) & legal.to_numpy() & frame["future_target"].notna().to_numpy()
    current = dates == checkpoint
    estimator.fit(frame.loc[train, ["return", "volume"]], frame.loc[train, "future_target"])
    prediction = estimator.predict(frame.loc[current, ["return", "volume"]])
    return pd.Series(prediction, index=frame.index[current], name="prediction")


def test_refit_availability_catches_training_on_immature_labels():
    panel = _panel_with_materialized_target(sessions=55, assets=5)
    rule = AvailabilityRule("future_target", "target_available")
    factory = sklearn_estimator_factory(Ridge(alpha=1.0))
    result = audit_refit_availability_invariance(
        factory,
        _fit_predict_bad,
        panel,
        rules=[rule],
        time_level="date",
        checkpoint_policy="uniform",
        n_checkpoints=8,
        min_history=15,
    )
    assert result.status is AuditStatus.FAIL


def test_refit_availability_passes_when_training_filters_by_availability():
    panel = _panel_with_materialized_target(sessions=40, assets=4)
    rule = AvailabilityRule("future_target", "target_available")
    factory = sklearn_estimator_factory(Ridge(alpha=1.0))
    result = audit_refit_availability_invariance(
        factory,
        _fit_predict_good,
        panel,
        rules=[rule],
        time_level="date",
        checkpoint_policy="all",
        min_history=15,
    )
    assert result.status is AuditStatus.PASS
