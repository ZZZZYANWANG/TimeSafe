import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from timesafe_quant import (
    AuditStatus,
    audit_refit_prefix_invariance,
    sklearn_estimator_factory,
)


def make_panel(n=50):
    dates = pd.bdate_range("2024-01-02", periods=n)
    index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "asset"])
    x = np.arange(len(index), dtype=float)
    y = 0.2 * x + np.sin(x / 5)
    return pd.DataFrame({"x": x, "y": y}, index=index)


def leaking_fit_predict(estimator, data: pd.DataFrame, checkpoint):
    estimator.fit(data[["x"]], data["y"])
    rows = data.index.get_level_values("date") == checkpoint
    pred = estimator.predict(data.loc[rows, ["x"]])
    return pd.Series(pred, index=data.index[rows], name="prediction")


def causal_fit_predict(estimator, data: pd.DataFrame, checkpoint):
    dates = data.index.get_level_values("date")
    train = dates < checkpoint
    current = dates == checkpoint
    estimator.fit(data.loc[train, ["x"]], data.loc[train, "y"])
    pred = estimator.predict(data.loc[current, ["x"]])
    return pd.Series(pred, index=data.index[current], name="prediction")


def test_refit_audit_detects_full_sample_training():
    panel = make_panel()
    result = audit_refit_prefix_invariance(
        sklearn_estimator_factory(LinearRegression()),
        leaking_fit_predict,
        panel,
        time_level="date",
        checkpoint_policy="all",
        min_history=10,
    )
    assert result.status is AuditStatus.FAIL
    assert result.metadata["audit_mode"] == "refit"


def test_refit_audit_passes_checkpoint_restricted_training():
    panel = make_panel()
    result = audit_refit_prefix_invariance(
        sklearn_estimator_factory(LinearRegression()),
        causal_fit_predict,
        panel,
        time_level="date",
        checkpoint_policy="all",
        min_history=10,
    )
    assert result.status is AuditStatus.PASS


def test_refit_factory_must_create_fresh_state():
    panel = make_panel()
    estimator = LinearRegression()
    result = audit_refit_prefix_invariance(
        lambda: estimator,
        causal_fit_predict,
        panel,
        time_level="date",
        n_checkpoints=3,
        min_history=10,
    )
    assert result.status is AuditStatus.ERROR
