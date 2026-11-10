"""Refit-aware audit for a scikit-learn estimator."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import Ridge

from timesafe_quant import (
    audit_refit_prefix_invariance,
    sklearn_estimator_factory,
)


def fit_predict(estimator, data: pd.DataFrame, checkpoint):
    dates = data.index.get_level_values("date")
    train = dates < checkpoint
    current = dates == checkpoint
    estimator.fit(data.loc[train, ["feature"]], data.loc[train, "target"])
    prediction = estimator.predict(data.loc[current, ["feature"]])
    return pd.Series(prediction, index=data.index[current], name="prediction")


def audit(panel: pd.DataFrame):
    return audit_refit_prefix_invariance(
        sklearn_estimator_factory(Ridge(alpha=10.0)),
        fit_predict,
        panel,
        time_level="date",
        checkpoint_policy="all",
        min_history=60,
    )
