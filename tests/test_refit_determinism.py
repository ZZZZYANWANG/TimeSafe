import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from timesafe_quant import (
    AuditStatus,
    audit_refit_determinism,
    sklearn_estimator_factory,
)
from timesafe_quant.demo import make_demo_panel


def _deterministic_fit_predict(estimator, frame: pd.DataFrame, checkpoint):
    dates = frame.index.get_level_values("date")
    train = dates < checkpoint
    current = dates == checkpoint
    estimator.fit(frame.loc[train, ["return", "volume"]], frame.loc[train, "close"])
    prediction = estimator.predict(frame.loc[current, ["return", "volume"]])
    return pd.Series(prediction, index=frame.index[current], name="prediction")


class RandomOffsetEstimator:
    def fit(self, x, y):
        self.offset = float(np.random.default_rng().normal())
        return self

    def predict(self, x):
        return np.full(len(x), self.offset)


def test_refit_determinism_passes_for_deterministic_estimator():
    panel = make_demo_panel(sessions=35, assets=3)
    result = audit_refit_determinism(
        sklearn_estimator_factory(Ridge(alpha=1.0)),
        _deterministic_fit_predict,
        panel,
        time_level="date",
        checkpoint_policy="all",
        min_history=12,
        repetitions=3,
    )
    assert result.status is AuditStatus.PASS
    assert result.total_changed_cells == 0


def test_refit_determinism_fails_for_unseeded_training():
    panel = make_demo_panel(sessions=30, assets=3)

    def fit_predict(estimator, frame, checkpoint):
        dates = frame.index.get_level_values("date")
        train = dates < checkpoint
        current = dates == checkpoint
        estimator.fit(frame.loc[train, ["return"]], frame.loc[train, "close"])
        return pd.Series(
            estimator.predict(frame.loc[current, ["return"]]),
            index=frame.index[current],
        )

    result = audit_refit_determinism(
        RandomOffsetEstimator,
        fit_predict,
        panel,
        time_level="date",
        checkpoint_policy="uniform",
        n_checkpoints=5,
        min_history=10,
        repetitions=3,
    )
    assert result.status is AuditStatus.FAIL
    assert result.total_changed_cells > 0


def test_refit_determinism_empty_predictions_are_inconclusive():
    panel = make_demo_panel(sessions=25, assets=2)

    def empty_fit_predict(estimator, frame, checkpoint):
        _ = estimator, frame, checkpoint
        index = pd.MultiIndex.from_arrays([[], []], names=["date", "asset"])
        return pd.Series(dtype=float, index=index)

    result = audit_refit_determinism(
        lambda: object(),
        empty_fit_predict,
        panel,
        time_level="date",
        checkpoint_policy="all",
        min_history=10,
    )
    assert result.status is AuditStatus.INCONCLUSIVE
