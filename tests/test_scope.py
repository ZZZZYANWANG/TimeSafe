import pandas as pd
from sklearn.linear_model import LinearRegression

from timesafe_quant import (
    AuditStatus,
    audit_prefix_invariance,
    audit_refit_prefix_invariance,
    sklearn_estimator_factory,
)
from timesafe_quant.demo import make_demo_panel


def test_prefit_external_state_is_outside_transform_audit_but_refit_detects_training_leak():
    panel = make_demo_panel(sessions=45, assets=2)
    mean = panel["return"].mean()
    std = panel["return"].std(ddof=0)

    def externally_prefit_transform(data: pd.DataFrame) -> pd.Series:
        return (data["return"] - mean) / std

    transform_result = audit_prefix_invariance(
        externally_prefit_transform,
        panel,
        time_level="date",
        checkpoint_policy="all",
        min_history=15,
    )
    assert transform_result.status is AuditStatus.PASS

    model_panel = panel.assign(
        x=panel["return"],
        y=panel["return"].groupby(level="asset").shift(-1).fillna(0.0),
    )

    def fit_predict(estimator, data: pd.DataFrame, checkpoint):
        estimator.fit(data[["x"]], data["y"])
        rows = data.index.get_level_values("date") == checkpoint
        return pd.Series(
            estimator.predict(data.loc[rows, ["x"]]),
            index=data.index[rows],
        )

    refit_result = audit_refit_prefix_invariance(
        sklearn_estimator_factory(LinearRegression()),
        fit_predict,
        model_panel,
        time_level="date",
        checkpoint_policy="all",
        min_history=15,
    )
    assert refit_result.status is AuditStatus.FAIL
