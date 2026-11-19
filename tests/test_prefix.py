import numpy as np
import pandas as pd
import pytest

from timesafe_quant import (
    AuditStatus,
    audit_determinism,
    audit_prefix_invariance,
    plan_checkpoints,
)
from timesafe_quant.demo import make_demo_panel


def causal_ema(data: pd.DataFrame) -> pd.Series:
    return (
        data["return"]
        .groupby(level="asset", group_keys=False)
        .ewm(span=12, adjust=False)
        .mean()
    )


def global_zscore(data: pd.DataFrame) -> pd.Series:
    values = data["return"]
    return (values - values.mean()) / values.std(ddof=0)


def centered_window(data: pd.DataFrame) -> pd.Series:
    return (
        data["return"]
        .groupby(level="asset", group_keys=False)
        .rolling(9, center=True, min_periods=1)
        .mean()
    )


def random_transform(data: pd.DataFrame) -> pd.Series:
    return pd.Series(np.random.default_rng().normal(size=len(data)), index=data.index)


def empty_transform(data: pd.DataFrame) -> pd.Series:
    index = data.index[:0]
    return pd.Series(index=index, dtype=float)


def test_causal_ema_exhaustive_passes_prefix_invariance():
    panel = make_demo_panel(sessions=60, assets=3)
    result = audit_prefix_invariance(
        causal_ema,
        panel,
        time_level="date",
        checkpoint_policy="all",
        min_history=25,
    )
    assert result.status is AuditStatus.PASS
    assert result.exhaustive
    assert result.coverage == 1.0
    assert result.first_violation is None


def test_clean_sampled_audit_is_warn_not_pass():
    panel = make_demo_panel(sessions=100, assets=4)
    result = audit_prefix_invariance(
        causal_ema,
        panel,
        time_level="date",
        n_checkpoints=8,
        min_history=25,
    )
    assert result.status is AuditStatus.WARN
    assert 0 < result.coverage < 1


def test_global_zscore_fails_prefix_invariance():
    panel = make_demo_panel(sessions=60, assets=3)
    result = audit_prefix_invariance(
        global_zscore,
        panel,
        time_level="date",
        checkpoint_policy="all",
        min_history=25,
    )
    assert result.status is AuditStatus.FAIL
    assert result.first_violation is not None
    assert result.total_changed_cells > 0


def test_centered_window_fails_prefix_invariance():
    panel = make_demo_panel(sessions=60, assets=3)
    result = audit_prefix_invariance(
        centered_window,
        panel,
        time_level="date",
        checkpoint_policy="all",
        min_history=25,
    )
    assert result.status is AuditStatus.FAIL


def test_empty_output_is_inconclusive():
    panel = make_demo_panel(sessions=40, assets=2)
    result = audit_prefix_invariance(
        empty_transform,
        panel,
        time_level="date",
        checkpoint_policy="all",
        min_history=15,
    )
    assert result.status is AuditStatus.INCONCLUSIVE
    assert result.total_compared_cells == 0


def test_localized_leak_can_be_missed_by_sampling_but_caught_exhaustively():
    panel = make_demo_panel(sessions=70, assets=2)
    dates = panel.index.get_level_values("date").unique()
    leak_date = dates[31]

    def local_leak(data: pd.DataFrame) -> pd.Series:
        out = data["return"].copy()
        all_dates = data.index.get_level_values("date")
        if all_dates.max() > leak_date and leak_date in all_dates:
            mask = all_dates == leak_date
            out.loc[mask] = out.loc[mask] + float(data["return"].iloc[-1])
        return out

    sampled = audit_prefix_invariance(
        local_leak,
        panel,
        time_level="date",
        n_checkpoints=4,
        min_history=15,
    )
    exhaustive = audit_prefix_invariance(
        local_leak,
        panel,
        time_level="date",
        checkpoint_policy="all",
        min_history=15,
    )
    assert sampled.status in {AuditStatus.WARN, AuditStatus.FAIL}
    assert exhaustive.status is AuditStatus.FAIL


def test_nondeterministic_transform_fails_determinism():
    panel = make_demo_panel(sessions=40, assets=3)
    result = audit_determinism(random_transform, panel, repetitions=3)
    assert result.status is AuditStatus.FAIL
    assert result.changed_cells > 0


def test_unknown_checkpoint_is_rejected():
    panel = make_demo_panel(sessions=40, assets=3)
    with np.testing.assert_raises(ValueError):
        audit_prefix_invariance(
            causal_ema,
            panel,
            time_level="date",
            checkpoints=[pd.Timestamp("1990-01-01")],
        )


def test_duplicate_output_columns_are_rejected():
    panel = make_demo_panel(sessions=40, assets=3)

    def duplicate_columns(data):
        out = pd.concat([data[["return"]], data[["return"]]], axis=1)
        out.columns = ["x", "x"]
        return out

    result = audit_prefix_invariance(
        duplicate_columns,
        panel,
        time_level="date",
        checkpoint_policy="all",
        min_history=15,
    )
    assert result.status is AuditStatus.ERROR


@pytest.mark.parametrize(
    ("keyword", "value"),
    [("min_history", 2.5), ("n_checkpoints", True)],
)


def test_checkpoint_plan_rejects_non_integer_counts(keyword, value):
    panel = make_demo_panel(sessions=20, assets=2)
    kwargs = {"min_history": 3, "n_checkpoints": 4}
    kwargs[keyword] = value
    with pytest.raises(TypeError, match="positive integer"):
        plan_checkpoints(panel, time_level="date", **kwargs)
