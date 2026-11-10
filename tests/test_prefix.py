import numpy as np
import pandas as pd

from timesafe_quant import AuditStatus, audit_determinism, audit_prefix_invariance
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


def test_causal_ema_passes_prefix_invariance():
    panel = make_demo_panel(sessions=100, assets=4)
    result = audit_prefix_invariance(
        causal_ema,
        panel,
        time_level="date",
        n_checkpoints=8,
        min_history=25,
    )
    assert result.status is AuditStatus.PASS
    assert result.first_violation is None
    assert result.total_changed_cells == 0


def test_global_zscore_fails_prefix_invariance():
    panel = make_demo_panel(sessions=100, assets=4)
    result = audit_prefix_invariance(
        global_zscore,
        panel,
        time_level="date",
        n_checkpoints=8,
        min_history=25,
    )
    assert result.status is AuditStatus.FAIL
    assert result.first_violation is not None
    assert result.total_changed_cells > 0


def test_centered_window_fails_prefix_invariance():
    panel = make_demo_panel(sessions=100, assets=4)
    result = audit_prefix_invariance(
        centered_window,
        panel,
        time_level="date",
        n_checkpoints=8,
        min_history=25,
    )
    assert result.status is AuditStatus.FAIL


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
