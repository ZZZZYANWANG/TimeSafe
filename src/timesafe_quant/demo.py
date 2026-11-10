"""Deterministic demonstration datasets and pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .maturity import (
    LabelContract,
    audit_label_maturity,
    build_label_intervals,
    purged_walk_forward_masks,
)
from .prefix import audit_determinism, audit_prefix_invariance
from .report import make_report


def make_demo_panel(*, sessions: int = 180, assets: int = 8, seed: int = 7) -> pd.DataFrame:
    """Create a deterministic panel suitable for leakage demonstrations."""

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=sessions)
    symbols = [f"A{index:02d}" for index in range(assets)]
    index = pd.MultiIndex.from_product([dates, symbols], names=["date", "asset"])
    common = rng.normal(0.0, 0.006, size=sessions)
    idiosyncratic = rng.normal(0.0, 0.012, size=(sessions, assets))
    returns = common[:, None] + idiosyncratic
    price = 100.0 * np.exp(np.cumsum(returns, axis=0))
    volume = np.exp(rng.normal(12.0, 0.35, size=(sessions, assets)))
    return pd.DataFrame(
        {
            "return": returns.reshape(-1),
            "close": price.reshape(-1),
            "volume": volume.reshape(-1),
        },
        index=index,
    )


def causal_ema(data: pd.DataFrame) -> pd.Series:
    """Legal one-sided EMA computed independently per asset."""

    return (
        data["return"]
        .groupby(level="asset", group_keys=False)
        .ewm(span=12, adjust=False)
        .mean()
        .rename("causal_ema")
    )


def leaking_global_zscore(data: pd.DataFrame) -> pd.Series:
    """Illegal full-sample standardization used as a demonstration."""

    values = data["return"]
    return ((values - values.mean()) / values.std(ddof=0)).rename("global_zscore")


def run_demo_report():
    """Run a failing prefix audit and a passing label-maturity audit."""

    panel = make_demo_panel()
    prefix = audit_prefix_invariance(
        leaking_global_zscore,
        panel,
        time_level="date",
        checkpoint_policy="all",
        min_history=40,
    )
    determinism = audit_determinism(leaking_global_zscore, panel, repetitions=3)

    calendar = pd.Index(panel.index.get_level_values("date").unique()).sort_values()
    contract = LabelContract(
        execution_offset_sessions=1,
        holding_period_sessions=20,
        safety_gap_sessions=4,
    )
    intervals = build_label_intervals(calendar, calendar=calendar, contract=contract)
    train_mask, test_mask = purged_walk_forward_masks(
        intervals,
        test_start=calendar[-40],
        test_end=calendar[-21],
        contract=contract,
        calendar=calendar,
    )
    valid_intervals = intervals[intervals["interval_valid"]].reset_index(drop=True)
    maturity = audit_label_maturity(
        valid_intervals,
        contract=contract,
        as_of=calendar[-1],
        as_of_event="after_close",
        calendar=calendar,
    )
    split_result = audit_label_maturity(
        intervals,
        contract=contract,
        train_mask=train_mask,
        test_mask=test_mask,
        calendar=calendar,
    )
    return make_report(
        title="TimeSafe demonstration report",
        prefix=prefix,
        determinism=determinism,
        maturity=maturity,
        findings=split_result.findings,
        metadata={"demo": True, "expected_prefix_status": "FAIL"},
    )
