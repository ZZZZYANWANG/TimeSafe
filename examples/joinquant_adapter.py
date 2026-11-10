"""JoinQuant-oriented example.

Run inside a JoinQuant research notebook. The import stays inside the loader so this file
can still be imported and tested outside JoinQuant.
"""

from __future__ import annotations

import pandas as pd

from timesafe_quant import audit_prefix_invariance


def load_joinquant_panel(securities: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Load a close-price panel and convert it to TimeSafe's (date, asset) index."""

    from jqdata import get_price  # type: ignore[import-not-found]  # JoinQuant runtime only

    raw = get_price(
        securities,
        start_date=start_date,
        end_date=end_date,
        frequency="daily",
        fields=["close", "volume", "money"],
        panel=False,
    )
    panel = raw.reset_index().rename(columns={"time": "date", "code": "asset"})
    panel["return"] = panel.groupby("asset")["close"].pct_change()
    return panel.set_index(["date", "asset"]).sort_index()


def causal_momentum(panel: pd.DataFrame) -> pd.Series:
    return (
        panel["close"]
        .groupby(level="asset", group_keys=False)
        .pct_change(20)
        .rename("momentum_20")
    )


def run_joinquant_audit(panel: pd.DataFrame):
    """Return a result that can be rendered in the notebook."""

    return audit_prefix_invariance(
        causal_momentum,
        panel,
        time_level="date",
        n_checkpoints=16,
        min_history=80,
    )
