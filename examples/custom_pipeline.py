"""Minimal API example for a (date, asset) panel."""

from __future__ import annotations

import pandas as pd

from timesafe_quant import (
    audit_determinism,
    audit_prefix_invariance,
    make_report,
    write_report,
)


def my_factor(panel: pd.DataFrame) -> pd.Series:
    """Replace this function with your own feature or prediction pipeline."""

    return (
        panel["return"]
        .groupby(level="asset", group_keys=False)
        .ewm(span=20, adjust=False)
        .mean()
        .rename("factor")
    )


def audit(panel: pd.DataFrame) -> None:
    prefix = audit_prefix_invariance(
        my_factor,
        panel,
        time_level="date",
        checkpoint_policy="all",
        min_history=60,
    )
    determinism = audit_determinism(my_factor, panel, repetitions=3)
    report = make_report(
        title="My factor temporal integrity report",
        prefix=prefix,
        determinism=determinism,
    )
    write_report(report, "timesafe_report.md")
    write_report(report, "timesafe_report.json")
