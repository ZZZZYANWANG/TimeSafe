"""Tiny callable used by the CLI availability example and tests."""

from __future__ import annotations

import pandas as pd


def read_materialized_target(panel: pd.DataFrame) -> pd.Series:
    return panel["future_target"].rename("prediction")
