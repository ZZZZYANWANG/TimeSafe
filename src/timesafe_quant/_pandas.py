"""Internal pandas helpers."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TypeAlias

import numpy as np
import pandas as pd

PandasObject: TypeAlias = pd.Series | pd.DataFrame
TimeLevel: TypeAlias = str | int


def ensure_pandas_output(value: object, *, name: str = "transform") -> PandasObject:
    """Require a labeled pandas output so historical rows can be aligned safely."""

    if not isinstance(value, (pd.Series, pd.DataFrame)):
        raise TypeError(
            f"{name} must return pandas.Series or pandas.DataFrame; "
            f"received {type(value).__name__}."
        )
    if not value.index.is_unique:
        raise ValueError(f"{name} output index must be unique for unambiguous alignment.")
    if isinstance(value, pd.DataFrame) and not value.columns.is_unique:
        raise ValueError(f"{name} output columns must be unique for unambiguous alignment.")
    return value


def time_values(index: pd.Index, level: TimeLevel = 0) -> pd.Index:
    """Extract the temporal coordinate from an Index or MultiIndex."""

    if isinstance(index, pd.MultiIndex):
        values = index.get_level_values(level)
    else:
        if level not in (0, index.name, None):
            raise ValueError(
                f"Single-level index does not contain requested time level {level!r}."
            )
        values = index
    return pd.Index(values)


def sorted_unique_times(index: pd.Index, level: TimeLevel = 0) -> pd.Index:
    """Return sorted unique time values."""

    values = time_values(index, level)
    try:
        return pd.Index(values.unique()).sort_values()
    except TypeError as exc:
        raise TypeError("Time values must be sortable.") from exc


def prefix_slice(data: pd.DataFrame, checkpoint: Hashable, level: TimeLevel = 0) -> pd.DataFrame:
    """Slice all observations whose time coordinate is <= checkpoint."""

    mask = time_values(data.index, level) <= checkpoint
    return data.loc[np.asarray(mask, dtype=bool)].copy(deep=True)


def point_slice(value: PandasObject, checkpoint: Hashable, level: TimeLevel = 0) -> PandasObject:
    """Select rows exactly at one time coordinate."""

    mask = time_values(value.index, level) == checkpoint
    return value.loc[np.asarray(mask, dtype=bool)]


def to_frame(value: PandasObject) -> pd.DataFrame:
    """Normalize Series and DataFrame outputs to DataFrame."""

    if isinstance(value, pd.Series):
        return value.to_frame(value.name if value.name is not None else "value")
    return value


def stable_checkpoint_repr(value: object) -> str:
    """Stable string representation for timestamps and scalar checkpoints."""

    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def scalar_jsonable(value: object) -> object:
    """Convert common pandas/numpy scalars to JSON-friendly Python values."""

    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def validate_temporal_order(
    data: pd.DataFrame,
    level: TimeLevel = 0,
) -> None:
    """Require a non-missing, non-decreasing temporal coordinate.

    TimeSafe never sorts user input silently because rolling transforms may depend
    on row order.  A temporal audit on unsorted observations would test a different
    object from the one the researcher intended.
    """

    values = time_values(data.index, level)
    if values.hasnans:
        raise ValueError("The temporal index level must not contain missing values.")
    if not values.is_monotonic_increasing:
        raise ValueError(
            "The temporal index level must be sorted in non-decreasing order; "
            "TimeSafe does not reorder research data implicitly."
        )
