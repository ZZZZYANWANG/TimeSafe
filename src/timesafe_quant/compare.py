"""Labeled output comparison utilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._pandas import PandasObject, ensure_pandas_output, to_frame


@dataclass(frozen=True)
class ComparisonStats:
    """Cell-level comparison statistics.

    ``changed_cells`` counts value differences only where both outputs contain the
    same labeled cell. Structural additions/removals are reported separately through
    ``missing_left`` and ``missing_right`` and are never double-counted.
    """

    compared_cells: int
    changed_cells: int
    missing_left: int
    missing_right: int
    max_abs_diff: float | None
    max_rel_diff: float | None

    @property
    def changed_fraction(self) -> float | None:
        if self.compared_cells == 0:
            return None
        return self.changed_cells / self.compared_cells

    @property
    def structural_differences(self) -> int:
        return self.missing_left + self.missing_right


def _is_missing_scalar(value: object) -> bool:
    """Return a boolean missingness flag without coercing array-like objects.

    ``pandas.isna`` returns an array for list- or ndarray-valued object cells.
    Treat those cells as ordinary values and let the equality branch decide, rather
    than raising while attempting ``bool(array)``.
    """

    if not pd.api.types.is_scalar(value):
        return False
    result = pd.isna(value)
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    return False


def _object_values_equal(
    left: object,
    right: object,
    *,
    _seen: set[tuple[int, int]] | None = None,
) -> bool:
    """Compare nested object cells with missing-aware, non-ambiguous semantics.

    ``numpy.array_equal(..., equal_nan=True)`` raises on arrays of strings and
    arbitrary objects.  TimeSafe therefore handles common containers recursively
    and reserves NumPy's numeric fast path for genuine numeric arrays.
    """

    if left is right:
        return True

    left_na = _is_missing_scalar(left)
    right_na = _is_missing_scalar(right)
    if left_na or right_na:
        return left_na and right_na

    if _seen is None:
        _seen = set()
    pair = (id(left), id(right))
    if pair in _seen:
        return True
    _seen.add(pair)

    if isinstance(left, (pd.Series, pd.DataFrame, pd.Index)) or isinstance(
        right, (pd.Series, pd.DataFrame, pd.Index)
    ):
        return type(left) is type(right) and bool(left.equals(right))

    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if left.keys() != right.keys():
            return False
        return all(
            _object_values_equal(left[key], right[key], _seen=_seen)
            for key in left
        )

    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            return False
        if left.shape != right.shape:
            return False
        if pd.api.types.is_numeric_dtype(left.dtype) and pd.api.types.is_numeric_dtype(
            right.dtype
        ):
            return bool(np.array_equal(left, right, equal_nan=True))
        return all(
            _object_values_equal(left_value, right_value, _seen=_seen)
            for left_value, right_value in zip(left.flat, right.flat)
        )

    sequence_types = (list, tuple)
    if isinstance(left, sequence_types) or isinstance(right, sequence_types):
        if type(left) is not type(right) or len(left) != len(right):
            return False
        return all(
            _object_values_equal(left_value, right_value, _seen=_seen)
            for left_value, right_value in zip(left, right)
        )

    if isinstance(left, (set, frozenset)) or isinstance(right, (set, frozenset)):
        if type(left) is not type(right) or len(left) != len(right):
            return False
        unmatched = list(right)
        for left_value in left:
            match_index = next(
                (
                    index
                    for index, right_value in enumerate(unmatched)
                    if _object_values_equal(
                        left_value,
                        right_value,
                        _seen=set(_seen),
                    )
                ),
                None,
            )
            if match_index is None:
                return False
            unmatched.pop(match_index)
        return True

    try:
        comparison = left == right
    except (TypeError, ValueError):
        return False
    if comparison is pd.NA:
        return False
    if isinstance(comparison, (bool, np.bool_)):
        return bool(comparison)
    try:
        comparison_array = np.asarray(comparison)
    except (TypeError, ValueError):
        return False
    return bool(comparison_array.dtype == bool and comparison_array.all())


def compare_labeled_outputs(
    left: PandasObject,
    right: PandasObject,
    *,
    atol: float = 1e-10,
    rtol: float = 1e-8,
) -> ComparisonStats:
    """Compare labeled outputs with strict label alignment and NaN-aware semantics."""

    if atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be non-negative.")
    left_frame = to_frame(ensure_pandas_output(left, name="left"))
    right_frame = to_frame(ensure_pandas_output(right, name="right"))

    union_index = left_frame.index.union(right_frame.index, sort=False)
    union_columns = left_frame.columns.union(right_frame.columns, sort=False)
    left_aligned = left_frame.reindex(index=union_index, columns=union_columns)
    right_aligned = right_frame.reindex(index=union_index, columns=union_columns)

    left_rows = np.asarray(union_index.isin(left_frame.index), dtype=bool)[:, None]
    right_rows = np.asarray(union_index.isin(right_frame.index), dtype=bool)[:, None]
    left_columns = np.asarray(union_columns.isin(left_frame.columns), dtype=bool)[None, :]
    right_columns = np.asarray(union_columns.isin(right_frame.columns), dtype=bool)[None, :]

    left_present = left_rows & left_columns
    right_present = right_rows & right_columns
    both_present = left_present & right_present

    missing_left = int((~left_present & right_present).sum())
    missing_right = int((left_present & ~right_present).sum())
    compared_cells = int(both_present.sum())
    changed_cells = 0
    numeric_abs: list[np.ndarray] = []
    numeric_rel: list[np.ndarray] = []

    for column_position, column in enumerate(union_columns):
        row_mask = both_present[:, column_position]
        if not row_mask.any():
            continue

        left_col = left_aligned[column]
        right_col = right_aligned[column]
        left_original = left_frame[column]
        right_original = right_frame[column]

        if pd.api.types.is_numeric_dtype(left_original) and pd.api.types.is_numeric_dtype(
            right_original
        ):
            complex_values = pd.api.types.is_complex_dtype(
                left_original.dtype
            ) or pd.api.types.is_complex_dtype(right_original.dtype)
            target_dtype = np.complex128 if complex_values else np.float64
            missing_value = np.nan + 0j if complex_values else np.nan
            left_values = left_col.to_numpy(
                dtype=target_dtype, na_value=missing_value
            )[row_mask]
            right_values = right_col.to_numpy(
                dtype=target_dtype, na_value=missing_value
            )[row_mask]
            equal = np.isclose(
                left_values,
                right_values,
                atol=atol,
                rtol=rtol,
                equal_nan=True,
            )
            changed_cells += int((~equal).sum())

            both_finite = np.isfinite(left_values) & np.isfinite(right_values)
            if both_finite.any():
                abs_diff = np.abs(left_values[both_finite] - right_values[both_finite])
                scale_floor = max(float(atol), float(np.finfo(np.float64).tiny))
                denom = np.maximum(
                    np.maximum(
                        np.abs(left_values[both_finite]),
                        np.abs(right_values[both_finite]),
                    ),
                    scale_floor,
                )
                numeric_abs.append(abs_diff)
                numeric_rel.append(abs_diff / denom)
        else:
            left_values = left_col.iloc[np.flatnonzero(row_mask)].astype(object).to_numpy()
            right_values = right_col.iloc[np.flatnonzero(row_mask)].astype(object).to_numpy()
            equal_values: list[bool] = []
            for left_value, right_value in zip(left_values, right_values):
                left_na = _is_missing_scalar(left_value)
                right_na = _is_missing_scalar(right_value)
                if left_na or right_na:
                    equal_values.append(left_na and right_na)
                    continue
                equal_values.append(_object_values_equal(left_value, right_value))
            changed_cells += int((~np.asarray(equal_values, dtype=bool)).sum())

    max_abs = None
    max_rel = None
    if numeric_abs:
        max_abs = float(max(float(values.max()) for values in numeric_abs if values.size))
    if numeric_rel:
        max_rel = float(max(float(values.max()) for values in numeric_rel if values.size))

    return ComparisonStats(
        compared_cells=compared_cells,
        changed_cells=changed_cells,
        missing_left=missing_left,
        missing_right=missing_right,
        max_abs_diff=max_abs,
        max_rel_diff=max_rel,
    )
