"""Labeled output comparison utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._pandas import PandasObject, to_frame


@dataclass(frozen=True)
class ComparisonStats:
    """Cell-level comparison statistics."""

    compared_cells: int
    changed_cells: int
    missing_left: int
    missing_right: int
    max_abs_diff: float | None
    max_rel_diff: float | None

    @property
    def changed_fraction(self) -> float:
        if self.compared_cells == 0:
            return 0.0
        return self.changed_cells / self.compared_cells


def compare_labeled_outputs(
    left: PandasObject,
    right: PandasObject,
    *,
    atol: float = 1e-10,
    rtol: float = 1e-8,
) -> ComparisonStats:
    """Compare labeled outputs with index/column alignment and NaN-aware semantics."""

    left_frame = to_frame(left)
    right_frame = to_frame(right)
    left_aligned, right_aligned = left_frame.align(right_frame, join="outer", axis=None)

    left_missing_rows = left_aligned.index.difference(left_frame.index)
    right_missing_rows = right_aligned.index.difference(right_frame.index)
    left_missing_cols = left_aligned.columns.difference(left_frame.columns)
    right_missing_cols = right_aligned.columns.difference(right_frame.columns)

    missing_left = len(left_missing_rows) * max(1, len(left_aligned.columns)) + len(
        left_missing_cols
    ) * max(1, len(left_aligned.index))
    missing_right = len(right_missing_rows) * max(1, len(right_aligned.columns)) + len(
        right_missing_cols
    ) * max(1, len(right_aligned.index))

    changed = np.zeros(left_aligned.shape, dtype=bool)
    numeric_abs: list[np.ndarray] = []
    numeric_rel: list[np.ndarray] = []

    for column_position, column in enumerate(left_aligned.columns):
        left_col = left_aligned[column]
        right_col = right_aligned[column]

        if pd.api.types.is_numeric_dtype(left_col) and pd.api.types.is_numeric_dtype(right_col):
            left_values = left_col.to_numpy(dtype=float, na_value=np.nan)
            right_values = right_col.to_numpy(dtype=float, na_value=np.nan)
            equal = np.isclose(
                left_values,
                right_values,
                atol=atol,
                rtol=rtol,
                equal_nan=True,
            )
            changed[:, column_position] = ~equal

            both_finite = np.isfinite(left_values) & np.isfinite(right_values)
            if both_finite.any():
                abs_diff = np.abs(left_values[both_finite] - right_values[both_finite])
                denom = np.maximum(
                    np.maximum(np.abs(left_values[both_finite]), np.abs(right_values[both_finite])),
                    atol,
                )
                numeric_abs.append(abs_diff)
                numeric_rel.append(abs_diff / denom)
        else:
            equal = left_col.eq(right_col) | (left_col.isna() & right_col.isna())
            changed[:, column_position] = ~equal.to_numpy(dtype=bool)

    compared_cells = int(changed.size)
    changed_cells = int(changed.sum())
    max_abs = None
    max_rel = None
    if numeric_abs:
        max_abs = float(max(float(values.max()) for values in numeric_abs if values.size))
    if numeric_rel:
        max_rel = float(max(float(values.max()) for values in numeric_rel if values.size))

    return ComparisonStats(
        compared_cells=compared_cells,
        changed_cells=changed_cells,
        missing_left=int(missing_left),
        missing_right=int(missing_right),
        max_abs_diff=max_abs,
        max_rel_diff=max_rel,
    )
