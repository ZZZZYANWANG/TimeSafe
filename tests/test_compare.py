import numpy as np
import pandas as pd

from timesafe_quant.compare import compare_labeled_outputs


def test_numeric_comparison_is_nan_aware():
    index = pd.Index(["a", "b"])
    left = pd.Series([1.0, np.nan], index=index)
    right = pd.Series([1.0 + 1e-12, np.nan], index=index)
    result = compare_labeled_outputs(left, right, atol=1e-10, rtol=1e-8)
    assert result.changed_cells == 0
    assert result.compared_cells == 2
    assert result.changed_fraction == 0.0


def test_index_mismatch_is_structural_not_double_counted():
    left = pd.Series([1.0], index=["a"])
    right = pd.Series([1.0], index=["b"])
    result = compare_labeled_outputs(left, right)
    assert result.compared_cells == 0
    assert result.changed_cells == 0
    assert result.missing_left == 1
    assert result.missing_right == 1
    assert result.structural_differences == 2
    assert result.changed_fraction is None


def test_shared_and_missing_cells_are_separated():
    left = pd.DataFrame({"x": [1.0, 2.0]}, index=["a", "b"])
    right = pd.DataFrame({"x": [1.0, 3.0]}, index=["a", "c"])
    result = compare_labeled_outputs(left, right)
    assert result.compared_cells == 1
    assert result.changed_cells == 0
    assert result.missing_left == 1
    assert result.missing_right == 1
