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


def test_index_mismatch_is_counted():
    left = pd.Series([1.0], index=["a"])
    right = pd.Series([1.0], index=["b"])
    result = compare_labeled_outputs(left, right)
    assert result.missing_left > 0
    assert result.missing_right > 0
