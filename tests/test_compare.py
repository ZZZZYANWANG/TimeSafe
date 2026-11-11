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


def test_nullable_string_and_category_comparison():
    index = pd.Index(["a", "b", "c"])
    left = pd.DataFrame(
        {
            "s": pd.Series(["x", pd.NA, "z"], index=index, dtype="string"),
            "c": pd.Series(pd.Categorical(["u", "v", None]), index=index),
        }
    )
    right = left.copy()
    result = compare_labeled_outputs(left, right)
    assert result.changed_cells == 0
    assert result.compared_cells == 6


def test_nullable_string_difference_is_detected():
    index = pd.Index(["a", "b"])
    left = pd.Series(["x", pd.NA], index=index, dtype="string")
    right = pd.Series(["y", pd.NA], index=index, dtype="string")
    result = compare_labeled_outputs(left, right)
    assert result.changed_cells == 1


def test_categorical_columns_with_different_category_sets_are_compared_safely():
    left = pd.Series(
        pd.Categorical(["a", "b"], categories=["a", "b"]),
        index=[0, 1],
    )
    right = pd.Series(
        pd.Categorical(["a", "c"], categories=["a", "c"]),
        index=[0, 1],
    )
    result = compare_labeled_outputs(left, right)
    assert result.compared_cells == 2
    assert result.changed_cells == 1


def test_equal_nested_object_cells_with_strings_and_missing_values():
    index = pd.Index(["row"])
    cases = [
        (["a", None], ["a", None]),
        (("a", np.nan), ("a", np.nan)),
        ({"key": ["a", pd.NA]}, {"key": ["a", pd.NA]}),
        (np.asarray(["a", "b"], dtype=object), np.asarray(["a", "b"], dtype=object)),
        ({float("nan"), "a"}, {float("nan"), "a"}),
        (frozenset({float("nan"), "a"}), frozenset({float("nan"), "a"})),
    ]
    for left_value, right_value in cases:
        left = pd.Series([left_value], index=index, dtype=object)
        right = pd.Series([right_value], index=index, dtype=object)
        stats = compare_labeled_outputs(left, right)
        assert stats.compared_cells == 1
        assert stats.changed_cells == 0


def test_nested_object_cells_detect_value_and_container_type_changes():
    index = pd.Index(["row"])
    cases = [
        (["a", None], ["b", None]),
        ({"key": [1, 2]}, {"key": [1, 3]}),
        (["a"], ("a",)),
        (np.asarray(["a"], dtype=object), ["a"]),
    ]
    for left_value, right_value in cases:
        left = pd.Series([left_value], index=index, dtype=object)
        right = pd.Series([right_value], index=index, dtype=object)
        stats = compare_labeled_outputs(left, right)
        assert stats.changed_cells == 1
