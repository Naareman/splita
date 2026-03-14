from __future__ import annotations

import warnings
from typing import ClassVar

import numpy as np
import pytest

from splita._validation import (
    check_array_like,
    check_in_range,
    check_is_integer,
    check_not_empty,
    check_one_of,
    check_positive,
    check_probabilities_sum_to_one,
    check_same_length,
    format_error,
)

# ─── format_error ──────────────────────────────────────────────────


class TestFormatError:
    def test_problem_only(self):
        msg = format_error("something went wrong.")
        assert msg == "something went wrong."

    def test_problem_and_detail(self):
        msg = format_error("problem.", detail="the detail.")
        assert "problem." in msg
        assert "  Detail: the detail." in msg
        assert "Hint" not in msg

    def test_problem_and_hint(self):
        msg = format_error("problem.", hint="try this.")
        assert "  Hint: try this." in msg
        assert "Detail" not in msg

    def test_all_three(self):
        msg = format_error("problem.", detail="detail.", hint="hint.")
        lines = msg.split("\n")
        assert len(lines) == 3
        assert lines[0] == "problem."
        assert lines[1] == "  Detail: detail."
        assert lines[2] == "  Hint: hint."


# ─── check_in_range ────────────────────────────────────────────────


class TestCheckInRange:
    def test_valid_value_passes(self):
        check_in_range(0.5, "alpha", 0, 1)

    def test_too_low_raises(self):
        with pytest.raises(ValueError, match=r"`alpha` must be in \(0, 1\)"):
            check_in_range(-0.1, "alpha", 0, 1)

    def test_too_high_raises(self):
        with pytest.raises(ValueError, match=r"above the maximum"):
            check_in_range(1.5, "alpha", 0, 1)

    def test_exclusive_boundary_low(self):
        """With exclusive low, value == low should raise."""
        with pytest.raises(ValueError):
            check_in_range(0, "alpha", 0, 1, low_inclusive=False)

    def test_inclusive_boundary_low(self):
        """With inclusive low, value == low should pass."""
        check_in_range(0, "alpha", 0, 1, low_inclusive=True)

    def test_exclusive_boundary_high(self):
        """With exclusive high, value == high should raise."""
        with pytest.raises(ValueError):
            check_in_range(1, "alpha", 0, 1, high_inclusive=False)

    def test_inclusive_boundary_high(self):
        """With inclusive high, value == high should pass."""
        check_in_range(1, "alpha", 0, 1, high_inclusive=True)

    def test_mixed_brackets_shown(self):
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            check_in_range(-1, "x", 0, 1, low_inclusive=True, high_inclusive=False)

    def test_custom_hint_appears(self):
        with pytest.raises(ValueError, match=r"typical values"):
            check_in_range(1.5, "alpha", 0, 1, hint="typical values are 0.05.")

    def test_check_in_range_nan_raises(self):
        with pytest.raises(ValueError, match=r"must be a finite number"):
            check_in_range(float("nan"), "alpha", 0, 1)

    def test_check_in_range_inf_raises(self):
        with pytest.raises(ValueError, match=r"must be a finite number"):
            check_in_range(float("inf"), "alpha", 0, 1)


# ─── check_positive ────────────────────────────────────────────────


class TestCheckPositive:
    def test_positive_passes(self):
        check_positive(5.0, "n")

    def test_negative_raises(self):
        with pytest.raises(ValueError, match=r"must be > 0"):
            check_positive(-1, "n")

    def test_zero_without_allow_zero_raises(self):
        with pytest.raises(ValueError, match=r"must be > 0"):
            check_positive(0, "n")

    def test_zero_with_allow_zero_passes(self):
        check_positive(0, "n", allow_zero=True)

    def test_negative_with_allow_zero_raises(self):
        with pytest.raises(ValueError, match=r"must be >= 0"):
            check_positive(-1, "n", allow_zero=True)

    def test_check_positive_nan_raises(self):
        with pytest.raises(ValueError, match=r"must be a finite number"):
            check_positive(float("nan"), "n")

    def test_check_positive_inf_raises(self):
        with pytest.raises(ValueError, match=r"must be a finite number"):
            check_positive(float("inf"), "n")


# ─── check_is_integer ──────────────────────────────────────────────


class TestCheckIsInteger:
    def test_int_passes(self):
        check_is_integer(5, "n")

    def test_int_like_float_passes(self):
        check_is_integer(5.0, "n")

    def test_non_integer_float_raises(self):
        with pytest.raises(ValueError, match=r"must be an integer"):
            check_is_integer(5.5, "n")

    def test_min_value_passes(self):
        check_is_integer(10, "n", min_value=5)

    def test_min_value_raises(self):
        with pytest.raises(ValueError, match=r"must be >= 5"):
            check_is_integer(3, "n", min_value=5)

    def test_non_numeric_raises_type_error(self):
        with pytest.raises(TypeError, match=r"must be an integer"):
            check_is_integer("five", "n")  # type: ignore[arg-type]

    def test_nan_raises(self):
        with pytest.raises(ValueError, match=r"finite integer"):
            check_is_integer(float("nan"), "n")

    def test_inf_raises(self):
        with pytest.raises(ValueError, match=r"finite integer"):
            check_is_integer(float("inf"), "n")


# ─── check_array_like ──────────────────────────────────────────────


class TestCheckArrayLike:
    def test_list_converts(self):
        arr = check_array_like([1, 2, 3], "data")
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_tuple_converts(self):
        arr = check_array_like((4, 5, 6), "data")
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, [4.0, 5.0, 6.0])

    def test_ndarray_passes_through(self):
        original = np.array([1.0, 2.0])
        arr = check_array_like(original, "data")
        np.testing.assert_array_equal(arr, original)

    def test_nan_warning_and_removal(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            arr = check_array_like([1, 2, float("nan"), 4], "control")
            assert len(w) == 1
            assert issubclass(w[0].category, RuntimeWarning)
            assert "1 NaN value" in str(w[0].message)
            assert "out of 4" in str(w[0].message)
        np.testing.assert_array_equal(arr, [1.0, 2.0, 4.0])

    def test_multiple_nans_plural(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            arr = check_array_like([float("nan"), 2, float("nan")], "x")
            assert "2 NaN values" in str(w[0].message)
        np.testing.assert_array_equal(arr, [2.0])

    def test_allow_nan_true_keeps_nans(self):
        arr = check_array_like([1, float("nan"), 3], "data", allow_nan=True)
        assert len(arr) == 3
        assert np.isnan(arr[1])

    def test_min_length_enforcement(self):
        with pytest.raises(ValueError, match=r"at least 5 elements"):
            check_array_like([1, 2, 3], "data", min_length=5)

    def test_min_length_after_nan_removal(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match=r"at least 3 elements"):
                check_array_like([1, float("nan"), float("nan")], "data", min_length=3)

    def test_bad_input_raises_type_error(self):
        with pytest.raises(TypeError, match=r"must be array-like"):
            check_array_like(42, "data")  # type: ignore[arg-type]

    def test_string_input_raises_type_error(self):
        with pytest.raises(TypeError, match=r"must be array-like"):
            check_array_like("hello", "data")  # type: ignore[arg-type]

    def test_list_with_unconvertible_objects_raises_type_error(self):
        """A list containing objects that can't be cast to float raises TypeError."""
        with pytest.raises(TypeError, match=r"can't be converted to a numeric array"):
            check_array_like([object(), object()], "data")

    def test_pandas_series_converts(self):
        pd = pytest.importorskip("pandas")
        series = pd.Series([10, 20, 30])
        arr = check_array_like(series, "data")
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, [10.0, 20.0, 30.0])


# ─── check_one_of ──────────────────────────────────────────────────


class TestCheckOneOf:
    OPTIONS: ClassVar[list[str]] = ["auto", "ttest", "ztest", "mannwhitney"]

    def test_valid_option_passes(self):
        check_one_of("ttest", "method", self.OPTIONS)

    def test_invalid_raises_with_options(self):
        with pytest.raises(ValueError, match=r"must be one of"):
            check_one_of("bayesian", "method", self.OPTIONS)

    def test_auto_suggestion_substring(self):
        """'mann' is a substring of 'mannwhitney', should suggest it."""
        with pytest.raises(ValueError, match=r"did you mean 'mannwhitney'"):
            check_one_of("mann", "method", self.OPTIONS)

    def test_auto_suggestion_superstring(self):
        """'ttesting' contains 'ttest', should suggest it."""
        with pytest.raises(ValueError, match=r"did you mean 'ttest'"):
            check_one_of("ttesting", "method", self.OPTIONS)

    def test_no_suggestion_for_unrelated(self):
        with pytest.raises(ValueError) as exc_info:
            check_one_of("xyz", "method", self.OPTIONS)
        assert "did you mean" not in str(exc_info.value)

    def test_explicit_hint_overrides(self):
        with pytest.raises(ValueError, match=r"use 'auto' for best results"):
            check_one_of(
                "bad", "method", self.OPTIONS, hint="use 'auto' for best results."
            )


# ─── check_same_length ─────────────────────────────────────────────


class TestCheckSameLength:
    def test_same_length_passes(self):
        check_same_length(np.array([1, 2, 3]), np.array([4, 5, 6]), "a", "b")

    def test_different_length_raises(self):
        with pytest.raises(ValueError, match=r"must have the same length") as exc_info:
            check_same_length(
                np.array([1, 2, 3]), np.array([1, 2]), "control", "treatment"
            )
        msg = str(exc_info.value)
        assert "control has 3 elements" in msg
        assert "treatment has 2 elements" in msg


# ─── check_not_empty ───────────────────────────────────────────────


class TestCheckNotEmpty:
    def test_non_empty_passes(self):
        check_not_empty([1, 2], "data")
        check_not_empty(np.array([1]), "data")

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match=r"can't be empty"):
            check_not_empty([], "data")

    def test_empty_array_raises(self):
        with pytest.raises(ValueError, match=r"can't be empty"):
            check_not_empty(np.array([]), "data")


# ─── check_probabilities_sum_to_one ────────────────────────────────


class TestCheckProbabilitiesSumToOne:
    def test_valid_sum_passes(self):
        check_probabilities_sum_to_one([0.3, 0.7], "weights")

    def test_valid_with_rounding(self):
        check_probabilities_sum_to_one([1 / 3, 1 / 3, 1 / 3], "weights")

    def test_invalid_sum_raises(self):
        with pytest.raises(ValueError, match=r"must sum to 1.0") as exc_info:
            check_probabilities_sum_to_one([0.5, 0.6], "weights")
        assert "1.1" in str(exc_info.value)

    def test_custom_tolerance(self):
        # Should pass with a loose tolerance
        check_probabilities_sum_to_one([0.99], "weights", tol=0.02)
        # Should fail with a tight tolerance
        with pytest.raises(ValueError):
            check_probabilities_sum_to_one([0.99], "weights", tol=1e-8)
