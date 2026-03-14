from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from splita._utils import (
    auto_detect_metric,
    cohens_d,
    cohens_h,
    ensure_rng,
    pooled_proportion,
    relative_lift,
    to_array,
)

# ─── ensure_rng ───────────────────────────────────────────────────


class TestEnsureRng:
    def test_none_returns_generator(self):
        rng = ensure_rng(None)
        assert isinstance(rng, np.random.Generator)

    def test_int_returns_deterministic_generator(self):
        rng1 = ensure_rng(42)
        rng2 = ensure_rng(42)
        assert rng1.random() == rng2.random()

    def test_generator_passes_through(self):
        orig = np.random.default_rng(99)
        result = ensure_rng(orig)
        assert result is orig

    def test_invalid_type_raises_typeerror(self):
        with pytest.raises(TypeError, match="random_state"):
            ensure_rng("bad")  # type: ignore[arg-type]


# ─── to_array ─────────────────────────────────────────────────────


class TestToArray:
    def test_list_to_1d_ndarray(self):
        arr = to_array([1, 2, 3], "x")
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 1
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_tuple_to_1d_ndarray(self):
        arr = to_array((4, 5, 6), "x")
        assert arr.ndim == 1
        np.testing.assert_array_equal(arr, [4.0, 5.0, 6.0])

    def test_ndarray_passthrough_as_float64(self):
        orig = np.array([1, 2, 3], dtype="int32")
        arr = to_array(orig, "x")
        assert arr.dtype == np.float64

    def test_2d_array_raises(self):
        with pytest.raises(ValueError, match="must be a 1-D array"):
            to_array(np.array([[1, 2], [3, 4]]), "x")

    def test_pandas_series(self):
        pd = pytest.importorskip("pandas")
        s = pd.Series([10, 20, 30])
        arr = to_array(s, "x")
        assert arr.ndim == 1
        np.testing.assert_array_equal(arr, [10.0, 20.0, 30.0])

    def test_non_numeric_raises_typeerror(self):
        with pytest.raises(TypeError, match="can't be converted"):
            to_array(["a", "b", "c"], "control")

    def test_to_array_scalar(self):
        arr = to_array(5.0, "x")
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 1
        assert len(arr) == 1
        assert arr[0] == 5.0


# ─── cohens_d ─────────────────────────────────────────────────────


class TestCohensD:
    def test_identical_arrays_zero(self):
        a = np.array([5.0, 5.0, 5.0, 5.0])
        assert cohens_d(a, a) == 0.0

    def test_known_large_d(self):
        control = np.array([0.0, 0.0, 0.0, 0.0])
        treatment = np.array([1.0, 1.0, 1.0, 1.0])
        # pooled_std = 0, all identical within each group
        # Both groups have std=0, so pooled_std=0, returns 0.0
        # Use groups with variance instead:
        control = np.array([0.0, 0.0, 1.0, 1.0])
        treatment = np.array([1.0, 1.0, 2.0, 2.0])
        d = cohens_d(control, treatment)
        assert d > 0
        # Manual: means 0.5 vs 1.5, both std = sqrt(1/3) ≈ 0.577
        # pooled_std = sqrt(((3*1/3)+(3*1/3))/6) = sqrt(1/3)
        # d = 1.0 / sqrt(1/3) ≈ 1.732
        assert abs(d - math.sqrt(3)) < 1e-10

    def test_symmetric_sign_flip(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        assert abs(cohens_d(a, b) + cohens_d(b, a)) < 1e-12

    def test_cohens_d_single_element_warning(self):
        """Single-element arrays with different values: pooled_std=0, warns."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            d = cohens_d(np.array([1.0]), np.array([2.0]))
            assert len(w) == 1
            assert issubclass(w[0].category, RuntimeWarning)
            assert "zero" in str(w[0].message).lower()
        assert d == float("inf")

    def test_cohens_d_zero_std_different_means(self):
        """Multiple identical values per group but different means."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            d = cohens_d(np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0]))
            assert len(w) == 1
        assert d == float("inf")

        # Reverse direction gives -inf
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            d2 = cohens_d(np.array([2.0, 2.0, 2.0]), np.array([1.0, 1.0, 1.0]))
        assert d2 == float("-inf")


# ─── cohens_h ─────────────────────────────────────────────────────


class TestCohensH:
    def test_same_proportions_zero(self):
        assert cohens_h(0.5, 0.5) == 0.0

    def test_known_value(self):
        expected = 2.0 * (math.asin(math.sqrt(0.7)) - math.asin(math.sqrt(0.5)))
        assert abs(cohens_h(0.5, 0.7) - expected) < 1e-12

    def test_antisymmetric(self):
        h_fwd = cohens_h(0.3, 0.6)
        h_rev = cohens_h(0.6, 0.3)
        assert abs(h_fwd + h_rev) < 1e-12

    def test_cohens_h_boundary_zero(self):
        h = cohens_h(0.0, 0.5)
        assert isinstance(h, float)
        assert h > 0

    def test_cohens_h_boundary_one(self):
        h = cohens_h(0.5, 1.0)
        assert isinstance(h, float)
        assert h > 0

    def test_cohens_h_negative_raises(self):
        with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
            cohens_h(-0.1, 0.5)

    def test_cohens_h_above_one_raises(self):
        with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
            cohens_h(0.5, 1.1)


# ─── relative_lift ────────────────────────────────────────────────


class TestRelativeLift:
    def test_basic_lift(self):
        assert relative_lift(100.0, 120.0) == pytest.approx(0.20)

    def test_zero_control_positive_treatment(self):
        assert relative_lift(0.0, 5.0) == float("inf")

    def test_zero_control_negative_treatment(self):
        assert relative_lift(0.0, -3.0) == float("-inf")

    def test_zero_both(self):
        assert relative_lift(0.0, 0.0) == 0.0

    def test_relative_lift_negative_control(self):
        # negative control mean: lift = (treatment - control) / |control|
        result = relative_lift(-100.0, -80.0)
        assert result == pytest.approx(0.20)


# ─── pooled_proportion ───────────────────────────────────────────


class TestPooledProportion:
    def test_all_ones(self):
        assert pooled_proportion(np.array([1, 1, 1]), np.array([1, 1])) == 1.0

    def test_all_zeros(self):
        assert pooled_proportion(np.array([0, 0]), np.array([0, 0, 0])) == 0.0

    def test_mixed(self):
        x1 = np.array([1, 0, 1])
        x2 = np.array([0, 0, 1])
        assert pooled_proportion(x1, x2) == pytest.approx(0.5)

    def test_pooled_proportion_empty_raises(self):
        """Empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="must not both be empty"):
            pooled_proportion(np.array([]), np.array([]))


# ─── auto_detect_metric ──────────────────────────────────────────


class TestAutoDetectMetric:
    def test_binary_is_conversion(self):
        assert auto_detect_metric(np.array([0, 1, 1, 0, 1])) == "conversion"

    def test_continuous(self):
        assert auto_detect_metric(np.array([1.5, 2.3, 0.7])) == "continuous"

    def test_all_zeros_is_conversion(self):
        assert auto_detect_metric(np.array([0, 0, 0])) == "conversion"

    def test_value_two_is_continuous(self):
        assert auto_detect_metric(np.array([0, 1, 2])) == "continuous"

    def test_auto_detect_metric_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            auto_detect_metric(np.array([]))

    def test_auto_detect_metric_single_element(self):
        assert auto_detect_metric(np.array([1])) == "conversion"
        assert auto_detect_metric(np.array([0])) == "conversion"
        assert auto_detect_metric(np.array([0.5])) == "continuous"
