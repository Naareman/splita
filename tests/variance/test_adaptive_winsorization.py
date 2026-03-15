"""Tests for AdaptiveWinsorizer."""

from __future__ import annotations

import numpy as np
import pytest

from splita.variance.adaptive_winsorization import AdaptiveWinsorizer


# ── helpers ──────────────────────────────────────────────────────────


def _make_heavy_tailed_data(
    n: int = 500,
    outlier_frac: float = 0.05,
    effect: float = 0.5,
    seed: int = 42,
):
    """Generate data with heavy tails (outliers in both groups)."""
    rng = np.random.default_rng(seed)
    n_outliers = int(n * outlier_frac)
    n_normal = n - n_outliers

    ctrl = np.concatenate([
        rng.normal(10, 2, n_normal),
        rng.normal(100, 50, n_outliers),
    ])
    trt = np.concatenate([
        rng.normal(10 + effect, 2, n_normal),
        rng.normal(100, 50, n_outliers),
    ])
    return ctrl, trt


# ── Basic behaviour ──────────────────────────────────────────────────


class TestBasic:
    """Basic AdaptiveWinsorizer behaviour."""

    def test_fit_transform_returns_arrays(self):
        """fit_transform returns two numpy arrays."""
        ctrl, trt = _make_heavy_tailed_data()
        winz = AdaptiveWinsorizer()
        ctrl_w, trt_w = winz.fit_transform(ctrl, trt)
        assert isinstance(ctrl_w, np.ndarray)
        assert isinstance(trt_w, np.ndarray)
        assert len(ctrl_w) == len(ctrl)
        assert len(trt_w) == len(trt)

    def test_variance_reduction_positive(self):
        """Heavy-tailed data yields positive variance reduction."""
        ctrl, trt = _make_heavy_tailed_data()
        winz = AdaptiveWinsorizer()
        winz.fit_transform(ctrl, trt)
        assert winz.variance_reduction_ > 0.0

    def test_optimal_thresholds_stored(self):
        """Optimal lower/upper percentiles are stored."""
        ctrl, trt = _make_heavy_tailed_data()
        winz = AdaptiveWinsorizer()
        winz.fit_transform(ctrl, trt)
        assert 0.0 < winz.optimal_lower_ <= 0.10
        assert 0.90 <= winz.optimal_upper_ < 1.0

    def test_winsorized_values_clipped(self):
        """Winsorized values fall within the threshold range."""
        ctrl, trt = _make_heavy_tailed_data()
        winz = AdaptiveWinsorizer()
        ctrl_w, trt_w = winz.fit_transform(ctrl, trt)
        # All values should be within the thresholds
        assert np.all(ctrl_w >= winz._lower_threshold - 1e-10)
        assert np.all(ctrl_w <= winz._upper_threshold + 1e-10)
        assert np.all(trt_w >= winz._lower_threshold - 1e-10)
        assert np.all(trt_w <= winz._upper_threshold + 1e-10)


# ── Comparison with fixed thresholds ─────────────────────────────────


class TestVsFixed:
    """Adaptive should find better thresholds than fixed."""

    def test_better_than_fixed_1_99(self):
        """Adaptive finds lower variance than fixed 1%/99%."""
        ctrl, trt = _make_heavy_tailed_data(outlier_frac=0.10, seed=123)
        n_c, n_t = len(ctrl), len(trt)

        # Fixed 1%/99% winsorization
        combined = np.concatenate([ctrl, trt])
        lo = np.percentile(combined, 1)
        hi = np.percentile(combined, 99)
        ctrl_fixed = np.clip(ctrl, lo, hi)
        trt_fixed = np.clip(trt, lo, hi)
        var_fixed = np.var(ctrl_fixed, ddof=1) / n_c + np.var(trt_fixed, ddof=1) / n_t

        # Adaptive
        winz = AdaptiveWinsorizer()
        ctrl_w, trt_w = winz.fit_transform(ctrl, trt)
        var_adaptive = np.var(ctrl_w, ddof=1) / n_c + np.var(trt_w, ddof=1) / n_t

        assert var_adaptive <= var_fixed + 1e-10

    def test_heavy_tailed_benefits_more(self):
        """Heavier tails yield greater absolute variance reduction.

        With more outliers, the absolute variance of the original data
        is much higher and winsorization removes more of it.
        """
        rng = np.random.default_rng(42)
        n = 500

        # Light: normal data, small outliers
        ctrl_light = rng.normal(10, 2, n)
        trt_light = rng.normal(10.5, 2, n)
        ctrl_light[:5] = 50  # mild outliers

        # Heavy: same base but extreme outliers
        ctrl_heavy = rng.normal(10, 2, n)
        trt_heavy = rng.normal(10.5, 2, n)
        ctrl_heavy[:50] = rng.normal(500, 200, 50)  # extreme outliers
        trt_heavy[:50] = rng.normal(500, 200, 50)

        winz_light = AdaptiveWinsorizer()
        winz_light.fit_transform(ctrl_light, trt_light)

        winz_heavy = AdaptiveWinsorizer()
        winz_heavy.fit_transform(ctrl_heavy, trt_heavy)

        # The heavy-tailed data has more room for improvement
        assert winz_heavy.variance_reduction_ > winz_light.variance_reduction_


# ── No outliers scenario ─────────────────────────────────────────────


class TestNoOutliers:
    """When data is clean, minimal capping should occur."""

    def test_clean_data_minimal_capping(self):
        """Clean normal data results in modest variance reduction.

        Even with normally distributed data, capping the tails reduces
        variance somewhat.  The reduction should be much less than for
        heavy-tailed data.
        """
        rng = np.random.default_rng(42)
        ctrl = rng.normal(10, 1, 500)
        trt = rng.normal(10.5, 1, 500)

        winz = AdaptiveWinsorizer()
        ctrl_w, trt_w = winz.fit_transform(ctrl, trt)

        # With clean data, variance reduction should be moderate
        # (some reduction from trimming normal tails, but < heavy-tailed)
        assert winz.variance_reduction_ < 0.50

    def test_ate_approximately_preserved(self):
        """ATE is approximately preserved after winsorization."""
        rng = np.random.default_rng(42)
        ctrl = rng.normal(10, 2, 500)
        trt = rng.normal(10.5, 2, 500)

        original_ate = np.mean(trt) - np.mean(ctrl)

        winz = AdaptiveWinsorizer()
        ctrl_w, trt_w = winz.fit_transform(ctrl, trt)
        adjusted_ate = np.mean(trt_w) - np.mean(ctrl_w)

        np.testing.assert_allclose(adjusted_ate, original_ate, atol=0.3)


# ── Grid configuration ──────────────────────────────────────────────


class TestGridConfig:
    """Tests for grid configuration parameters."""

    def test_custom_grid_size(self):
        """Custom n_grid works."""
        ctrl, trt = _make_heavy_tailed_data()
        winz = AdaptiveWinsorizer(n_grid=10)
        winz.fit_transform(ctrl, trt)
        assert winz.variance_reduction_ >= 0.0

    def test_custom_ranges(self):
        """Custom lower/upper ranges work."""
        ctrl, trt = _make_heavy_tailed_data()
        winz = AdaptiveWinsorizer(
            lower_range=(0.01, 0.05),
            upper_range=(0.95, 0.99),
        )
        winz.fit_transform(ctrl, trt)
        assert 0.01 <= winz.optimal_lower_ <= 0.05
        assert 0.95 <= winz.optimal_upper_ <= 0.99


# ── Fit / transform lifecycle ────────────────────────────────────────


class TestLifecycle:
    """Tests for fit/transform/fit_transform lifecycle."""

    def test_transform_before_fit_raises(self):
        """Calling transform before fit raises RuntimeError."""
        winz = AdaptiveWinsorizer()
        with pytest.raises(RuntimeError, match="must be fitted"):
            winz.transform([1, 2, 3], [4, 5, 6])

    def test_fit_returns_self(self):
        """fit() returns the instance for chaining."""
        ctrl, trt = _make_heavy_tailed_data()
        winz = AdaptiveWinsorizer()
        result = winz.fit(ctrl, trt)
        assert result is winz

    def test_separate_fit_transform(self):
        """Separate fit/transform gives same result as fit_transform."""
        ctrl, trt = _make_heavy_tailed_data()

        winz1 = AdaptiveWinsorizer()
        ctrl_w1, trt_w1 = winz1.fit_transform(ctrl, trt)

        winz2 = AdaptiveWinsorizer()
        winz2.fit(ctrl, trt)
        ctrl_w2, trt_w2 = winz2.transform(ctrl, trt)

        np.testing.assert_array_equal(ctrl_w1, ctrl_w2)
        np.testing.assert_array_equal(trt_w1, trt_w2)


# ── Validation ───────────────────────────────────────────────────────


class TestValidation:
    """Input validation tests."""

    def test_n_grid_too_small(self):
        """n_grid < 2 raises ValueError."""
        with pytest.raises(ValueError, match="n_grid"):
            AdaptiveWinsorizer(n_grid=1)

    def test_invalid_lower_range(self):
        """Invalid lower_range raises ValueError."""
        with pytest.raises(ValueError, match="lower_range"):
            AdaptiveWinsorizer(lower_range=(0.6, 0.8))

    def test_invalid_upper_range(self):
        """Invalid upper_range raises ValueError."""
        with pytest.raises(ValueError, match="upper_range"):
            AdaptiveWinsorizer(upper_range=(0.3, 0.4))

    def test_lower_range_inverted(self):
        """lower_range with a >= b raises ValueError."""
        with pytest.raises(ValueError, match="lower_range"):
            AdaptiveWinsorizer(lower_range=(0.10, 0.01))

    def test_upper_range_inverted(self):
        """upper_range with a >= b raises ValueError."""
        with pytest.raises(ValueError, match="upper_range"):
            AdaptiveWinsorizer(upper_range=(0.99, 0.90))

    def test_control_too_short(self):
        """Control array with < 2 elements raises ValueError."""
        winz = AdaptiveWinsorizer()
        with pytest.raises(ValueError, match="at least"):
            winz.fit([1.0], [1.0, 2.0])

    def test_zero_variance_data(self):
        """Constant data (zero variance) yields variance_reduction_ = 0."""
        ctrl = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        trt = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        winz = AdaptiveWinsorizer()
        winz.fit(ctrl, trt)
        assert winz.variance_reduction_ == 0.0

    def test_lower_threshold_above_upper_skipped(self):
        """Grid points where lower threshold >= upper threshold are skipped.

        This happens when lower_range percentiles produce values above
        the upper_range percentiles (extreme data distribution).
        """
        # Data where many values are the same, making thresholds overlap
        ctrl = np.array([1.0] * 50 + [100.0] * 50)
        trt = np.array([1.0] * 50 + [100.0] * 50)
        # Use ranges that are close enough that some grid points overlap
        winz = AdaptiveWinsorizer(
            n_grid=5,
            lower_range=(0.001, 0.10),
            upper_range=(0.90, 0.999),
        )
        winz.fit(ctrl, trt)
        # Should still produce valid results
        assert winz._is_fitted
