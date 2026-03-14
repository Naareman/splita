"""Tests for OutlierHandler."""

from __future__ import annotations

import numpy as np
import pytest

from splita.variance import OutlierHandler

# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


@pytest.fixture()
def normal_data(rng):
    """Control and treatment with a few extreme outliers."""
    ctrl = rng.normal(10, 2, size=500)
    trt = rng.normal(10.5, 2, size=500)
    # inject outliers
    ctrl[0] = 100.0
    ctrl[1] = -50.0
    trt[0] = 200.0
    trt[1] = -80.0
    return ctrl, trt


# ═══════════════════════════════════════════════════════════════════════
# Basic tests
# ═══════════════════════════════════════════════════════════════════════


class TestWinsorize:
    """Test 1: Winsorize caps values, extreme values capped, non-extreme unchanged."""

    def test_caps_extreme_values(self, normal_data):
        ctrl, trt = normal_data
        handler = OutlierHandler(method="winsorize")
        ctrl_c, trt_c = handler.fit_transform(ctrl, trt)

        # The 100 and 200 outliers should be capped down
        assert ctrl_c[0] < 100.0
        assert trt_c[0] < 200.0

    def test_preserves_normal_values(self, rng):
        ctrl = rng.normal(10, 1, size=500)
        trt = rng.normal(10, 1, size=500)
        handler = OutlierHandler(method="winsorize")
        ctrl_c, _ = handler.fit_transform(ctrl, trt)

        # Non-extreme values in the middle should be unchanged
        mid = ctrl[250]
        assert ctrl_c[250] == mid


class TestTrim:
    """Test 2: Trim removes values, arrays shrink."""

    def test_removes_extreme_values(self, normal_data):
        ctrl, trt = normal_data
        handler = OutlierHandler(method="trim")
        ctrl_t, trt_t = handler.fit_transform(ctrl, trt)

        # Trimmed arrays should be shorter (outliers removed)
        assert len(ctrl_t) < len(ctrl)
        assert len(trt_t) < len(trt)

        # 100, -50 should not be in result
        assert 100.0 not in ctrl_t
        assert -50.0 not in ctrl_t


class TestIQR:
    """Test 3: IQR method caps values outside Q1-1.5*IQR, Q3+1.5*IQR."""

    def test_caps_outliers(self, normal_data):
        ctrl, trt = normal_data
        handler = OutlierHandler(method="iqr")
        ctrl_c, trt_c = handler.fit_transform(ctrl, trt)

        assert ctrl_c[0] < 100.0
        assert trt_c[0] < 200.0


class TestDefault:
    """Test 4: Default behavior is winsorize at 1st/99th percentile."""

    def test_default_params(self):
        handler = OutlierHandler()
        assert handler.method == "winsorize"
        assert handler.lower == 0.01
        assert handler.upper == 0.99


# ═══════════════════════════════════════════════════════════════════════
# Statistical correctness
# ═══════════════════════════════════════════════════════════════════════


class TestPooledThresholds:
    """Test 5: Thresholds computed on combined data, not per-group."""

    def test_thresholds_from_pooled(self):
        ctrl = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        trt = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
        handler = OutlierHandler(method="winsorize", lower=0.1, upper=0.9)
        handler.fit(ctrl, trt)

        # Thresholds should be from combined [1..10], not from each group
        combined = np.concatenate([ctrl, trt])
        expected_lower = float(np.percentile(combined, 10))
        expected_upper = float(np.percentile(combined, 90))
        assert handler.lower_threshold_ == pytest.approx(expected_lower)
        assert handler.upper_threshold_ == pytest.approx(expected_upper)


class TestWinsorizePreservesLength:
    """Test 6: Winsorize preserves array length."""

    def test_same_length(self, normal_data):
        ctrl, trt = normal_data
        handler = OutlierHandler(method="winsorize")
        ctrl_c, trt_c = handler.fit_transform(ctrl, trt)

        assert len(ctrl_c) == len(ctrl)
        assert len(trt_c) == len(trt)


class TestTrimReducesLength:
    """Test 7: Trim can reduce array length."""

    def test_shorter_arrays(self, normal_data):
        ctrl, trt = normal_data
        handler = OutlierHandler(method="trim")
        ctrl_t, trt_t = handler.fit_transform(ctrl, trt)

        assert len(ctrl_t) <= len(ctrl)
        assert len(trt_t) <= len(trt)


class TestIQRThresholds:
    """Test 8: IQR thresholds match manual calculation."""

    def test_iqr_manual(self):
        ctrl = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        trt = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
        handler = OutlierHandler(method="iqr", iqr_multiplier=1.5)
        handler.fit(ctrl, trt)

        combined = np.concatenate([ctrl, trt])
        q1 = float(np.percentile(combined, 25))
        q3 = float(np.percentile(combined, 75))
        iqr = q3 - q1

        assert handler.lower_threshold_ == pytest.approx(q1 - 1.5 * iqr)
        assert handler.upper_threshold_ == pytest.approx(q3 + 1.5 * iqr)


# ═══════════════════════════════════════════════════════════════════════
# Side parameter
# ═══════════════════════════════════════════════════════════════════════


class TestSideParameter:
    """Tests 9-12: side parameter controls which tails are capped."""

    def test_side_upper(self, normal_data):
        """Test 9: side='upper' only caps upper tail."""
        ctrl, trt = normal_data
        handler = OutlierHandler(side="upper")
        ctrl_c, _ = handler.fit_transform(ctrl, trt)

        # Lower outlier -50 should remain untouched
        assert ctrl_c[1] == -50.0
        # Upper outlier 100 should be capped
        assert ctrl_c[0] < 100.0

    def test_side_lower(self, normal_data):
        """Test 10: side='lower' only caps lower tail."""
        ctrl, trt = normal_data
        handler = OutlierHandler(side="lower")
        ctrl_c, _ = handler.fit_transform(ctrl, trt)

        # Upper outlier 100 should remain untouched
        assert ctrl_c[0] == 100.0
        # Lower outlier -50 should be capped
        assert ctrl_c[1] > -50.0

    def test_side_both(self, normal_data):
        """Test 11: side='both' caps both tails."""
        ctrl, trt = normal_data
        handler = OutlierHandler(side="both")
        ctrl_c, _ = handler.fit_transform(ctrl, trt)

        assert ctrl_c[0] < 100.0
        assert ctrl_c[1] > -50.0

    def test_side_none_with_custom_bounds(self):
        """Test 12: side=None uses provided lower/upper."""
        handler = OutlierHandler(lower=0.05, upper=0.95)
        assert handler.lower == 0.05
        assert handler.upper == 0.95


# ═══════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Tests 13-17: edge cases."""

    def test_no_outliers(self):
        """Test 13: No outliers — arrays unchanged."""
        ctrl = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
        trt = np.array([5.5, 6.5, 7.5, 8.5, 9.5])
        handler = OutlierHandler(method="winsorize", lower=0.0, upper=1.0)
        ctrl_c, trt_c = handler.fit_transform(ctrl, trt)

        np.testing.assert_array_equal(ctrl_c, ctrl)
        np.testing.assert_array_equal(trt_c, trt)
        assert handler.n_capped_ == 0

    def test_all_outliers(self):
        """Test 14: All values outside bounds."""
        ctrl = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        trt = np.array([96.0, 97.0, 98.0, 99.0, 100.0])
        # With aggressive percentiles, many values will be capped
        handler = OutlierHandler(method="winsorize", lower=0.25, upper=0.75)
        _ctrl_c, _trt_c = handler.fit_transform(ctrl, trt)

        # At least some values should have been capped
        assert handler.n_capped_ > 0

    def test_lower_none(self):
        """Test 15: lower=None means no lower capping."""
        ctrl = np.array([1.0, 2.0, 3.0, 100.0, 5.0])
        trt = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        handler = OutlierHandler(lower=None, upper=0.99)
        handler.fit(ctrl, trt)

        assert handler.lower_threshold_ is None
        assert handler.upper_threshold_ is not None

    def test_upper_none(self):
        """Test 16: upper=None means no upper capping."""
        ctrl = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        trt = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        handler = OutlierHandler(lower=0.01, upper=None)
        handler.fit(ctrl, trt)

        assert handler.lower_threshold_ is not None
        assert handler.upper_threshold_ is None

    def test_iqr_multiplier_3(self):
        """Test 17: iqr_multiplier=3.0 is less aggressive."""
        ctrl = np.arange(1.0, 101.0)
        trt = np.arange(1.0, 101.0)

        handler_15 = OutlierHandler(method="iqr", iqr_multiplier=1.5)
        handler_30 = OutlierHandler(method="iqr", iqr_multiplier=3.0)
        handler_15.fit(ctrl, trt)
        handler_30.fit(ctrl, trt)

        # 3.0 multiplier should produce wider bounds
        assert handler_30.lower_threshold_ < handler_15.lower_threshold_
        assert handler_30.upper_threshold_ > handler_15.upper_threshold_


# ═══════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════


class TestValidation:
    """Tests 18-22: input validation."""

    def test_invalid_method(self):
        """Test 18: Invalid method raises ValueError with suggestion."""
        with pytest.raises(ValueError, match=r"must be one of"):
            OutlierHandler(method="bogus")

    def test_clustering_deferred(self):
        """Test 19: method='clustering' raises ValueError with v0.2.0 hint."""
        with pytest.raises(ValueError, match=r"v0.2.0"):
            OutlierHandler(method="clustering")

    def test_lower_gte_upper(self):
        """Test 20: invalid lower/upper combos raise ValueError."""
        # upper=0.4 is out of range (must be > 0.5)
        with pytest.raises(ValueError, match=r"must be in"):
            OutlierHandler(lower=0.4, upper=0.4)

        # lower=0.6 is out of range (must be < 0.5)
        with pytest.raises(ValueError, match=r"must be in"):
            OutlierHandler(lower=0.6, upper=0.9)

        # Valid combo should not raise
        OutlierHandler(lower=0.01, upper=0.99)

    def test_transform_before_fit(self):
        """Test 21: transform before fit raises RuntimeError."""
        handler = OutlierHandler()
        with pytest.raises(RuntimeError, match=r"must be fitted"):
            handler.transform([1, 2, 3], [4, 5, 6])

    def test_iqr_multiplier_lte_zero(self):
        """Test 22: iqr_multiplier <= 0 raises ValueError."""
        with pytest.raises(ValueError, match=r"must be > 0"):
            OutlierHandler(iqr_multiplier=0)

        with pytest.raises(ValueError, match=r"must be > 0"):
            OutlierHandler(iqr_multiplier=-1.0)

    def test_lower_out_of_range(self):
        with pytest.raises(ValueError, match=r"\[0, 0\.5\)"):
            OutlierHandler(lower=0.6)

    def test_upper_out_of_range(self):
        with pytest.raises(ValueError, match=r"\(0\.5, 1\]"):
            OutlierHandler(upper=0.3)


# ═══════════════════════════════════════════════════════════════════════
# Integration
# ═══════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Test 23: OutlierHandler -> CUPED pipeline."""

    def test_outlier_then_cuped(self, rng):
        """Full variance reduction pipeline: outliers -> CUPED."""
        from splita.variance import CUPED

        pre = rng.normal(10, 2, size=200)
        ctrl = pre[:100] + rng.normal(0, 1, 100)
        trt = pre[100:] + 0.5 + rng.normal(0, 1, 100)

        # Inject outliers
        ctrl[0] = 500.0
        trt[0] = -300.0

        # Step 1: outlier handling
        handler = OutlierHandler(method="winsorize")
        ctrl_c, trt_c = handler.fit_transform(ctrl, trt)

        # Step 2: CUPED
        cuped = CUPED()
        _ctrl_adj, _trt_adj = cuped.fit_transform(ctrl_c, trt_c, pre[:100], pre[100:])

        # Outliers should be capped
        assert ctrl_c[0] < 500.0
        assert trt_c[0] > -300.0
        # CUPED should reduce variance
        assert cuped.variance_reduction_ > 0.0

    def test_binary_data_note(self):
        """Test 24: Binary data — outlier handling is a no-op (documented)."""
        ctrl = np.array([0, 1, 0, 0, 1, 0, 1, 0, 1, 0], dtype=float)
        trt = np.array([1, 1, 0, 1, 1, 0, 1, 0, 1, 1], dtype=float)

        handler = OutlierHandler(method="winsorize")
        ctrl_c, trt_c = handler.fit_transform(ctrl, trt)

        # For binary data winsorize is effectively a no-op:
        # percentile 1 >= 0 and percentile 99 <= 1
        np.testing.assert_array_equal(ctrl_c, ctrl)
        np.testing.assert_array_equal(trt_c, trt)


# ═══════════════════════════════════════════════════════════════════════
# sklearn API
# ═══════════════════════════════════════════════════════════════════════


class TestSklearnAPI:
    """Tests 25-26: sklearn-style API."""

    def test_fit_returns_self(self, normal_data):
        """Test 25: fit() returns self."""
        ctrl, trt = normal_data
        handler = OutlierHandler()
        result = handler.fit(ctrl, trt)
        assert result is handler

    def test_fit_transform_matches(self, normal_data):
        """Test 26: fit_transform returns same as fit().transform()."""
        ctrl, trt = normal_data

        handler1 = OutlierHandler(method="winsorize", lower=0.05, upper=0.95)
        ctrl_ft, trt_ft = handler1.fit_transform(ctrl, trt)

        handler2 = OutlierHandler(method="winsorize", lower=0.05, upper=0.95)
        handler2.fit(ctrl, trt)
        ctrl_t, trt_t = handler2.transform(ctrl, trt)

        np.testing.assert_array_equal(ctrl_ft, ctrl_t)
        np.testing.assert_array_equal(trt_ft, trt_t)


class TestNTotal:
    """Verify n_total_ is set correctly."""

    def test_n_total(self):
        ctrl = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        trt = np.array([6.0, 7.0, 8.0])
        handler = OutlierHandler()
        handler.fit(ctrl, trt)
        assert handler.n_total_ == 8


class TestInvalidSide:
    """Cover line 148: invalid side value."""

    def test_invalid_side_raises(self):
        with pytest.raises(ValueError, match=r"side"):
            OutlierHandler(side="left")
