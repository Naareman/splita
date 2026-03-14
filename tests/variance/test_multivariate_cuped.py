"""Tests for MultivariateCUPED."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from splita.variance.cuped import CUPED
from splita.variance.multivariate_cuped import MultivariateCUPED


# ── helpers ──────────────────────────────────────────────────────────


def _make_multivariate_data(
    n: int = 500,
    effect: float = 0.5,
    n_covariates: int = 2,
    seed: int = 42,
):
    """Generate data with multiple correlated covariates."""
    rng = np.random.default_rng(seed)
    X_all = rng.normal(10, 2, size=(2 * n, n_covariates))

    # Outcome = weighted sum of covariates + effect + noise
    weights = np.linspace(1.0, 0.5, n_covariates)
    base = X_all @ weights

    ctrl = base[:n] + rng.normal(0, 1, n)
    trt = base[n:] + effect + rng.normal(0, 1, n)

    X_ctrl = X_all[:n]
    X_trt = X_all[n:]

    return ctrl, trt, X_ctrl, X_trt


# ── Basic behaviour ──────────────────────────────────────────────────


class TestBasic:
    """Basic MultivariateCUPED behaviour."""

    def test_variance_reduction_works(self):
        """Adjusted data has lower variance than original."""
        ctrl, trt, X_c, X_t = _make_multivariate_data()
        mcuped = MultivariateCUPED()
        ctrl_adj, trt_adj = mcuped.fit_transform(ctrl, trt, X_c, X_t)

        original_var = np.var(np.concatenate([ctrl, trt]))
        adjusted_var = np.var(np.concatenate([ctrl_adj, trt_adj]))
        assert adjusted_var < original_var

    def test_ate_preserved(self):
        """Mean difference is approximately preserved after adjustment.

        ATE is exactly preserved when control and treatment have the same
        covariate means.  With random data the means differ slightly,
        so we use identical covariate distributions seeded to have
        equal means and allow a generous tolerance.
        """
        rng = np.random.default_rng(42)
        n = 500
        # Same covariate distribution for both groups
        X_all = rng.normal(10, 2, size=(2 * n, 2))
        weights = np.array([1.0, 0.5])
        base = X_all @ weights
        ctrl = base[:n] + rng.normal(0, 1, n)
        trt = base[n:] + 0.5 + rng.normal(0, 1, n)
        X_c, X_t = X_all[:n], X_all[n:]

        mcuped = MultivariateCUPED()
        ctrl_adj, trt_adj = mcuped.fit_transform(ctrl, trt, X_c, X_t)

        original_ate = np.mean(trt) - np.mean(ctrl)
        adjusted_ate = np.mean(trt_adj) - np.mean(ctrl_adj)
        np.testing.assert_allclose(adjusted_ate, original_ate, atol=0.25)

    def test_theta_vector_length(self):
        """theta_ has one element per covariate."""
        ctrl, trt, X_c, X_t = _make_multivariate_data(n_covariates=3)
        mcuped = MultivariateCUPED()
        mcuped.fit(ctrl, trt, X_c, X_t)
        assert len(mcuped.theta_) == 3

    def test_variance_reduction_stored(self):
        """variance_reduction_ is a number in [0, 1]."""
        ctrl, trt, X_c, X_t = _make_multivariate_data()
        mcuped = MultivariateCUPED()
        mcuped.fit(ctrl, trt, X_c, X_t)
        assert 0.0 <= mcuped.variance_reduction_ <= 1.0

    def test_correlation_stored(self):
        """correlation_ (multiple R) is a number in [0, 1]."""
        ctrl, trt, X_c, X_t = _make_multivariate_data()
        mcuped = MultivariateCUPED()
        mcuped.fit(ctrl, trt, X_c, X_t)
        assert 0.0 <= mcuped.correlation_ <= 1.0


# ── Single covariate matches scalar CUPED ────────────────────────────


class TestSingleCovariate:
    """Single covariate should match scalar CUPED."""

    def test_single_covariate_theta_matches(self):
        """With 1 covariate, theta matches scalar CUPED theta."""
        rng = np.random.default_rng(42)
        n = 500
        pre = rng.normal(10, 2, size=2 * n)
        ctrl = pre[:n] + rng.normal(0, 1, n)
        trt = pre[n:] + 0.5 + rng.normal(0, 1, n)
        pre_ctrl, pre_trt = pre[:n], pre[n:]

        # Scalar CUPED
        cuped = CUPED()
        cuped.fit(ctrl, trt, pre_ctrl, pre_trt)

        # Multivariate CUPED with 1 covariate
        mcuped = MultivariateCUPED()
        mcuped.fit(ctrl, trt, pre_ctrl.reshape(-1, 1), pre_trt.reshape(-1, 1))

        np.testing.assert_allclose(mcuped.theta_[0], cuped.theta_, rtol=0.01)

    def test_single_covariate_adjustment_matches(self):
        """With 1 covariate, adjusted values match scalar CUPED."""
        rng = np.random.default_rng(42)
        n = 500
        pre = rng.normal(10, 2, size=2 * n)
        ctrl = pre[:n] + rng.normal(0, 1, n)
        trt = pre[n:] + 0.5 + rng.normal(0, 1, n)
        pre_ctrl, pre_trt = pre[:n], pre[n:]

        cuped = CUPED()
        ctrl_adj_s, trt_adj_s = cuped.fit_transform(ctrl, trt, pre_ctrl, pre_trt)

        mcuped = MultivariateCUPED()
        ctrl_adj_m, trt_adj_m = mcuped.fit_transform(
            ctrl, trt, pre_ctrl.reshape(-1, 1), pre_trt.reshape(-1, 1)
        )

        np.testing.assert_allclose(ctrl_adj_m, ctrl_adj_s, atol=0.01)
        np.testing.assert_allclose(trt_adj_m, trt_adj_s, atol=0.01)


# ── Multiple covariates better than single ───────────────────────────


class TestMultipleCovariates:
    """Multiple covariates should outperform single covariate."""

    def test_two_covariates_better_than_one(self):
        """Two correlated covariates reduce more variance than one."""
        rng = np.random.default_rng(42)
        n = 500
        X1 = rng.normal(10, 2, size=2 * n)
        X2 = rng.normal(5, 1, size=2 * n)
        noise = rng.normal(0, 1, 2 * n)
        y = X1 + 0.5 * X2 + noise
        ctrl, trt = y[:n], y[n:] + 0.3

        # Single covariate (X1 only)
        cuped = CUPED()
        cuped.fit(ctrl, trt, X1[:n], X1[n:])
        vr_single = cuped.variance_reduction_

        # Two covariates
        X_ctrl = np.column_stack([X1[:n], X2[:n]])
        X_trt = np.column_stack([X1[n:], X2[n:]])
        mcuped = MultivariateCUPED()
        mcuped.fit(ctrl, trt, X_ctrl, X_trt)
        vr_multi = mcuped.variance_reduction_

        assert vr_multi > vr_single

    def test_three_covariates(self):
        """Three covariates work correctly."""
        ctrl, trt, X_c, X_t = _make_multivariate_data(n_covariates=3)
        mcuped = MultivariateCUPED()
        mcuped.fit_transform(ctrl, trt, X_c, X_t)
        assert len(mcuped.theta_) == 3
        assert mcuped.variance_reduction_ > 0.0


# ── Fit / transform lifecycle ────────────────────────────────────────


class TestLifecycle:
    """Tests for fit/transform/fit_transform lifecycle."""

    def test_transform_before_fit_raises(self):
        """Calling transform before fit raises RuntimeError."""
        mcuped = MultivariateCUPED()
        with pytest.raises(RuntimeError, match="must be fitted"):
            mcuped.transform([1, 2, 3], [4, 5, 6], [[1], [2], [3]], [[4], [5], [6]])

    def test_fit_returns_self(self):
        """fit() returns the instance for chaining."""
        ctrl, trt, X_c, X_t = _make_multivariate_data()
        mcuped = MultivariateCUPED()
        result = mcuped.fit(ctrl, trt, X_c, X_t)
        assert result is mcuped


# ── Validation ───────────────────────────────────────────────────────


class TestValidation:
    """Input validation tests."""

    def test_alpha_out_of_range(self):
        """alpha outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            MultivariateCUPED(alpha=1.5)

    def test_mismatched_covariate_columns(self):
        """Different number of covariates in control/treatment raises ValueError."""
        ctrl = np.array([1.0, 2.0, 3.0])
        trt = np.array([4.0, 5.0, 6.0])
        X_c = np.array([[1, 2], [3, 4], [5, 6]])
        X_t = np.array([[1], [2], [3]])
        mcuped = MultivariateCUPED()
        with pytest.raises(ValueError, match="same number of covariates"):
            mcuped.fit(ctrl, trt, X_c, X_t)

    def test_mismatched_covariate_rows(self):
        """Covariate rows not matching data length raises ValueError."""
        ctrl = np.array([1.0, 2.0, 3.0])
        trt = np.array([4.0, 5.0, 6.0])
        X_c = np.array([[1], [2]])  # 2 rows, but control has 3
        X_t = np.array([[1], [2], [3]])
        mcuped = MultivariateCUPED()
        with pytest.raises(ValueError, match="3 rows"):
            mcuped.fit(ctrl, trt, X_c, X_t)

    def test_covariate_not_array_like(self):
        """Non-array covariate raises TypeError."""
        ctrl = np.array([1.0, 2.0, 3.0])
        trt = np.array([4.0, 5.0, 6.0])
        mcuped = MultivariateCUPED()
        with pytest.raises(TypeError, match="array-like"):
            mcuped.fit(ctrl, trt, 42, np.array([[1], [2], [3]]))

    def test_singular_covariates(self):
        """Perfectly collinear covariates raise ValueError."""
        rng = np.random.default_rng(42)
        n = 50
        ctrl = rng.normal(10, 1, n)
        trt = rng.normal(10.5, 1, n)
        # X2 = 2 * X1 (perfectly collinear)
        X1_c = rng.normal(0, 1, n)
        X1_t = rng.normal(0, 1, n)
        X_c = np.column_stack([X1_c, 2 * X1_c])
        X_t = np.column_stack([X1_t, 2 * X1_t])
        mcuped = MultivariateCUPED()
        with pytest.raises(ValueError, match="singular"):
            mcuped.fit(ctrl, trt, X_c, X_t)

    def test_1d_covariate_auto_promoted(self):
        """1-D covariate array is automatically promoted to 2-D."""
        rng = np.random.default_rng(42)
        n = 100
        pre = rng.normal(10, 2, 2 * n)
        ctrl = pre[:n] + rng.normal(0, 1, n)
        trt = pre[n:] + 0.5 + rng.normal(0, 1, n)

        mcuped = MultivariateCUPED()
        ctrl_adj, trt_adj = mcuped.fit_transform(ctrl, trt, pre[:n], pre[n:])
        assert len(mcuped.theta_) == 1
