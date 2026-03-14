"""Tests for CUPED variance reduction."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from splita.variance import CUPED

# ── helpers ──────────────────────────────────────────────────────────


def _make_correlated_data(
    n: int = 500,
    effect: float = 0.5,
    noise: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate correlated pre/post data for both groups."""
    rng = np.random.default_rng(seed)
    pre_ctrl = rng.normal(10, 2, size=n)
    pre_trt = rng.normal(10, 2, size=n)
    ctrl = pre_ctrl + rng.normal(0, noise, n)
    trt = pre_trt + effect + rng.normal(0, noise, n)
    return ctrl, trt, pre_ctrl, pre_trt


# ── Basic tests ──────────────────────────────────────────────────────


class TestBasic:
    """Basic CUPED behaviour."""

    def test_variance_reduction_works(self):
        """Adjusted data has lower variance than original."""
        ctrl, trt, pre_ctrl, pre_trt = _make_correlated_data()
        cuped = CUPED()
        ctrl_adj, trt_adj = cuped.fit_transform(ctrl, trt, pre_ctrl, pre_trt)

        original_var = np.var(np.concatenate([ctrl, trt]))
        adjusted_var = np.var(np.concatenate([ctrl_adj, trt_adj]))
        assert adjusted_var < original_var

    def test_ate_preserved(self):
        """Mean difference is preserved after CUPED adjustment.

        ATE is exactly preserved when both groups share the same
        pre-experiment mean.  With random data the pre-means differ
        slightly, so we use identical pre-data distributions seeded
        to have equal means.
        """
        rng = np.random.default_rng(42)
        n = 500
        # Use the *same* pre-distribution so mean(X_ctrl) == mean(X_trt)
        pre = rng.normal(10, 2, size=2 * n)
        pre_ctrl, pre_trt = pre[:n], pre[n:]
        ctrl = pre_ctrl + rng.normal(0, 1, n)
        trt = pre_trt + 0.5 + rng.normal(0, 1, n)

        cuped = CUPED()
        ctrl_adj, trt_adj = cuped.fit_transform(ctrl, trt, pre_ctrl, pre_trt)

        original_ate = np.mean(trt) - np.mean(ctrl)
        adjusted_ate = np.mean(trt_adj) - np.mean(ctrl_adj)
        # ATE = original_ate - theta * (mean(X_trt) - mean(X_ctrl))
        # With large n the pre-means are close, so allow small tolerance
        np.testing.assert_allclose(adjusted_ate, original_ate, atol=0.15)

    def test_known_theta(self):
        """Manually set theta=1.0 — verify adjustment formula."""
        np.random.default_rng(0)
        ctrl = np.array([10.0, 12.0, 11.0, 13.0, 14.0])
        trt = np.array([11.0, 13.0, 12.0, 14.0, 15.0])
        pre_ctrl = np.array([9.0, 11.0, 10.0, 12.0, 13.0])
        pre_trt = np.array([10.0, 12.0, 11.0, 13.0, 14.0])

        cuped = CUPED(theta=1.0)
        ctrl_adj, trt_adj = cuped.fit_transform(ctrl, trt, pre_ctrl, pre_trt)

        x_pool_mean = np.mean(np.concatenate([pre_ctrl, pre_trt]))
        expected_ctrl = ctrl - 1.0 * (pre_ctrl - x_pool_mean)
        expected_trt = trt - 1.0 * (pre_trt - x_pool_mean)

        np.testing.assert_allclose(ctrl_adj, expected_ctrl)
        np.testing.assert_allclose(trt_adj, expected_trt)

    def test_high_correlation_large_variance_reduction(self):
        """Highly correlated pre/post data gives variance_reduction_ > 0.3."""
        ctrl, trt, pre_ctrl, pre_trt = _make_correlated_data(noise=0.5)
        cuped = CUPED()
        cuped.fit(ctrl, trt, pre_ctrl, pre_trt)
        assert cuped.variance_reduction_ > 0.3


# ── Statistical correctness ─────────────────────────────────────────


class TestStatistical:
    """Verify CUPED computes statistics correctly."""

    def test_theta_computation(self):
        """Theta matches Cov(Y,X)/Var(X) computed manually."""
        ctrl, trt, pre_ctrl, pre_trt = _make_correlated_data()
        cuped = CUPED()
        cuped.fit(ctrl, trt, pre_ctrl, pre_trt)

        y = np.concatenate([ctrl, trt])
        x = np.concatenate([pre_ctrl, pre_trt])
        expected_theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
        np.testing.assert_allclose(cuped.theta_, expected_theta, rtol=1e-10)

    def test_correlation_stored_correctly(self):
        """correlation_ matches np.corrcoef."""
        ctrl, trt, pre_ctrl, pre_trt = _make_correlated_data()
        cuped = CUPED()
        cuped.fit(ctrl, trt, pre_ctrl, pre_trt)

        y = np.concatenate([ctrl, trt])
        x = np.concatenate([pre_ctrl, pre_trt])
        expected_corr = np.corrcoef(y, x)[0, 1]
        np.testing.assert_allclose(cuped.correlation_, expected_corr, rtol=1e-10)

    def test_variance_reduction_is_r_squared(self):
        """variance_reduction_ equals correlation_^2."""
        ctrl, trt, pre_ctrl, pre_trt = _make_correlated_data()
        cuped = CUPED()
        cuped.fit(ctrl, trt, pre_ctrl, pre_trt)

        np.testing.assert_allclose(
            cuped.variance_reduction_,
            cuped.correlation_**2,
            rtol=1e-10,
        )


# ── Edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases and special inputs."""

    def test_zero_correlation(self):
        """Random (uncorrelated) pre-data → theta ≈ 0, minimal adjustment."""
        rng = np.random.default_rng(99)
        ctrl = rng.normal(10, 2, 500)
        trt = rng.normal(10.5, 2, 500)
        pre_ctrl = rng.normal(0, 1, 500)  # unrelated
        pre_trt = rng.normal(0, 1, 500)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cuped = CUPED()
            cuped.fit(ctrl, trt, pre_ctrl, pre_trt)

        assert abs(cuped.correlation_) < 0.15
        assert cuped.variance_reduction_ < 0.05

    def test_perfect_correlation(self):
        """pre = post → theta ≈ 1.0, variance_reduction ≈ 1.0."""
        rng = np.random.default_rng(7)
        pre_ctrl = rng.normal(10, 2, 100)
        pre_trt = rng.normal(10, 2, 100)
        ctrl = pre_ctrl.copy()
        trt = pre_trt + 0.5  # shift but same correlation

        cuped = CUPED()
        cuped.fit(ctrl, trt, pre_ctrl, pre_trt)

        np.testing.assert_allclose(cuped.theta_, 1.0, atol=0.05)
        assert cuped.variance_reduction_ > 0.95

    def test_custom_covariate_array(self):
        """Pass an explicit covariate array instead of 'auto'."""
        rng = np.random.default_rng(42)
        ctrl = rng.normal(10, 2, 50)
        trt = rng.normal(10.5, 2, 50)
        custom_cov = np.concatenate([ctrl, trt]) + rng.normal(0, 0.1, 100)

        cuped = CUPED(covariate=custom_cov)
        _ctrl_adj, _trt_adj = cuped.fit_transform(ctrl, trt)

        assert cuped._is_fitted
        assert cuped.variance_reduction_ > 0.5

    def test_manual_theta_used(self):
        """Pass theta=0.5, verify it's used instead of computed."""
        ctrl, trt, pre_ctrl, pre_trt = _make_correlated_data(n=100)
        cuped = CUPED(theta=0.5)
        cuped.fit(ctrl, trt, pre_ctrl, pre_trt)

        assert cuped.theta_ == 0.5


# ── Validation ───────────────────────────────────────────────────────


class TestValidation:
    """Input validation."""

    def test_missing_pre_data_auto(self):
        """covariate='auto' without pre-data raises ValueError."""
        ctrl = np.array([1.0, 2.0, 3.0])
        trt = np.array([2.0, 3.0, 4.0])
        cuped = CUPED()

        with pytest.raises(ValueError, match=r"control_pre.*treatment_pre.*required"):
            cuped.fit(ctrl, trt)

    def test_length_mismatch(self):
        """control and control_pre with different lengths → ValueError."""
        ctrl = np.array([1.0, 2.0, 3.0])
        trt = np.array([2.0, 3.0, 4.0])
        pre_ctrl = np.array([0.5, 1.5])  # wrong length
        pre_trt = np.array([1.5, 2.5, 3.5])
        cuped = CUPED()

        with pytest.raises(ValueError, match=r"same length"):
            cuped.fit(ctrl, trt, pre_ctrl, pre_trt)

    def test_transform_before_fit(self):
        """transform() before fit() raises RuntimeError."""
        ctrl = np.array([1.0, 2.0, 3.0])
        trt = np.array([2.0, 3.0, 4.0])
        pre_ctrl = np.array([0.5, 1.5, 2.5])
        pre_trt = np.array([1.5, 2.5, 3.5])
        cuped = CUPED()

        with pytest.raises(RuntimeError, match=r"fitted before calling transform"):
            cuped.transform(ctrl, trt, pre_ctrl, pre_trt)

    def test_min_length_control(self):
        """Arrays shorter than 2 elements raise ValueError."""
        ctrl = np.array([1.0])
        trt = np.array([2.0, 3.0])
        cuped = CUPED()

        with pytest.raises(ValueError, match=r"at least 2"):
            cuped.fit(ctrl, trt, ctrl, trt)

    def test_invalid_covariate_string(self):
        """Invalid string for covariate raises ValueError."""
        with pytest.raises(ValueError, match=r"covariate"):
            CUPED(covariate="invalid")


# ── Integration ──────────────────────────────────────────────────────


class TestIntegration:
    """Integration and pipeline tests."""

    def test_pipeline_cuped_then_experiment(self):
        """CUPED → Experiment pipeline works end-to-end."""
        from splita import Experiment

        ctrl, trt, pre_ctrl, pre_trt = _make_correlated_data(
            n=500,
            effect=0.5,
            seed=123,
        )
        cuped = CUPED()
        ctrl_adj, trt_adj = cuped.fit_transform(ctrl, trt, pre_ctrl, pre_trt)

        result = Experiment(ctrl_adj, trt_adj, metric="continuous").run()
        assert hasattr(result, "pvalue")
        assert hasattr(result, "significant")

    def test_low_correlation_warning(self):
        """Uncorrelated data emits RuntimeWarning about low correlation."""
        rng = np.random.default_rng(0)
        ctrl = rng.normal(10, 2, 200)
        trt = rng.normal(10.5, 2, 200)
        pre_ctrl = rng.normal(0, 1, 200)
        pre_trt = rng.normal(0, 1, 200)

        cuped = CUPED()
        with pytest.warns(RuntimeWarning, match=r"Low correlation"):
            cuped.fit(ctrl, trt, pre_ctrl, pre_trt)


# ── sklearn API compliance ───────────────────────────────────────────


class TestSklearnAPI:
    """Verify sklearn-style API conventions."""

    def test_fit_returns_self(self):
        """fit() returns self for method chaining."""
        ctrl, trt, pre_ctrl, pre_trt = _make_correlated_data(n=50)
        cuped = CUPED()
        result = cuped.fit(ctrl, trt, pre_ctrl, pre_trt)
        assert result is cuped

    def test_fit_transform_matches_separate_calls(self):
        """fit_transform() returns same result as fit() then transform()."""
        ctrl, trt, pre_ctrl, pre_trt = _make_correlated_data(n=100, seed=7)

        cuped1 = CUPED()
        ctrl_adj1, trt_adj1 = cuped1.fit_transform(ctrl, trt, pre_ctrl, pre_trt)

        cuped2 = CUPED()
        cuped2.fit(ctrl, trt, pre_ctrl, pre_trt)
        ctrl_adj2, trt_adj2 = cuped2.transform(ctrl, trt, pre_ctrl, pre_trt)

        np.testing.assert_allclose(ctrl_adj1, ctrl_adj2)
        np.testing.assert_allclose(trt_adj1, trt_adj2)


# ── Additional coverage tests ────────────────────────────────────────


class TestZeroVarianceCovariate:
    """Cover the zero-variance covariate error (line 151)."""

    def test_constant_covariate_raises(self):
        """Covariate with zero variance raises ValueError."""
        ctrl = np.array([10.0, 12.0, 11.0, 13.0, 14.0])
        trt = np.array([11.0, 13.0, 12.0, 14.0, 15.0])
        # All pre values identical -> zero variance
        pre_ctrl = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        pre_trt = np.array([5.0, 5.0, 5.0, 5.0, 5.0])

        cuped = CUPED()
        with pytest.raises(ValueError, match=r"zero variance"):
            cuped.fit(ctrl, trt, pre_ctrl, pre_trt)


class TestCustomCovariateLengthMismatch:
    """Cover the custom covariate length mismatch error (line 297)."""

    def test_wrong_length_covariate_array(self):
        """Custom covariate array with wrong length raises ValueError."""
        ctrl = np.array([10.0, 12.0, 11.0])
        trt = np.array([11.0, 13.0, 12.0])
        # Total is 6, but covariate has 4
        custom_cov = np.array([1.0, 2.0, 3.0, 4.0])

        cuped = CUPED(covariate=custom_cov)
        with pytest.raises(ValueError, match=r"covariate.*length"):
            cuped.fit(ctrl, trt)


class TestCustomModeMissingPre:
    """Cover covariate='custom' with missing pre data (line 323)."""

    def test_custom_mode_without_pre_raises(self):
        """covariate='custom' without pre data raises ValueError."""
        ctrl = np.array([10.0, 12.0, 11.0])
        trt = np.array([11.0, 13.0, 12.0])

        cuped = CUPED(covariate="custom")
        with pytest.raises(ValueError, match=r"control_pre.*treatment_pre.*required"):
            cuped.fit(ctrl, trt)
