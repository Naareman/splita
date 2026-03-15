"""Tests for Lin's Regression Adjustment variance reduction."""

from __future__ import annotations

import numpy as np
import pytest

from splita.variance import CUPED, RegressionAdjustment


# ── helpers ──────────────────────────────────────────────────────────


def _make_data(
    n: int = 500,
    effect: float = 0.5,
    noise: float = 1.0,
    seed: int = 42,
    p_covariates: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate correlated control/treatment data with covariates.

    Returns (ctrl, trt, X_ctrl, X_trt) where X has shape (n, p).
    """
    rng = np.random.default_rng(seed)
    X_ctrl = rng.normal(10, 2, size=(n, p_covariates))
    X_trt = rng.normal(10, 2, size=(n, p_covariates))

    # Outcome depends on first covariate
    ctrl = X_ctrl[:, 0] + rng.normal(0, noise, n)
    trt = X_trt[:, 0] + effect + rng.normal(0, noise, n)
    return ctrl, trt, X_ctrl, X_trt


def _make_1d_data(
    n: int = 500,
    effect: float = 0.5,
    noise: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate data with a single 1-D covariate."""
    ctrl, trt, Xc, Xt = _make_data(n=n, effect=effect, noise=noise, seed=seed)
    return ctrl, trt, Xc[:, 0], Xt[:, 0]


# ── Basic ATE tests ─────────────────────────────────────────────────


class TestBasicATE:
    """Basic average treatment effect estimation."""

    def test_positive_effect_detected(self):
        """Positive true effect yields positive ATE."""
        ctrl, trt, Xc, Xt = _make_1d_data(n=500, effect=1.0)
        ra = RegressionAdjustment()
        result = ra.fit_transform(ctrl, trt, Xc, Xt)
        assert result.ate > 0

    def test_zero_effect_not_significant(self):
        """No true effect → not significant at alpha=0.05 (most of the time)."""
        ctrl, trt, Xc, Xt = _make_1d_data(n=200, effect=0.0, seed=7)
        ra = RegressionAdjustment()
        result = ra.fit_transform(ctrl, trt, Xc, Xt)
        # With no effect, p-value should generally be > 0.05
        # Use a generous threshold to avoid flaky tests
        assert result.pvalue > 0.01

    def test_known_large_effect_recovery(self):
        """Large known effect (5.0) recovered within CI."""
        ctrl, trt, Xc, Xt = _make_1d_data(n=1000, effect=5.0, seed=99)
        ra = RegressionAdjustment()
        result = ra.fit_transform(ctrl, trt, Xc, Xt)
        assert result.ci_lower < 5.0 < result.ci_upper

    def test_ate_close_to_true_effect(self):
        """ATE estimate is within 0.5 of the true effect."""
        ctrl, trt, Xc, Xt = _make_1d_data(n=2000, effect=1.0, seed=42)
        ra = RegressionAdjustment()
        result = ra.fit_transform(ctrl, trt, Xc, Xt)
        assert abs(result.ate - 1.0) < 0.5


# ── Comparison with CUPED ────────────────────────────────────────────


class TestCUPEDComparison:
    """Compare regression adjustment with CUPED."""

    def test_matches_cuped_single_covariate(self):
        """With one covariate, regression adjustment SE <= CUPED SE."""
        rng = np.random.default_rng(42)
        n = 1000
        pre_ctrl = rng.normal(10, 2, n)
        pre_trt = rng.normal(10, 2, n)
        ctrl = pre_ctrl + rng.normal(0, 1, n)
        trt = pre_trt + 0.5 + rng.normal(0, 1, n)

        # CUPED
        cuped = CUPED()
        ctrl_adj, trt_adj = cuped.fit_transform(ctrl, trt, pre_ctrl, pre_trt)
        se_cuped = np.sqrt(
            np.var(ctrl_adj, ddof=1) / n + np.var(trt_adj, ddof=1) / n
        )

        # Regression adjustment
        ra = RegressionAdjustment()
        result = ra.fit_transform(ctrl, trt, pre_ctrl, pre_trt)

        # RA should be at least as efficient (SE <= CUPED SE, with tolerance)
        assert result.se <= se_cuped * 1.15  # allow 15% tolerance

    def test_better_than_cuped_multiple_covariates(self):
        """With multiple covariates, RA uses all of them (CUPED uses one)."""
        rng = np.random.default_rng(42)
        n = 1000
        p = 3
        X_ctrl = rng.normal(0, 1, (n, p))
        X_trt = rng.normal(0, 1, (n, p))

        # Outcome depends on ALL covariates
        beta_true = np.array([1.0, 0.8, 0.6])
        ctrl = X_ctrl @ beta_true + rng.normal(0, 0.5, n)
        trt = X_trt @ beta_true + 0.5 + rng.normal(0, 0.5, n)

        # CUPED with first covariate only
        cuped = CUPED()
        ctrl_adj, trt_adj = cuped.fit_transform(
            ctrl, trt, X_ctrl[:, 0], X_trt[:, 0]
        )
        se_cuped = np.sqrt(
            np.var(ctrl_adj, ddof=1) / n + np.var(trt_adj, ddof=1) / n
        )

        # RA with all covariates
        ra = RegressionAdjustment()
        result = ra.fit_transform(ctrl, trt, X_ctrl, X_trt)

        # RA should have strictly smaller SE
        assert result.se < se_cuped


# ── HC2 robust SE tests ─────────────────────────────────────────────


class TestHC2:
    """HC2 robust standard errors."""

    def test_se_positive(self):
        """SE is always positive."""
        ctrl, trt, Xc, Xt = _make_1d_data()
        ra = RegressionAdjustment()
        result = ra.fit_transform(ctrl, trt, Xc, Xt)
        assert result.se > 0

    def test_heteroskedastic_data(self):
        """HC2 handles heteroskedastic errors without inflating SE wildly."""
        rng = np.random.default_rng(42)
        n = 500
        X_ctrl = rng.normal(10, 2, n)
        X_trt = rng.normal(10, 2, n)

        # Variance proportional to |X| (heteroskedastic)
        ctrl = X_ctrl + rng.normal(0, 1, n) * np.abs(X_ctrl) * 0.1
        trt = X_trt + 2.0 + rng.normal(0, 1, n) * np.abs(X_trt) * 0.1

        ra = RegressionAdjustment()
        result = ra.fit_transform(ctrl, trt, X_ctrl, X_trt)

        # Should still detect the effect
        assert result.se > 0
        assert result.ate > 0

    def test_hc2_vs_homoskedastic_se(self):
        """HC2 SE and homoskedastic SE should be similar for homoskedastic data."""
        rng = np.random.default_rng(42)
        n = 1000
        X_ctrl = rng.normal(0, 1, n)
        X_trt = rng.normal(0, 1, n)
        # Perfectly homoskedastic
        ctrl = X_ctrl + rng.normal(0, 1, n)
        trt = X_trt + 0.5 + rng.normal(0, 1, n)

        ra = RegressionAdjustment()
        result = ra.fit_transform(ctrl, trt, X_ctrl, X_trt)

        # Compute homoskedastic SE for comparison
        Y = np.concatenate([ctrl, trt])
        T = np.concatenate([np.zeros(n), np.ones(n)])
        X = np.concatenate([X_ctrl, X_trt])
        X_c = X - X.mean()
        Z = np.column_stack([np.ones(2 * n), T, X_c, T * X_c])
        beta = np.linalg.lstsq(Z, Y, rcond=None)[0]
        resid = Y - Z @ beta
        sigma2 = np.sum(resid**2) / (2 * n - 4)
        ZtZ_inv = np.linalg.inv(Z.T @ Z)
        se_homo = np.sqrt(sigma2 * ZtZ_inv[1, 1])

        # HC2 and homoskedastic should be within 30% for homoskedastic data
        ratio = result.se / se_homo
        assert 0.7 < ratio < 1.3


# ── Result dataclass ─────────────────────────────────────────────────


class TestResult:
    """Result dataclass properties."""

    def test_result_fields(self):
        """Result has all expected fields."""
        ctrl, trt, Xc, Xt = _make_1d_data()
        ra = RegressionAdjustment()
        result = ra.fit_transform(ctrl, trt, Xc, Xt)

        assert hasattr(result, "ate")
        assert hasattr(result, "se")
        assert hasattr(result, "pvalue")
        assert hasattr(result, "ci_lower")
        assert hasattr(result, "ci_upper")
        assert hasattr(result, "significant")
        assert hasattr(result, "alpha")
        assert hasattr(result, "variance_reduction")
        assert hasattr(result, "r_squared")

    def test_result_frozen(self):
        """Result is immutable (frozen dataclass)."""
        ctrl, trt, Xc, Xt = _make_1d_data()
        ra = RegressionAdjustment()
        result = ra.fit_transform(ctrl, trt, Xc, Xt)

        with pytest.raises(AttributeError):
            result.ate = 999.0  # type: ignore[misc]

    def test_to_dict(self):
        """to_dict() returns a plain Python dict."""
        ctrl, trt, Xc, Xt = _make_1d_data()
        ra = RegressionAdjustment()
        result = ra.fit_transform(ctrl, trt, Xc, Xt)
        d = result.to_dict()

        assert isinstance(d, dict)
        assert "ate" in d
        assert isinstance(d["ate"], float)

    def test_repr(self):
        """__repr__ produces readable output."""
        ctrl, trt, Xc, Xt = _make_1d_data()
        ra = RegressionAdjustment()
        result = ra.fit_transform(ctrl, trt, Xc, Xt)
        text = repr(result)

        assert "RegressionAdjustmentResult" in text
        assert "ate" in text
        assert "se" in text

    def test_ci_contains_ate(self):
        """Confidence interval always contains the point estimate."""
        ctrl, trt, Xc, Xt = _make_1d_data()
        ra = RegressionAdjustment()
        result = ra.fit_transform(ctrl, trt, Xc, Xt)
        assert result.ci_lower <= result.ate <= result.ci_upper

    def test_pvalue_consistent_with_significance(self):
        """pvalue < alpha ↔ significant=True."""
        ctrl, trt, Xc, Xt = _make_1d_data(n=1000, effect=2.0)
        ra = RegressionAdjustment(alpha=0.05)
        result = ra.fit_transform(ctrl, trt, Xc, Xt)

        assert result.significant == (result.pvalue < 0.05)

    def test_variance_reduction_positive(self):
        """Variance reduction is positive when covariates are predictive."""
        ctrl, trt, Xc, Xt = _make_1d_data(n=500, noise=0.5)
        ra = RegressionAdjustment()
        result = ra.fit_transform(ctrl, trt, Xc, Xt)
        assert result.variance_reduction > 0.0

    def test_r_squared_range(self):
        """R-squared is in [0, 1]."""
        ctrl, trt, Xc, Xt = _make_1d_data()
        ra = RegressionAdjustment()
        result = ra.fit_transform(ctrl, trt, Xc, Xt)
        assert 0.0 <= result.r_squared <= 1.0


# ── Validation ───────────────────────────────────────────────────────


class TestValidation:
    """Input validation."""

    def test_alpha_out_of_range(self):
        """Alpha outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError, match=r"alpha"):
            RegressionAdjustment(alpha=1.5)

    def test_alpha_zero(self):
        """Alpha=0 raises ValueError."""
        with pytest.raises(ValueError, match=r"alpha"):
            RegressionAdjustment(alpha=0.0)

    def test_covariate_length_mismatch(self):
        """X_control rows != len(control) raises ValueError."""
        ctrl = np.array([1.0, 2.0, 3.0])
        trt = np.array([2.0, 3.0, 4.0])
        Xc = np.array([1.0, 2.0])  # wrong length
        Xt = np.array([2.0, 3.0, 4.0])

        ra = RegressionAdjustment()
        with pytest.raises(ValueError, match=r"rows"):
            ra.fit_transform(ctrl, trt, Xc, Xt)

    def test_covariate_column_mismatch(self):
        """Different number of covariates raises ValueError."""
        ctrl = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        trt = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        Xc = np.random.default_rng(0).normal(0, 1, (5, 2))
        Xt = np.random.default_rng(0).normal(0, 1, (5, 3))

        ra = RegressionAdjustment()
        with pytest.raises(ValueError, match=r"same number of covariates"):
            ra.fit_transform(ctrl, trt, Xc, Xt)

    def test_too_few_observations(self):
        """Single observation raises ValueError."""
        ctrl = np.array([1.0])
        trt = np.array([2.0, 3.0])
        Xc = np.array([1.0])
        Xt = np.array([2.0, 3.0])

        ra = RegressionAdjustment()
        with pytest.raises(ValueError, match=r"at least 2"):
            ra.fit_transform(ctrl, trt, Xc, Xt)

    def test_3d_covariate_raises(self):
        """3-D covariate array raises ValueError."""
        ctrl = np.array([1.0, 2.0, 3.0])
        trt = np.array([2.0, 3.0, 4.0])
        Xc = np.ones((3, 2, 2))
        Xt = np.ones((3, 2, 2))

        ra = RegressionAdjustment()
        with pytest.raises(ValueError, match=r"1-D or 2-D"):
            ra.fit_transform(ctrl, trt, Xc, Xt)

    def test_non_array_covariate_raises_typeerror(self):
        """Non-array covariate (e.g. int) raises TypeError."""
        ctrl = np.array([1.0, 2.0, 3.0])
        trt = np.array([2.0, 3.0, 4.0])

        ra = RegressionAdjustment()
        with pytest.raises(TypeError, match=r"array-like"):
            ra.fit_transform(ctrl, trt, 42, np.array([1.0, 2.0, 3.0]))

    def test_singular_design_matrix_raises(self):
        """Singular design matrix from perfectly collinear covariates raises."""
        # Create a situation where Z'Z is singular by making covariates
        # that are exact linear combinations (intercept is already in Z)
        ctrl = np.array([1.0, 2.0])
        trt = np.array([3.0, 4.0])
        # Constant covariates -> collinear with intercept
        Xc = np.array([[1.0, 1.0], [1.0, 1.0]])
        Xt = np.array([[1.0, 1.0], [1.0, 1.0]])

        ra = RegressionAdjustment()
        with pytest.raises(ValueError, match=r"singular"):
            ra.fit_transform(ctrl, trt, Xc, Xt)

    def test_too_many_covariates_raises(self):
        """More covariates than observations yields df < 1 error."""
        ctrl = np.array([1.0, 2.0])
        trt = np.array([3.0, 4.0])
        # 10 covariates for 4 observations -> Z has 1+1+10+10=22 cols, n=4, df<1
        rng = np.random.default_rng(42)
        Xc = rng.normal(0, 1, (2, 10))
        Xt = rng.normal(0, 1, (2, 10))

        ra = RegressionAdjustment()
        with pytest.raises(ValueError, match=r"Not enough observations|singular"):
            ra.fit_transform(ctrl, trt, Xc, Xt)


# ── Multi-covariate tests ───────────────────────────────────────────


class TestMultiCovariate:
    """Multiple covariates."""

    def test_two_covariates(self):
        """Works with 2-D covariate matrix."""
        ctrl, trt, Xc, Xt = _make_data(p_covariates=2)
        ra = RegressionAdjustment()
        result = ra.fit_transform(ctrl, trt, Xc, Xt)
        assert result.ate > 0

    def test_five_covariates(self):
        """Works with 5 covariates."""
        ctrl, trt, Xc, Xt = _make_data(n=500, p_covariates=5)
        ra = RegressionAdjustment()
        result = ra.fit_transform(ctrl, trt, Xc, Xt)
        assert result.se > 0

    def test_list_input(self):
        """Accepts list inputs (not just arrays)."""
        ctrl = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        trt = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        Xc = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
        Xt = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

        ra = RegressionAdjustment()
        result = ra.fit_transform(ctrl, trt, Xc, Xt)
        assert isinstance(result.ate, float)


# ── Integration ──────────────────────────────────────────────────────


class TestIntegration:
    """Integration tests."""

    def test_import_from_splita(self):
        """Can import RegressionAdjustment from splita top-level."""
        from splita import RegressionAdjustment as RA
        from splita import RegressionAdjustmentResult as RAR

        assert RA is not None
        assert RAR is not None

    def test_pipeline_with_experiment(self):
        """RA result can inform downstream analysis."""
        ctrl, trt, Xc, Xt = _make_1d_data(n=500, effect=1.0)
        ra = RegressionAdjustment()
        result = ra.fit_transform(ctrl, trt, Xc, Xt)

        # Basic sanity: result is usable
        d = result.to_dict()
        assert d["ate"] == result.ate
        assert d["pvalue"] == result.pvalue

    def test_custom_alpha(self):
        """Custom alpha propagates to result."""
        ctrl, trt, Xc, Xt = _make_1d_data()
        ra = RegressionAdjustment(alpha=0.01)
        result = ra.fit_transform(ctrl, trt, Xc, Xt)
        assert result.alpha == 0.01

    def test_wider_ci_with_smaller_alpha(self):
        """Smaller alpha → wider CI."""
        ctrl, trt, Xc, Xt = _make_1d_data(seed=99)

        r05 = RegressionAdjustment(alpha=0.05).fit_transform(ctrl, trt, Xc, Xt)
        r01 = RegressionAdjustment(alpha=0.01).fit_transform(ctrl, trt, Xc, Xt)

        width_05 = r05.ci_upper - r05.ci_lower
        width_01 = r01.ci_upper - r01.ci_lower
        assert width_01 > width_05
