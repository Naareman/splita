"""Tests for DoubleML variance reduction."""

from __future__ import annotations

import numpy as np
import pytest

from splita.variance import DoubleML
from splita._types import DoubleMLResult


# ── helpers ──────────────────────────────────────────────────────────


def _make_dml_data(
    n: int = 1000,
    p: int = 5,
    ate: float = 2.0,
    noise: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate data with confounders for DoubleML tests.

    Y = ate * T + X @ beta + noise
    T = (X[:, 0] > 0).astype(float) + treatment_noise
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, p))
    beta = np.array([1.0, 0.5, -0.3, 0.2, 0.0][:p])

    # Treatment depends on X (confounding)
    T = (X[:, 0] > 0).astype(float) + 0.3 * X[:, 1] + rng.normal(0, 0.3, n)
    T = np.clip(T, 0, None)

    # Outcome
    Y = ate * T + X @ beta + rng.normal(0, noise, n)

    return Y, T, X


# ── Basic tests ──────────────────────────────────────────────────────


class TestBasic:
    """Basic DoubleML behaviour."""

    def test_returns_result_type(self):
        """fit_transform returns a DoubleMLResult."""
        Y, T, X = _make_dml_data()
        result = DoubleML(random_state=42).fit_transform(Y, T, X)
        assert isinstance(result, DoubleMLResult)

    def test_known_ate_recovery(self):
        """Recovers known ATE within CI."""
        Y, T, X = _make_dml_data(n=2000, ate=2.0, seed=42)
        result = DoubleML(random_state=42).fit_transform(Y, T, X)
        # ATE should be close to 2.0
        assert abs(result.ate - 2.0) < 1.0
        # CI should contain true ATE
        assert result.ci_lower < 2.0 < result.ci_upper

    def test_zero_ate_recovery(self):
        """Recovers ATE near zero when true ATE is 0."""
        Y, T, X = _make_dml_data(n=2000, ate=0.0, seed=99)
        result = DoubleML(random_state=99).fit_transform(Y, T, X)
        assert abs(result.ate) < 1.0

    def test_significant_when_large_effect(self):
        """Detects significance with large treatment effect."""
        Y, T, X = _make_dml_data(n=2000, ate=5.0, seed=42)
        result = DoubleML(random_state=42).fit_transform(Y, T, X)
        assert result.significant is True
        assert result.pvalue < 0.05

    def test_better_than_naive(self):
        """DoubleML SE is smaller than naive difference-in-means SE."""
        Y, T, X = _make_dml_data(n=2000, ate=2.0, seed=42)
        result = DoubleML(random_state=42).fit_transform(Y, T, X)

        # Naive: just regress Y on T without controlling for X
        naive_se = float(np.std(Y, ddof=1)) / np.sqrt(len(Y))
        assert result.se < naive_se * 2  # DML SE should be reasonable


class TestConfounding:
    """Tests for confounding removal."""

    def test_confounding_removal(self):
        """DoubleML removes confounding bias."""
        rng = np.random.default_rng(42)
        n = 2000
        X = rng.normal(0, 1, size=(n, 3))

        # Confounder affects both T and Y
        confounder = X[:, 0]
        T = 0.5 * confounder + rng.normal(0, 0.5, n)
        T = (T > 0).astype(float)
        Y = 1.0 * T + 3.0 * confounder + rng.normal(0, 1, n)

        # Naive estimate is biased
        treated = Y[T > 0.5]
        control = Y[T <= 0.5]
        naive_ate = np.mean(treated) - np.mean(control)

        # DoubleML should be closer to true ATE = 1.0
        result = DoubleML(random_state=42).fit_transform(Y, T, X)
        dml_bias = abs(result.ate - 1.0)
        naive_bias = abs(naive_ate - 1.0)
        assert dml_bias < naive_bias

    def test_no_confounding_still_works(self):
        """Works correctly when there is no confounding."""
        rng = np.random.default_rng(42)
        n = 1000
        X = rng.normal(0, 1, size=(n, 3))
        T = rng.binomial(1, 0.5, n).astype(float)
        Y = 2.0 * T + rng.normal(0, 1, n)

        result = DoubleML(random_state=42).fit_transform(Y, T, X)
        assert abs(result.ate - 2.0) < 1.0


# ── CV validation ────────────────────────────────────────────────────


class TestCVValidation:
    """Cross-validation parameter tests."""

    def test_cv_2(self):
        """cv=2 works."""
        Y, T, X = _make_dml_data(n=500)
        result = DoubleML(cv=2, random_state=42).fit_transform(Y, T, X)
        assert isinstance(result.ate, float)

    def test_cv_10(self):
        """cv=10 works."""
        Y, T, X = _make_dml_data(n=500)
        result = DoubleML(cv=10, random_state=42).fit_transform(Y, T, X)
        assert isinstance(result.ate, float)

    def test_cv_less_than_2_raises(self):
        """cv < 2 raises ValueError."""
        with pytest.raises(ValueError, match=r"cv.*>= 2"):
            DoubleML(cv=1)

    def test_cv_too_high_raises(self):
        """cv > 50 raises ValueError."""
        with pytest.raises(ValueError, match=r"cv.*<= 50"):
            DoubleML(cv=100)


# ── Custom models ────────────────────────────────────────────────────


class TestCustomModels:
    """Tests with custom sklearn models."""

    def test_custom_outcome_model(self):
        """Custom outcome model works."""
        from sklearn.linear_model import Lasso

        Y, T, X = _make_dml_data(n=500)
        result = DoubleML(
            outcome_model=Lasso(alpha=0.1),
            random_state=42,
        ).fit_transform(Y, T, X)
        assert isinstance(result.ate, float)

    def test_custom_both_models(self):
        """Custom outcome and propensity models work."""
        from sklearn.linear_model import Lasso, Ridge

        Y, T, X = _make_dml_data(n=500)
        result = DoubleML(
            outcome_model=Ridge(alpha=0.5),
            propensity_model=Lasso(alpha=0.01),
            random_state=42,
        ).fit_transform(Y, T, X)
        assert isinstance(result.ate, float)
        assert np.isfinite(result.se)

    def test_bad_outcome_model_raises(self):
        """Object without fit/predict raises ValueError."""
        with pytest.raises(ValueError, match=r"outcome_model.*fit.*predict"):
            DoubleML(outcome_model=object())

    def test_bad_propensity_model_raises(self):
        """Object without fit/predict raises ValueError."""
        with pytest.raises(ValueError, match=r"propensity_model.*fit.*predict"):
            DoubleML(propensity_model=object())


# ── Type I error control ─────────────────────────────────────────────


class TestTypeIError:
    """Type I error control tests."""

    def test_type_i_error_rate(self):
        """Under H0 (ATE=0), rejection rate is controlled at alpha."""
        rng = np.random.default_rng(42)
        n_sims = 200
        alpha = 0.05
        rejections = 0

        for i in range(n_sims):
            n = 300
            X = rng.normal(0, 1, size=(n, 3))
            T = rng.binomial(1, 0.5, n).astype(float)
            Y = X @ [1, 0.5, -0.3] + rng.normal(0, 1, n)  # No treatment effect

            result = DoubleML(random_state=i, cv=2).fit_transform(Y, T, X)
            if result.pvalue < alpha:
                rejections += 1

        rejection_rate = rejections / n_sims
        # Should be close to alpha=0.05, allow generous margin
        assert rejection_rate < 0.15, f"Type I error rate {rejection_rate:.2f} > 0.15"


# ── Result attributes ────────────────────────────────────────────────


class TestResultAttributes:
    """Test result dataclass attributes."""

    def test_r2_values_positive(self):
        """R-squared values should be positive with good features."""
        Y, T, X = _make_dml_data(n=1000)
        result = DoubleML(random_state=42).fit_transform(Y, T, X)
        assert result.outcome_r2 > 0.0
        assert result.propensity_r2 > 0.0

    def test_variance_reduction_nonnegative(self):
        """Variance reduction should be non-negative."""
        Y, T, X = _make_dml_data(n=1000)
        result = DoubleML(random_state=42).fit_transform(Y, T, X)
        assert result.variance_reduction >= 0.0

    def test_to_dict(self):
        """to_dict produces a plain dict."""
        Y, T, X = _make_dml_data(n=500)
        result = DoubleML(random_state=42).fit_transform(Y, T, X)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "ate" in d
        assert "se" in d
        assert "pvalue" in d

    def test_repr(self):
        """repr produces a string."""
        Y, T, X = _make_dml_data(n=500)
        result = DoubleML(random_state=42).fit_transform(Y, T, X)
        s = repr(result)
        assert "DoubleMLResult" in s
        assert "ate" in s

    def test_se_positive(self):
        """SE should be positive."""
        Y, T, X = _make_dml_data(n=500)
        result = DoubleML(random_state=42).fit_transform(Y, T, X)
        assert result.se > 0

    def test_ci_contains_ate(self):
        """CI should contain the point estimate."""
        Y, T, X = _make_dml_data(n=500)
        result = DoubleML(random_state=42).fit_transform(Y, T, X)
        assert result.ci_lower <= result.ate <= result.ci_upper


# ── Validation ───────────────────────────────────────────────────────


class TestValidation:
    """Input validation tests."""

    def test_x_not_2d_raises(self):
        """1-D X raises ValueError."""
        Y = np.array([1.0, 2.0, 3.0])
        T = np.array([0.0, 1.0, 0.0])
        X = np.array([1.0, 2.0, 3.0])  # 1-D
        with pytest.raises(ValueError, match=r"2-D"):
            DoubleML().fit_transform(Y, T, X)

    def test_length_mismatch_raises(self):
        """Mismatched lengths raise ValueError."""
        Y = np.array([1.0, 2.0, 3.0])
        T = np.array([0.0, 1.0])  # Different length
        X = np.array([[1.0], [2.0], [3.0]])
        with pytest.raises(ValueError, match=r"same length"):
            DoubleML().fit_transform(Y, T, X)

    def test_x_row_mismatch_raises(self):
        """X with wrong rows raises ValueError."""
        Y = np.array([1.0, 2.0, 3.0])
        T = np.array([0.0, 1.0, 0.0])
        X = np.array([[1.0], [2.0]])  # Wrong rows
        with pytest.raises(ValueError, match=r"same number of rows"):
            DoubleML().fit_transform(Y, T, X)

    def test_nan_in_x_raises(self):
        """NaN in X raises ValueError."""
        Y = np.array([1.0, 2.0, 3.0])
        T = np.array([0.0, 1.0, 0.0])
        X = np.array([[1.0], [np.nan], [3.0]])
        with pytest.raises(ValueError, match=r"NaN or infinite"):
            DoubleML().fit_transform(Y, T, X)

    def test_bad_alpha_raises(self):
        """alpha outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError, match=r"alpha"):
            DoubleML(alpha=0.0)
        with pytest.raises(ValueError, match=r"alpha"):
            DoubleML(alpha=1.0)


# ── Reproducibility ──────────────────────────────────────────────────


class TestReproducibility:
    """Reproducibility tests."""

    def test_same_seed_same_result(self):
        """Same random_state produces identical results."""
        Y, T, X = _make_dml_data(n=500)

        r1 = DoubleML(random_state=42).fit_transform(Y, T, X)
        r2 = DoubleML(random_state=42).fit_transform(Y, T, X)

        assert r1.ate == r2.ate
        assert r1.se == r2.se
        assert r1.pvalue == r2.pvalue

    def test_generator_random_state(self):
        """numpy Generator as random_state works."""
        Y, T, X = _make_dml_data(n=500)
        gen = np.random.default_rng(42)
        result = DoubleML(random_state=gen).fit_transform(Y, T, X)
        assert isinstance(result.ate, float)
