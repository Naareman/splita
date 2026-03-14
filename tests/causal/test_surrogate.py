"""Tests for SurrogateEstimator (M15: Long-run Effect Estimation)."""

from __future__ import annotations

import numpy as np
import pytest

from splita.causal.surrogate import SurrogateEstimator


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def valid_data(rng):
    """Generate data where short-term is a strong surrogate for long-term."""
    n = 200
    short = rng.normal(5, 1, n)
    treatment = np.array([0] * 100 + [1] * 100)
    # long = 2*short + 3*treatment + noise  => strong relationship
    long = 2 * short + 3 * treatment + rng.normal(0, 0.5, n)
    return short, long, treatment


class TestSurrogateFit:
    """Tests for SurrogateEstimator.fit()."""

    def test_fit_returns_self(self, valid_data):
        short, long, trt = valid_data
        est = SurrogateEstimator(random_state=42)
        result = est.fit(short, long, trt)
        assert result is est

    def test_valid_surrogate_high_r2(self, valid_data):
        short, long, trt = valid_data
        est = SurrogateEstimator(random_state=42)
        est.fit(short, long, trt)
        result = est.predict_long_term_effect(
            short_term_control=np.random.default_rng(0).normal(5, 1, 50),
            short_term_treatment=np.random.default_rng(1).normal(5, 1, 50),
        )
        assert result.is_valid_surrogate is True
        assert result.surrogate_r2 > 0.3

    def test_weak_surrogate_low_r2(self, rng):
        """When short-term has no relationship to long-term, R2 is low."""
        n = 200
        short = rng.normal(0, 1, n)
        treatment = np.array([0] * 100 + [1] * 100)
        # long is pure noise — no relationship to short
        long = rng.normal(0, 10, n)
        est = SurrogateEstimator(random_state=42)
        with pytest.warns(RuntimeWarning, match="low R-squared"):
            est.fit(short, long, treatment)

    def test_predict_lift_direction(self, valid_data):
        """The predicted lift should be positive when treatment has positive effect."""
        short, long, trt = valid_data
        est = SurrogateEstimator(random_state=42)
        est.fit(short, long, trt)
        result = est.predict_long_term_effect(
            short_term_control=np.random.default_rng(0).normal(5, 1, 50),
            short_term_treatment=np.random.default_rng(1).normal(5, 1, 50),
        )
        assert result.predicted_long_term_lift > 0

    def test_prediction_ci_contains_lift(self, valid_data):
        """CI should contain the predicted lift."""
        short, long, trt = valid_data
        est = SurrogateEstimator(random_state=42)
        est.fit(short, long, trt)
        result = est.predict_long_term_effect(
            short_term_control=np.random.default_rng(0).normal(5, 1, 50),
            short_term_treatment=np.random.default_rng(1).normal(5, 1, 50),
        )
        ci_lo, ci_hi = result.prediction_ci
        assert ci_lo <= result.predicted_long_term_lift <= ci_hi

    def test_prediction_ci_is_tuple(self, valid_data):
        short, long, trt = valid_data
        est = SurrogateEstimator(random_state=42)
        est.fit(short, long, trt)
        result = est.predict_long_term_effect(
            short_term_control=np.random.default_rng(0).normal(5, 1, 50),
            short_term_treatment=np.random.default_rng(1).normal(5, 1, 50),
        )
        assert isinstance(result.prediction_ci, tuple)
        assert len(result.prediction_ci) == 2


class TestSurrogateValidation:
    """Tests for input validation."""

    def test_treatment_not_binary(self, rng):
        short = rng.normal(0, 1, 100)
        long = rng.normal(0, 1, 100)
        trt = rng.integers(0, 3, 100).astype(float)
        est = SurrogateEstimator()
        with pytest.raises(ValueError, match="must contain only 0 and 1"):
            est.fit(short, long, trt)

    def test_mismatched_lengths(self, rng):
        est = SurrogateEstimator()
        with pytest.raises(ValueError, match="same length"):
            est.fit(
                rng.normal(0, 1, 50),
                rng.normal(0, 1, 100),
                np.array([0] * 50),
            )

    def test_too_few_samples(self, rng):
        est = SurrogateEstimator()
        with pytest.raises(ValueError, match="at least"):
            est.fit(
                rng.normal(0, 1, 3),
                rng.normal(0, 1, 3),
                np.array([0.0, 1.0, 0.0]),
            )

    def test_predict_before_fit(self):
        est = SurrogateEstimator()
        with pytest.raises(RuntimeError, match="must be fitted"):
            est.predict_long_term_effect(
                short_term_control=[1.0, 2.0, 3.0],
                short_term_treatment=[1.0, 2.0, 3.0],
            )

    def test_predict_too_few_samples(self, valid_data):
        short, long, trt = valid_data
        est = SurrogateEstimator(random_state=42)
        est.fit(short, long, trt)
        with pytest.raises(ValueError, match="at least"):
            est.predict_long_term_effect(
                short_term_control=[1.0],
                short_term_treatment=[1.0],
            )

    def test_non_array_input(self):
        est = SurrogateEstimator()
        with pytest.raises(TypeError, match="array-like"):
            est.fit("not_an_array", [1, 2, 3, 4, 5], [0, 0, 0, 1, 1])


class TestSurrogateResult:
    """Tests for result properties."""

    def test_to_dict(self, valid_data):
        short, long, trt = valid_data
        est = SurrogateEstimator(random_state=42)
        est.fit(short, long, trt)
        result = est.predict_long_term_effect(
            short_term_control=np.random.default_rng(0).normal(5, 1, 50),
            short_term_treatment=np.random.default_rng(1).normal(5, 1, 50),
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "predicted_long_term_lift" in d
        assert "prediction_ci" in d
        assert "surrogate_r2" in d
        assert "is_valid_surrogate" in d

    def test_repr(self, valid_data):
        short, long, trt = valid_data
        est = SurrogateEstimator(random_state=42)
        est.fit(short, long, trt)
        result = est.predict_long_term_effect(
            short_term_control=np.random.default_rng(0).normal(5, 1, 50),
            short_term_treatment=np.random.default_rng(1).normal(5, 1, 50),
        )
        r = repr(result)
        assert "SurrogateResult" in r

    def test_custom_estimator(self, valid_data):
        """Using a custom sklearn estimator should work."""
        from sklearn.linear_model import LinearRegression

        short, long, trt = valid_data
        est = SurrogateEstimator(estimator=LinearRegression(), random_state=42)
        est.fit(short, long, trt)
        result = est.predict_long_term_effect(
            short_term_control=np.random.default_rng(0).normal(5, 1, 50),
            short_term_treatment=np.random.default_rng(1).normal(5, 1, 50),
        )
        assert result.is_valid_surrogate is True
