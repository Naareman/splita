"""Tests for RobustMeanEstimator (robust treatment effect estimation)."""

from __future__ import annotations

import numpy as np
import pytest

from splita.variance.robust_estimators import RobustMeanEstimator


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestRobustBasic:
    """Basic functionality tests."""

    def test_huber_known_effect(self, rng):
        """Huber estimator should recover a known effect."""
        ctrl = rng.normal(10, 1, 200)
        trt = rng.normal(12, 1, 200)

        r = RobustMeanEstimator(method="huber").fit_transform(ctrl, trt)
        assert abs(r.ate - 2.0) < 0.5
        assert r.significant is True
        assert r.method == "huber"

    def test_median_of_means_known_effect(self, rng):
        """Median of means should recover a known effect."""
        ctrl = rng.normal(10, 1, 200)
        trt = rng.normal(12, 1, 200)

        r = RobustMeanEstimator(method="median_of_means").fit_transform(ctrl, trt)
        assert abs(r.ate - 2.0) < 1.0
        assert r.method == "median_of_means"

    def test_catoni_known_effect(self, rng):
        """Catoni estimator should recover a known effect."""
        ctrl = rng.normal(10, 1, 200)
        trt = rng.normal(12, 1, 200)

        r = RobustMeanEstimator(method="catoni").fit_transform(ctrl, trt)
        assert abs(r.ate - 2.0) < 1.0
        assert r.method == "catoni"

    def test_reduces_outlier_influence(self, rng):
        """Robust estimator should be less affected by outliers than raw mean."""
        ctrl = np.concatenate([rng.normal(10, 1, 190), [100, 200]])
        trt = np.concatenate([rng.normal(12, 1, 190), [-100, -200]])

        r_robust = RobustMeanEstimator(method="huber").fit_transform(ctrl, trt)
        raw_diff = float(np.mean(trt) - np.mean(ctrl))

        # Robust estimate should be closer to true effect (2.0)
        assert abs(r_robust.ate - 2.0) < abs(raw_diff - 2.0)

    def test_no_effect(self, rng):
        """No effect should yield non-significant result."""
        ctrl = rng.normal(10, 1, 200)
        trt = rng.normal(10, 1, 200)

        r = RobustMeanEstimator().fit_transform(ctrl, trt)
        assert abs(r.ate) < 1.0

    def test_ci_contains_estimate(self, rng):
        """CI should contain the point estimate."""
        ctrl = rng.normal(10, 1, 200)
        trt = rng.normal(12, 1, 200)

        r = RobustMeanEstimator().fit_transform(ctrl, trt)
        assert r.ci_lower <= r.ate <= r.ci_upper

    def test_ci_contains_true_effect(self, rng):
        """CI should contain the true effect."""
        true_effect = 2.0
        ctrl = rng.normal(10, 1, 500)
        trt = rng.normal(10 + true_effect, 1, 500)

        r = RobustMeanEstimator().fit_transform(ctrl, trt)
        assert r.ci_lower <= true_effect <= r.ci_upper

    def test_se_positive(self, rng):
        """Standard error should be positive."""
        ctrl = rng.normal(10, 1, 100)
        trt = rng.normal(12, 1, 100)

        r = RobustMeanEstimator().fit_transform(ctrl, trt)
        assert r.se > 0

    def test_pvalue_range(self, rng):
        """p-value should be in [0, 1]."""
        ctrl = rng.normal(10, 1, 100)
        trt = rng.normal(12, 1, 100)

        r = RobustMeanEstimator().fit_transform(ctrl, trt)
        assert 0.0 <= r.pvalue <= 1.0

    def test_significant_flag(self, rng):
        """significant should match pvalue < alpha."""
        ctrl = rng.normal(10, 1, 200)
        trt = rng.normal(12, 1, 200)

        alpha = 0.05
        r = RobustMeanEstimator(alpha=alpha).fit_transform(ctrl, trt)
        assert r.significant == (r.pvalue < alpha)

    def test_heavy_tailed_data(self, rng):
        """Robust estimators should handle heavy-tailed distributions."""
        # Cauchy-like: normal with occasional extreme values
        ctrl = rng.standard_cauchy(200) + 10
        trt = rng.standard_cauchy(200) + 12

        r = RobustMeanEstimator(method="huber").fit_transform(ctrl, trt)
        # Just check it runs and produces finite values
        assert np.isfinite(r.ate)
        assert np.isfinite(r.se)

    def test_all_methods_produce_results(self, rng):
        """All three methods should produce valid results."""
        ctrl = rng.normal(10, 1, 100)
        trt = rng.normal(12, 1, 100)

        for method in ["huber", "median_of_means", "catoni"]:
            r = RobustMeanEstimator(method=method).fit_transform(ctrl, trt)
            assert np.isfinite(r.ate)
            assert r.method == method


class TestRobustValidation:
    """Input validation tests."""

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method"):
            RobustMeanEstimator(method="invalid")

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            RobustMeanEstimator(alpha=1.5)

    def test_too_few_samples(self, rng):
        with pytest.raises(ValueError, match="at least"):
            RobustMeanEstimator().fit_transform([1.0], [2.0])

    def test_non_array_input(self):
        with pytest.raises(TypeError, match="array-like"):
            RobustMeanEstimator().fit_transform("not_array", [1, 2, 3])


class TestRobustResult:
    """Result object tests."""

    def test_to_dict(self, rng):
        r = RobustMeanEstimator().fit_transform(
            rng.normal(10, 1, 100), rng.normal(12, 1, 100)
        )
        d = r.to_dict()
        assert isinstance(d, dict)
        assert "ate" in d
        assert "method" in d

    def test_repr(self, rng):
        r = RobustMeanEstimator().fit_transform(
            rng.normal(10, 1, 100), rng.normal(12, 1, 100)
        )
        assert "RobustMeanResult" in repr(r)
