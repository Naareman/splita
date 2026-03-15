"""Tests for MetricSensitivity and VarianceEstimator — pre-experiment validation (M13)."""

from __future__ import annotations

import numpy as np
import pytest

from splita._types import MetricSensitivityResult, VarianceEstimateResult
from splita.diagnostics import MetricSensitivity, VarianceEstimator


# ─── VarianceEstimator tests ─────────────────────────────────────


class TestVarianceEstimator:
    def test_normal_data(self):
        rng = np.random.default_rng(42)
        data = rng.normal(10.0, 2.0, size=500)
        result = VarianceEstimator().fit(data).result()
        assert isinstance(result, VarianceEstimateResult)
        assert abs(result.mean - 10.0) < 0.5
        assert abs(result.std - 2.0) < 0.5
        assert not result.is_heavy_tailed
        assert not result.is_skewed

    def test_heavy_tailed_detection(self):
        rng = np.random.default_rng(42)
        # t-distribution with 3 df has kurtosis = inf (excess kurt >> 5)
        from scipy.stats import t as t_dist

        data = t_dist.rvs(df=3, size=5000, random_state=42)
        result = VarianceEstimator().fit(data).result()
        assert result.is_heavy_tailed
        assert result.kurtosis > 5.0
        assert any("outlier" in r.lower() for r in result.recommendations)

    def test_skewed_detection(self):
        rng = np.random.default_rng(42)
        # Exponential distribution has skewness = 2
        data = rng.exponential(scale=1.0, size=5000) ** 3  # cube to increase skew
        result = VarianceEstimator().fit(data).result()
        assert result.is_skewed
        assert abs(result.skewness) > 2.0
        assert any("mann-whitney" in r.lower() for r in result.recommendations)

    def test_percentiles(self):
        data = np.arange(1.0, 101.0)
        result = VarianceEstimator().fit(data).result()
        assert "p5" in result.percentiles
        assert "p25" in result.percentiles
        assert "p50" in result.percentiles
        assert "p75" in result.percentiles
        assert "p95" in result.percentiles
        assert result.percentiles["p50"] == pytest.approx(50.5, abs=1.0)

    def test_well_behaved_recommendations(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200)
        result = VarianceEstimator().fit(data).result()
        assert any("well-behaved" in r.lower() for r in result.recommendations)

    def test_to_dict(self):
        data = np.arange(1.0, 101.0)
        result = VarianceEstimator().fit(data).result()
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "mean" in d
        assert "percentiles" in d

    def test_repr(self):
        data = np.arange(1.0, 101.0)
        r = repr(VarianceEstimator().fit(data).result())
        assert "VarianceEstimateResult" in r

    def test_not_fitted(self):
        with pytest.raises(RuntimeError, match="must be fitted"):
            VarianceEstimator().result()

    def test_too_short(self):
        with pytest.raises(ValueError, match="at least"):
            VarianceEstimator().fit(np.array([1.0, 2.0, 3.0]))


# ─── MetricSensitivity tests ────────────────────────────────────


class TestMetricSensitivity:
    def test_large_mde_high_power(self):
        rng = np.random.default_rng(42)
        data = rng.normal(10.0, 1.0, size=500)
        result = MetricSensitivity(
            n_simulations=200, random_state=0
        ).run(data, mde=2.0)
        assert isinstance(result, MetricSensitivityResult)
        assert result.estimated_power > 0.7
        assert result.is_sensitive

    def test_tiny_mde_low_power(self):
        rng = np.random.default_rng(42)
        data = rng.normal(10.0, 5.0, size=50)
        result = MetricSensitivity(
            n_simulations=200, random_state=0
        ).run(data, mde=0.01)
        assert result.estimated_power < 0.3

    def test_recommended_n_scales_with_variance(self):
        rng = np.random.default_rng(42)
        data_low_var = rng.normal(10.0, 1.0, size=200)
        data_high_var = rng.normal(10.0, 10.0, size=200)
        r_low = MetricSensitivity(random_state=0).run(data_low_var, mde=0.5)
        r_high = MetricSensitivity(random_state=0).run(data_high_var, mde=0.5)
        assert r_high.recommended_n > r_low.recommended_n

    def test_zero_variance(self):
        data = np.ones(100)
        result = MetricSensitivity(random_state=0).run(data, mde=0.5)
        assert result.estimated_power == 1.0
        assert result.estimated_std == 0.0

    def test_estimated_std(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 3.0, size=5000)
        result = MetricSensitivity(random_state=0).run(data, mde=1.0)
        assert abs(result.estimated_std - 3.0) < 0.2

    def test_to_dict(self):
        data = np.arange(1.0, 101.0)
        result = MetricSensitivity(n_simulations=50, random_state=0).run(data, mde=5.0)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "estimated_power" in d

    def test_repr(self):
        data = np.arange(1.0, 101.0)
        result = MetricSensitivity(n_simulations=50, random_state=0).run(data, mde=5.0)
        r = repr(result)
        assert "MetricSensitivityResult" in r


# ─── Validation tests ────────────────────────────────────────────


class TestPreExperimentValidation:
    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="must be in"):
            MetricSensitivity(alpha=1.5)

    def test_invalid_n_simulations(self):
        with pytest.raises(ValueError, match="must be >= 10"):
            MetricSensitivity(n_simulations=5)

    def test_negative_mde(self):
        data = np.ones(100)
        with pytest.raises(ValueError, match="must be > 0"):
            MetricSensitivity(random_state=0).run(data, mde=-1.0)

    def test_data_too_short(self):
        with pytest.raises(ValueError, match="at least"):
            MetricSensitivity(random_state=0).run(np.array([1.0]), mde=0.5)
