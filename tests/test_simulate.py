"""Tests for splita.simulate — A/B test simulation."""

from __future__ import annotations

import numpy as np
import pytest

from splita.simulate import simulate
from splita._types import SimulationResult


# ─── Basic Functionality ─────────────────────────────────────────────


class TestSimulateBasic:
    def test_returns_simulation_result(self):
        result = simulate(0.10, 0.02, 5000, random_state=42)
        assert isinstance(result, SimulationResult)

    def test_power_in_range(self):
        result = simulate(0.10, 0.02, 5000, random_state=42)
        assert 0.0 <= result.estimated_power <= 1.0

    def test_median_pvalue_in_range(self):
        result = simulate(0.10, 0.02, 5000, random_state=42)
        assert 0.0 <= result.median_pvalue <= 1.0

    def test_significant_rate_equals_power(self):
        result = simulate(0.10, 0.05, 5000, random_state=42)
        assert result.significant_rate == result.estimated_power

    def test_false_negative_rate_complement(self):
        result = simulate(0.10, 0.02, 5000, random_state=42)
        assert abs(result.false_negative_rate + result.estimated_power - 1.0) < 1e-10

    def test_ci_width_positive(self):
        result = simulate(0.10, 0.02, 5000, random_state=42)
        assert result.ci_width_median > 0


# ─── Power Behavior ─────────────────────────────────────────────────


class TestSimulatePower:
    def test_large_effect_high_power(self):
        result = simulate(0.10, 0.10, 10000, n_simulations=500, random_state=42)
        assert result.estimated_power > 0.9

    def test_small_effect_low_power(self):
        result = simulate(0.10, 0.001, 100, n_simulations=500, random_state=42)
        assert result.estimated_power < 0.3

    def test_more_samples_more_power(self):
        r_small = simulate(0.10, 0.02, 500, n_simulations=500, random_state=42)
        r_large = simulate(0.10, 0.02, 10000, n_simulations=500, random_state=42)
        assert r_large.estimated_power > r_small.estimated_power


# ─── Metric Types ───────────────────────────────────────────────────


class TestSimulateMetrics:
    def test_conversion_metric(self):
        result = simulate(0.10, 0.02, 5000, metric="conversion", random_state=42)
        assert isinstance(result, SimulationResult)

    def test_continuous_metric(self):
        result = simulate(10.0, 0.5, 5000, metric="continuous", random_state=42)
        assert isinstance(result, SimulationResult)
        assert result.median_lift > 0

    def test_continuous_high_power(self):
        result = simulate(
            10.0, 2.0, 5000, metric="continuous",
            n_simulations=500, random_state=42,
        )
        assert result.estimated_power > 0.8


# ─── Recommendations ────────────────────────────────────────────────


class TestSimulateRecommendations:
    def test_good_power_recommendation(self):
        result = simulate(0.10, 0.10, 10000, n_simulations=200, random_state=42)
        assert "Good power" in result.recommendation or "good" in result.recommendation.lower()

    def test_low_power_recommendation(self):
        result = simulate(0.10, 0.001, 100, n_simulations=200, random_state=42)
        assert "Low power" in result.recommendation or "power" in result.recommendation.lower()


# ─── Reproducibility ────────────────────────────────────────────────


class TestSimulateReproducibility:
    def test_same_seed_same_result(self):
        r1 = simulate(0.10, 0.02, 5000, random_state=42)
        r2 = simulate(0.10, 0.02, 5000, random_state=42)
        assert r1.estimated_power == r2.estimated_power
        assert r1.median_pvalue == r2.median_pvalue


# ─── Validation ──────────────────────────────────────────────────────


class TestSimulateValidation:
    def test_rejects_small_n(self):
        with pytest.raises(ValueError, match="n_per_variant"):
            simulate(0.10, 0.02, 1)

    def test_rejects_small_n_simulations(self):
        with pytest.raises(ValueError, match="n_simulations"):
            simulate(0.10, 0.02, 100, n_simulations=5)

    def test_rejects_bad_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            simulate(0.10, 0.02, 100, alpha=1.5)

    def test_rejects_bad_metric(self):
        with pytest.raises(ValueError, match="metric"):
            simulate(0.10, 0.02, 100, metric="unknown")


# ─── Serialization ──────────────────────────────────────────────────


class TestSimulateSerialization:
    def test_to_dict(self):
        result = simulate(0.10, 0.02, 5000, random_state=42, n_simulations=100)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "estimated_power" in d
        assert "recommendation" in d

    def test_repr(self):
        result = simulate(0.10, 0.02, 5000, random_state=42, n_simulations=100)
        text = repr(result)
        assert "SimulationResult" in text
