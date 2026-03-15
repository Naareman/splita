"""Tests for splita.core.power_simulation — Monte Carlo power analysis."""

from __future__ import annotations

import numpy as np
import pytest

from splita import SampleSize
from splita._types import PowerSimulationResult
from splita.core import PowerSimulation


# ═══════════════════════════════════════════════════════════════════════
# for_proportion
# ═══════════════════════════════════════════════════════════════════════


class TestForProportion:
    """Tests for PowerSimulation.for_proportion convenience method."""

    @pytest.mark.slow
    def test_matches_analytical_80pct_power(self):
        """Simulated power for the analytical n should be ~80% (within 5pp)."""
        ss = SampleSize.for_proportion(0.10, 0.02)
        result = PowerSimulation.for_proportion(
            0.10, 0.02, ss.n_per_variant,
            n_simulations=2000, random_state=42,
        )
        assert abs(result.power - 0.80) < 0.05

    def test_basic_result_fields(self):
        """Result has correct types and field values."""
        result = PowerSimulation.for_proportion(
            0.10, 0.05, 500,
            n_simulations=100, random_state=0,
        )
        assert isinstance(result, PowerSimulationResult)
        assert result.n_simulations == 100
        assert result.n_per_variant == 500
        assert result.alpha == 0.05
        assert result.power == result.rejection_rate

    def test_power_increases_with_sample_size(self):
        """Larger sample → higher power."""
        small = PowerSimulation.for_proportion(
            0.10, 0.03, 200,
            n_simulations=300, random_state=1,
        )
        large = PowerSimulation.for_proportion(
            0.10, 0.03, 2000,
            n_simulations=300, random_state=1,
        )
        assert large.power > small.power

    def test_power_increases_with_effect(self):
        """Larger MDE → higher power at the same sample size."""
        small_effect = PowerSimulation.for_proportion(
            0.10, 0.01, 1000,
            n_simulations=300, random_state=2,
        )
        large_effect = PowerSimulation.for_proportion(
            0.10, 0.05, 1000,
            n_simulations=300, random_state=2,
        )
        assert large_effect.power > small_effect.power


# ═══════════════════════════════════════════════════════════════════════
# for_mean
# ═══════════════════════════════════════════════════════════════════════


class TestForMean:
    """Tests for PowerSimulation.for_mean convenience method."""

    @pytest.mark.slow
    def test_matches_analytical_80pct_power(self):
        """Simulated power for the analytical n should be ~80% (within 5pp)."""
        ss = SampleSize.for_mean(10.0, 2.0, 0.5)
        result = PowerSimulation.for_mean(
            10.0, 2.0, 0.5, ss.n_per_variant,
            n_simulations=2000, random_state=42,
        )
        assert abs(result.power - 0.80) < 0.05

    def test_basic_result_fields(self):
        """Result has correct types and field values."""
        result = PowerSimulation.for_mean(
            10.0, 2.0, 1.0, 200,
            n_simulations=100, random_state=0,
        )
        assert isinstance(result, PowerSimulationResult)
        assert result.n_simulations == 100
        assert result.n_per_variant == 200
        assert 0.0 <= result.power <= 1.0

    def test_mean_effect_near_true_mde(self):
        """Average observed effect should be close to the true MDE."""
        mde = 1.0
        result = PowerSimulation.for_mean(
            10.0, 2.0, mde, 500,
            n_simulations=500, random_state=7,
        )
        assert abs(result.mean_effect - mde) < 0.3

    def test_negative_std_raises(self):
        """Negative std should raise ValueError."""
        with pytest.raises(ValueError, match="baseline_std"):
            PowerSimulation.for_mean(10.0, -1.0, 1.0, 100, n_simulations=50)


# ═══════════════════════════════════════════════════════════════════════
# Custom DGP
# ═══════════════════════════════════════════════════════════════════════


class TestCustomDGP:
    """Tests using custom data-generating processes."""

    def test_cuped_adjusted_higher_power(self):
        """A CUPED-like variance-reduced DGP should yield higher power."""
        n = 500

        # Unadjusted DGP (high variance)
        def dgp_unadjusted(rng):
            ctrl = rng.normal(10, 5, n)
            trt = rng.normal(10.5, 5, n)
            return ctrl, trt

        # Adjusted DGP (lower variance, same effect)
        def dgp_adjusted(rng):
            ctrl = rng.normal(10, 2, n)
            trt = rng.normal(10.5, 2, n)
            return ctrl, trt

        unadj = PowerSimulation(n_simulations=500, random_state=99).run(
            dgp_unadjusted, n,
        )
        adj = PowerSimulation(n_simulations=500, random_state=99).run(
            dgp_adjusted, n,
        )
        assert adj.power > unadj.power

    def test_dgp_called_correct_number_of_times(self):
        """DGP should be called exactly n_simulations times."""
        call_count = 0
        n = 50

        def counting_dgp(rng):
            nonlocal call_count
            call_count += 1
            ctrl = rng.normal(0, 1, n)
            trt = rng.normal(0, 1, n)
            return ctrl, trt

        n_sims = 77
        PowerSimulation(n_simulations=n_sims, random_state=0).run(counting_dgp, n)
        assert call_count == n_sims


# ═══════════════════════════════════════════════════════════════════════
# Type I error / zero effect
# ═══════════════════════════════════════════════════════════════════════


class TestTypeIError:
    """Zero effect should produce rejection rate near alpha."""

    @pytest.mark.slow
    def test_proportion_zero_effect(self):
        """Power under H0 should be approximately alpha."""
        result = PowerSimulation.for_proportion(
            0.10, 0.0, 1000,
            n_simulations=2000, alpha=0.05, random_state=42,
        )
        # Should be near 0.05, allow generous tolerance
        assert abs(result.power - 0.05) < 0.03

    @pytest.mark.slow
    def test_mean_zero_effect(self):
        """Power under H0 should be approximately alpha."""
        result = PowerSimulation.for_mean(
            10.0, 2.0, 0.0, 500,
            n_simulations=2000, alpha=0.05, random_state=42,
        )
        assert abs(result.power - 0.05) < 0.03


# ═══════════════════════════════════════════════════════════════════════
# Large effect → power ≈ 1.0
# ═══════════════════════════════════════════════════════════════════════


class TestLargeEffect:
    """Very large effects should yield power close to 1.0."""

    def test_proportion_large_effect(self):
        """Huge MDE on proportions → power ≈ 1.0."""
        result = PowerSimulation.for_proportion(
            0.10, 0.30, 500,
            n_simulations=200, random_state=0,
        )
        assert result.power > 0.95

    def test_mean_large_effect(self):
        """Huge MDE on means → power ≈ 1.0."""
        result = PowerSimulation.for_mean(
            10.0, 1.0, 5.0, 200,
            n_simulations=200, random_state=0,
        )
        assert result.power > 0.95


# ═══════════════════════════════════════════════════════════════════════
# Wilson CI
# ═══════════════════════════════════════════════════════════════════════


class TestWilsonCI:
    """Wilson confidence interval on the power estimate."""

    def test_ci_contains_true_power(self):
        """CI should contain the observed power (by construction)."""
        result = PowerSimulation.for_mean(
            10.0, 2.0, 0.5, 200,
            n_simulations=500, random_state=0,
        )
        assert result.ci_power_lower <= result.power <= result.ci_power_upper

    def test_ci_bounds_valid(self):
        """CI lower <= CI upper, both in [0, 1]."""
        result = PowerSimulation.for_proportion(
            0.10, 0.02, 1000,
            n_simulations=200, random_state=0,
        )
        assert 0.0 <= result.ci_power_lower <= result.ci_power_upper <= 1.0

    def test_ci_narrows_with_more_sims(self):
        """More simulations → narrower CI."""
        few = PowerSimulation.for_mean(
            10.0, 2.0, 0.5, 200,
            n_simulations=100, random_state=0,
        )
        many = PowerSimulation.for_mean(
            10.0, 2.0, 0.5, 200,
            n_simulations=1000, random_state=0,
        )
        width_few = few.ci_power_upper - few.ci_power_lower
        width_many = many.ci_power_upper - many.ci_power_lower
        assert width_many < width_few


# ═══════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════


class TestValidation:
    """Input validation for PowerSimulation."""

    def test_n_simulations_too_low(self):
        """n_simulations < 10 raises ValueError."""
        with pytest.raises(ValueError, match="n_simulations"):
            PowerSimulation(n_simulations=5)

    def test_alpha_out_of_range_high(self):
        """alpha >= 1 raises ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            PowerSimulation(alpha=1.5)

    def test_alpha_out_of_range_low(self):
        """alpha <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            PowerSimulation(alpha=0.0)

    def test_n_per_variant_too_low(self):
        """n_per_variant < 2 raises ValueError."""
        def dgp(rng):
            return rng.normal(0, 1, 1), rng.normal(0, 1, 1)

        with pytest.raises(ValueError, match="n_per_variant"):
            PowerSimulation(n_simulations=10, random_state=0).run(dgp, 1)

    def test_baseline_out_of_range(self):
        """baseline outside (0,1) raises ValueError."""
        with pytest.raises(ValueError, match="baseline"):
            PowerSimulation.for_proportion(0.0, 0.02, 100, n_simulations=10)

    def test_treatment_rate_out_of_range(self):
        """baseline + mde outside (0,1) raises ValueError."""
        with pytest.raises(ValueError, match="baseline \\+ mde"):
            PowerSimulation.for_proportion(0.90, 0.20, 100, n_simulations=10)


# ═══════════════════════════════════════════════════════════════════════
# Reproducibility
# ═══════════════════════════════════════════════════════════════════════


class TestReproducibility:
    """Same random_state → same results."""

    def test_same_seed_same_result(self):
        """Two runs with the same seed should produce identical power."""
        r1 = PowerSimulation.for_mean(
            10.0, 2.0, 0.5, 200,
            n_simulations=100, random_state=42,
        )
        r2 = PowerSimulation.for_mean(
            10.0, 2.0, 0.5, 200,
            n_simulations=100, random_state=42,
        )
        assert r1.power == r2.power
        assert r1.mean_effect == r2.mean_effect
        assert r1.mean_pvalue == r2.mean_pvalue

    def test_different_seed_different_result(self):
        """Different seeds should (almost certainly) give different power."""
        r1 = PowerSimulation.for_mean(
            10.0, 2.0, 0.5, 200,
            n_simulations=200, random_state=1,
        )
        r2 = PowerSimulation.for_mean(
            10.0, 2.0, 0.5, 200,
            n_simulations=200, random_state=2,
        )
        # Not strictly guaranteed, but extremely unlikely to be identical
        assert r1.power != r2.power or r1.mean_pvalue != r2.mean_pvalue


# ═══════════════════════════════════════════════════════════════════════
# to_dict / repr
# ═══════════════════════════════════════════════════════════════════════


class TestSerialization:
    """Result serialization."""

    def test_to_dict(self):
        """to_dict returns a plain dict with all fields."""
        result = PowerSimulation.for_mean(
            10.0, 2.0, 1.0, 100,
            n_simulations=50, random_state=0,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "power" in d
        assert "ci_power_lower" in d
        assert "ci_power_upper" in d

    def test_repr(self):
        """repr produces a readable string."""
        result = PowerSimulation.for_mean(
            10.0, 2.0, 1.0, 100,
            n_simulations=50, random_state=0,
        )
        s = repr(result)
        assert "PowerSimulationResult" in s
        assert "power" in s
