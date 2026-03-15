"""Tests for InstrumentalVariables (2SLS estimation)."""

from __future__ import annotations

import numpy as np
import pytest

from splita.causal.instrumental_variables import InstrumentalVariables


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestIVBasic:
    """Basic functionality tests."""

    def test_known_late_recovery(self, rng):
        """2SLS should recover the true LATE."""
        n = 2000
        true_late = 3.0
        z = rng.binomial(1, 0.5, n).astype(float)
        # Compliers: treatment follows instrument with noise
        u = rng.normal(0, 0.5, n)
        t = (z * 0.7 + u > 0.2).astype(float)
        y = true_late * t + rng.normal(0, 1, n)

        r = InstrumentalVariables().fit(y, t, z)
        assert abs(r.late - true_late) < 2.0
        assert r.significant is True

    def test_no_effect(self, rng):
        """Zero treatment effect should yield non-significant LATE."""
        n = 500
        z = rng.binomial(1, 0.5, n).astype(float)
        t = (z + rng.normal(0, 0.3, n) > 0.5).astype(float)
        y = rng.normal(0, 1, n)  # no effect

        r = InstrumentalVariables().fit(y, t, z)
        assert abs(r.late) < 2.0

    def test_weak_instrument_warning(self, rng):
        """A weak instrument (low F) should be flagged."""
        n = 500
        z = rng.binomial(1, 0.5, n).astype(float)
        # Treatment barely related to instrument
        t = rng.binomial(1, 0.5, n).astype(float)
        y = 2.0 * t + rng.normal(0, 1, n)

        r = InstrumentalVariables().fit(y, t, z)
        assert r.weak_instrument is True
        assert r.first_stage_f < 10.0

    def test_strong_instrument(self, rng):
        """A strong instrument should have high F and not be flagged."""
        n = 1000
        z = rng.binomial(1, 0.5, n).astype(float)
        t = z.copy()  # perfect compliance
        y = 2.0 * t + rng.normal(0, 1, n)

        r = InstrumentalVariables().fit(y, t, z)
        assert r.weak_instrument is False
        assert r.first_stage_f > 10.0

    def test_with_covariates(self, rng):
        """2SLS with covariates should still recover the LATE."""
        n = 1000
        true_late = 2.0
        x = rng.normal(0, 1, (n, 2))
        z = rng.binomial(1, 0.5, n).astype(float)
        t = (z * 0.8 + x[:, 0] * 0.3 + rng.normal(0, 0.3, n) > 0.3).astype(float)
        y = true_late * t + x[:, 0] + x[:, 1] * 0.5 + rng.normal(0, 1, n)

        r = InstrumentalVariables().fit(y, t, z, covariates=x)
        assert abs(r.late - true_late) < 2.0

    def test_ci_contains_estimate(self, rng):
        """CI should contain the point estimate."""
        n = 1000
        z = rng.binomial(1, 0.5, n).astype(float)
        t = (z + rng.normal(0, 0.3, n) > 0.5).astype(float)
        y = 2.0 * t + rng.normal(0, 1, n)

        r = InstrumentalVariables().fit(y, t, z)
        assert r.ci_lower <= r.late <= r.ci_upper

    def test_se_positive(self, rng):
        """Standard error should be positive."""
        n = 500
        z = rng.binomial(1, 0.5, n).astype(float)
        t = (z + rng.normal(0, 0.3, n) > 0.5).astype(float)
        y = 2.0 * t + rng.normal(0, 1, n)

        r = InstrumentalVariables().fit(y, t, z)
        assert r.se > 0

    def test_pvalue_range(self, rng):
        """p-value should be in [0, 1]."""
        n = 500
        z = rng.binomial(1, 0.5, n).astype(float)
        t = (z + rng.normal(0, 0.3, n) > 0.5).astype(float)
        y = 2.0 * t + rng.normal(0, 1, n)

        r = InstrumentalVariables().fit(y, t, z)
        assert 0.0 <= r.pvalue <= 1.0

    def test_significant_flag(self, rng):
        """significant should match pvalue < alpha."""
        n = 1000
        z = rng.binomial(1, 0.5, n).astype(float)
        t = z.copy()
        y = 3.0 * t + rng.normal(0, 1, n)

        alpha = 0.05
        r = InstrumentalVariables(alpha=alpha).fit(y, t, z)
        assert r.significant == (r.pvalue < alpha)

    def test_negative_effect(self, rng):
        """Negative treatment effect should yield negative LATE."""
        n = 1000
        z = rng.binomial(1, 0.5, n).astype(float)
        t = z.copy()
        y = -2.0 * t + rng.normal(0, 1, n)

        r = InstrumentalVariables().fit(y, t, z)
        assert r.late < 0

    def test_1d_covariates(self, rng):
        """1-D covariate array should be reshaped automatically."""
        n = 500
        z = rng.binomial(1, 0.5, n).astype(float)
        t = (z + rng.normal(0, 0.3, n) > 0.5).astype(float)
        x = rng.normal(0, 1, n)
        y = 2.0 * t + x + rng.normal(0, 1, n)

        r = InstrumentalVariables().fit(y, t, z, covariates=x)
        assert np.isfinite(r.late)

    def test_custom_alpha(self, rng):
        """Custom alpha should affect significance threshold."""
        n = 500
        z = rng.binomial(1, 0.5, n).astype(float)
        t = (z + rng.normal(0, 0.3, n) > 0.5).astype(float)
        y = 0.5 * t + rng.normal(0, 1, n)

        r1 = InstrumentalVariables(alpha=0.05).fit(y, t, z)
        r2 = InstrumentalVariables(alpha=0.001).fit(y, t, z)

        # Same p-value
        assert abs(r1.pvalue - r2.pvalue) < 0.001


class TestIVValidation:
    """Input validation tests."""

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            InstrumentalVariables(alpha=1.5)

    def test_too_few_samples(self, rng):
        with pytest.raises(ValueError, match="at least"):
            InstrumentalVariables().fit([1, 2, 3], [1, 0, 1], [0, 1, 0])

    def test_mismatched_lengths(self, rng):
        n = 100
        y = rng.normal(0, 1, n)
        t = rng.binomial(1, 0.5, n + 10).astype(float)
        z = rng.binomial(1, 0.5, n).astype(float)

        with pytest.raises(ValueError, match="same length"):
            InstrumentalVariables().fit(y, t, z)

    def test_non_array_input(self):
        with pytest.raises(TypeError, match="array-like"):
            InstrumentalVariables().fit("not_array", [1, 2, 3, 4, 5], [0, 1, 0, 1, 0])


class TestIVResult:
    """Result object tests."""

    def test_to_dict(self, rng):
        n = 500
        z = rng.binomial(1, 0.5, n).astype(float)
        t = z.copy()
        y = 2.0 * t + rng.normal(0, 1, n)

        r = InstrumentalVariables().fit(y, t, z)
        d = r.to_dict()
        assert isinstance(d, dict)
        assert "late" in d
        assert "first_stage_f" in d
        assert "weak_instrument" in d

    def test_repr(self, rng):
        n = 500
        z = rng.binomial(1, 0.5, n).astype(float)
        t = z.copy()
        y = 2.0 * t + rng.normal(0, 1, n)

        r = InstrumentalVariables().fit(y, t, z)
        assert "IVResult" in repr(r)
