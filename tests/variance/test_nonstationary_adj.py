"""Tests for NonstationaryAdjustment."""

from __future__ import annotations

import numpy as np
import pytest

from splita.variance.nonstationary import NonstationaryAdjustment


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestNonstationaryBasic:
    """Basic functionality tests."""

    def test_known_effect_with_trend(self, rng):
        """Should recover effect despite a linear trend."""
        n = 100
        ts = np.arange(n, dtype=float)
        trend = 0.05 * ts
        control = 10 + trend + rng.normal(0, 1, n)
        treatment = 12 + trend + rng.normal(0, 1, n)

        r = NonstationaryAdjustment().fit_transform(control, treatment, ts)
        assert abs(r.ate_corrected - 2.0) < 1.0

    def test_no_trend(self, rng):
        """Without a trend, corrected and naive should be similar."""
        n = 100
        ts = np.arange(n, dtype=float)
        control = rng.normal(10, 1, n)
        treatment = rng.normal(12, 1, n)

        r = NonstationaryAdjustment().fit_transform(control, treatment, ts)
        assert abs(r.ate_corrected - r.ate_naive) < 0.5

    def test_bias_is_difference(self, rng):
        """Bias should equal naive - corrected."""
        n = 50
        ts = np.arange(n, dtype=float)
        control = 10 + 0.1 * ts + rng.normal(0, 1, n)
        treatment = 12 + 0.1 * ts + rng.normal(0, 1, n)

        r = NonstationaryAdjustment().fit_transform(control, treatment, ts)
        assert abs(r.bias - (r.ate_naive - r.ate_corrected)) < 1e-10

    def test_ci_contains_effect(self, rng):
        """CI should contain the true effect."""
        n = 200
        ts = np.arange(n, dtype=float)
        control = 10 + rng.normal(0, 0.5, n)
        treatment = 13 + rng.normal(0, 0.5, n)

        r = NonstationaryAdjustment().fit_transform(control, treatment, ts)
        assert r.ci_lower < 3.0 < r.ci_upper

    def test_significant_effect(self, rng):
        """Large effect should be significant."""
        n = 100
        ts = np.arange(n, dtype=float)
        control = rng.normal(10, 1, n)
        treatment = rng.normal(15, 1, n)

        r = NonstationaryAdjustment().fit_transform(control, treatment, ts)
        assert r.pvalue < 0.05

    def test_no_effect_not_significant(self, rng):
        """No effect should not be significant."""
        n = 50
        ts = np.arange(n, dtype=float)
        control = rng.normal(10, 1, n)
        treatment = rng.normal(10, 1, n)

        r = NonstationaryAdjustment().fit_transform(control, treatment, ts)
        assert r.pvalue > 0.01

    def test_to_dict(self, rng):
        n = 30
        ts = np.arange(n, dtype=float)
        r = NonstationaryAdjustment().fit_transform(
            rng.normal(0, 1, n), rng.normal(1, 1, n), ts
        )
        d = r.to_dict()
        assert isinstance(d, dict)
        assert "ate_corrected" in d

    def test_repr(self, rng):
        n = 30
        ts = np.arange(n, dtype=float)
        r = NonstationaryAdjustment().fit_transform(
            rng.normal(0, 1, n), rng.normal(1, 1, n), ts
        )
        assert "NonstationaryAdjResult" in repr(r)


class TestNonstationaryValidation:
    """Validation and error tests."""

    def test_alpha_out_of_range(self):
        with pytest.raises(ValueError, match="alpha"):
            NonstationaryAdjustment(alpha=0.0)

    def test_too_few_observations(self, rng):
        with pytest.raises(ValueError, match="at least"):
            NonstationaryAdjustment().fit_transform([1.0, 2.0], [3.0, 4.0], [0.0, 1.0])

    def test_mismatched_lengths(self, rng):
        with pytest.raises(ValueError, match="same length"):
            NonstationaryAdjustment().fit_transform(
                [1.0, 2.0, 3.0, 4.0], [4.0, 5.0, 6.0], [0.0, 1.0, 2.0, 3.0]
            )

    def test_non_array_input(self):
        with pytest.raises(TypeError):
            NonstationaryAdjustment().fit_transform("bad", [1, 2, 3], [0, 1, 2])
