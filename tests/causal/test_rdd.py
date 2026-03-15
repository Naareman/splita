"""Tests for RegressionDiscontinuity (RDD)."""

from __future__ import annotations

import numpy as np
import pytest

from splita.causal.rdd import RegressionDiscontinuity


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestRDDBasic:
    """Basic functionality tests."""

    def test_known_effect(self, rng):
        """RDD should recover a known jump at the cutoff."""
        n = 500
        x = rng.uniform(-2, 2, n)
        y = 3.0 * (x >= 0) + 0.5 * x + rng.normal(0, 0.5, n)

        r = RegressionDiscontinuity().fit(y, x, cutoff=0.0)
        assert abs(r.late - 3.0) < 1.5
        assert r.pvalue < 0.05

    def test_no_effect(self, rng):
        """No discontinuity should yield non-significant result."""
        n = 400
        x = rng.uniform(-2, 2, n)
        y = 0.5 * x + rng.normal(0, 1, n)

        r = RegressionDiscontinuity().fit(y, x, cutoff=0.0)
        assert abs(r.late) < 2.0
        assert r.pvalue > 0.01

    def test_nonzero_cutoff(self, rng):
        """RDD with a non-zero cutoff."""
        n = 500
        x = rng.uniform(-1, 3, n)
        y = 2.0 * (x >= 1.0) + 0.3 * x + rng.normal(0, 0.5, n)

        r = RegressionDiscontinuity().fit(y, x, cutoff=1.0)
        assert abs(r.late - 2.0) < 1.5

    def test_manual_bandwidth(self, rng):
        """Specifying bandwidth explicitly."""
        n = 400
        x = rng.uniform(-3, 3, n)
        y = 4.0 * (x >= 0) + rng.normal(0, 0.5, n)

        r = RegressionDiscontinuity().fit(y, x, cutoff=0.0, bandwidth=1.5)
        assert r.bandwidth_used == 1.5
        assert abs(r.late - 4.0) < 2.0

    def test_ci_contains_true_effect(self, rng):
        """CI should contain the true effect."""
        n = 600
        x = rng.uniform(-2, 2, n)
        true_effect = 2.5
        y = true_effect * (x >= 0) + rng.normal(0, 0.5, n)

        r = RegressionDiscontinuity().fit(y, x, cutoff=0.0)
        assert r.ci_lower < true_effect < r.ci_upper

    def test_n_left_n_right(self, rng):
        """n_left and n_right should be positive."""
        n = 300
        x = rng.uniform(-2, 2, n)
        y = rng.normal(0, 1, n)

        r = RegressionDiscontinuity().fit(y, x, cutoff=0.0)
        assert r.n_left > 0
        assert r.n_right > 0

    def test_to_dict(self, rng):
        """Result should be serialisable."""
        x = rng.uniform(-2, 2, 200)
        y = rng.normal(0, 1, 200)
        r = RegressionDiscontinuity().fit(y, x, cutoff=0.0)
        d = r.to_dict()
        assert isinstance(d, dict)
        assert "late" in d
        assert "bandwidth_used" in d

    def test_repr(self, rng):
        """repr should be a string."""
        x = rng.uniform(-2, 2, 200)
        y = rng.normal(0, 1, 200)
        r = RegressionDiscontinuity().fit(y, x, cutoff=0.0)
        assert "RDDResult" in repr(r)


class TestRDDValidation:
    """Validation and error tests."""

    def test_alpha_out_of_range(self):
        with pytest.raises(ValueError, match="alpha"):
            RegressionDiscontinuity(alpha=1.5)

    def test_alpha_zero(self):
        with pytest.raises(ValueError, match="alpha"):
            RegressionDiscontinuity(alpha=0.0)

    def test_too_few_observations(self, rng):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # All above cutoff, so n_left = 0
        with pytest.raises(ValueError, match="Too few"):
            RegressionDiscontinuity().fit(y, x, cutoff=0.0)

    def test_negative_bandwidth(self, rng):
        x = rng.uniform(-2, 2, 100)
        y = rng.normal(0, 1, 100)
        with pytest.raises(ValueError, match="bandwidth"):
            RegressionDiscontinuity().fit(y, x, cutoff=0.0, bandwidth=-1.0)

    def test_mismatched_lengths(self, rng):
        x = rng.uniform(-2, 2, 100)
        y = rng.normal(0, 1, 50)
        with pytest.raises(ValueError, match="same length"):
            RegressionDiscontinuity().fit(y, x, cutoff=0.0)

    def test_non_array_input(self):
        with pytest.raises(TypeError):
            RegressionDiscontinuity().fit("not_array", [1, 2, 3, 4], cutoff=0.0)
