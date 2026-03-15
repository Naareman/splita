"""Tests for TMLE."""

from __future__ import annotations

import numpy as np
import pytest

from splita.causal.tmle import TMLE


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestTMLEBasic:
    """Basic functionality tests."""

    def test_known_effect(self, rng):
        """TMLE should recover a known ATE."""
        n = 500
        X = rng.normal(0, 1, (n, 3))
        A = rng.binomial(1, 0.5, n)
        Y = 2.0 * A + X[:, 0] + rng.normal(0, 1, n)

        r = TMLE().fit(Y, A, X)
        assert abs(r.ate - 2.0) < 1.5

    def test_no_effect(self, rng):
        """No treatment effect should yield non-significant result."""
        n = 300
        X = rng.normal(0, 1, (n, 2))
        A = rng.binomial(1, 0.5, n)
        Y = X[:, 0] + rng.normal(0, 1, n)

        r = TMLE().fit(Y, A, X)
        assert abs(r.ate) < 2.0

    def test_targeted_vs_initial(self, rng):
        """Targeted estimate should differ from initial in general."""
        n = 300
        X = rng.normal(0, 1, (n, 3))
        A = rng.binomial(1, 0.5, n)
        Y = 3.0 * A + X[:, 0] + rng.normal(0, 1, n)

        r = TMLE().fit(Y, A, X)
        # Both should be close to 3.0
        assert abs(r.initial_estimate - 3.0) < 2.0
        assert abs(r.targeted_estimate - 3.0) < 2.0

    def test_ci_contains_true_effect(self, rng):
        """CI should contain the true ATE."""
        n = 500
        X = rng.normal(0, 1, (n, 2))
        A = rng.binomial(1, 0.5, n)
        Y = 1.5 * A + rng.normal(0, 1, n)

        r = TMLE().fit(Y, A, X)
        assert r.ci_lower < 1.5 < r.ci_upper

    def test_pvalue_significant(self, rng):
        """Large effect should be significant."""
        n = 300
        X = rng.normal(0, 1, (n, 2))
        A = rng.binomial(1, 0.5, n)
        Y = 5.0 * A + rng.normal(0, 1, n)

        r = TMLE().fit(Y, A, X)
        assert r.pvalue < 0.05

    def test_1d_covariates(self, rng):
        """Should work with 1-D covariates."""
        n = 200
        X = rng.normal(0, 1, n)
        A = rng.binomial(1, 0.5, n)
        Y = 2.0 * A + rng.normal(0, 1, n)

        r = TMLE().fit(Y, A, X)
        assert isinstance(r.ate, float)

    def test_to_dict(self, rng):
        n = 100
        X = rng.normal(0, 1, (n, 2))
        A = rng.binomial(1, 0.5, n)
        Y = A + rng.normal(0, 1, n)
        r = TMLE().fit(Y, A, X)
        d = r.to_dict()
        assert isinstance(d, dict)
        assert "ate" in d
        assert "initial_estimate" in d

    def test_repr(self, rng):
        n = 100
        X = rng.normal(0, 1, (n, 2))
        A = rng.binomial(1, 0.5, n)
        Y = A + rng.normal(0, 1, n)
        r = TMLE().fit(Y, A, X)
        assert "TMLEResult" in repr(r)


class TestTMLEValidation:
    """Validation and error tests."""

    def test_alpha_out_of_range(self):
        with pytest.raises(ValueError, match="alpha"):
            TMLE(alpha=0.0)

    def test_non_binary_treatment(self, rng):
        n = 50
        X = rng.normal(0, 1, (n, 2))
        A = rng.choice([0, 1, 2], n)
        Y = rng.normal(0, 1, n)
        with pytest.raises(ValueError, match="binary"):
            TMLE().fit(Y, A, X)

    def test_mismatched_covariates(self, rng):
        with pytest.raises(ValueError, match="same number of rows"):
            TMLE().fit(
                rng.normal(0, 1, 50),
                rng.binomial(1, 0.5, 50),
                rng.normal(0, 1, (100, 2)),
            )

    def test_too_few_observations(self, rng):
        with pytest.raises(ValueError, match="at least"):
            TMLE().fit(
                rng.normal(0, 1, 5),
                rng.binomial(1, 0.5, 5),
                rng.normal(0, 1, (5, 2)),
            )
