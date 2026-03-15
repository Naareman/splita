"""Tests for MediationAnalysis (causal mediation)."""

from __future__ import annotations

import numpy as np
import pytest

from splita.causal.mediation import MediationAnalysis


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestMediationBasic:
    """Basic functionality tests."""

    def test_known_mediation(self, rng):
        """Should recover a known mediation structure."""
        n = 500
        t = rng.binomial(1, 0.5, n).astype(float)
        m = 0.8 * t + rng.normal(0, 0.5, n)
        y = 0.3 * t + 0.6 * m + rng.normal(0, 0.5, n)

        r = MediationAnalysis().fit(y, t, m).result()

        # Indirect effect = a * b ~ 0.8 * 0.6 = 0.48
        assert abs(r.indirect_effect - 0.48) < 0.2
        assert r.a_path > 0
        assert r.b_path > 0
        assert r.proportion_mediated > 0.3

    def test_no_mediation(self, rng):
        """When mediator is unrelated, indirect effect should be small."""
        n = 500
        t = rng.binomial(1, 0.5, n).astype(float)
        m = rng.normal(0, 1, n)  # unrelated to treatment
        y = 1.0 * t + rng.normal(0, 1, n)

        r = MediationAnalysis().fit(y, t, m).result()

        assert abs(r.indirect_effect) < 0.3
        assert abs(r.total_effect - 1.0) < 0.3

    def test_full_mediation(self, rng):
        """Full mediation: all effect goes through mediator."""
        n = 500
        t = rng.binomial(1, 0.5, n).astype(float)
        m = 1.0 * t + rng.normal(0, 0.3, n)
        y = 0.0 * t + 1.0 * m + rng.normal(0, 0.3, n)

        r = MediationAnalysis().fit(y, t, m).result()

        assert r.proportion_mediated > 0.6
        assert abs(r.direct_effect) < 0.3

    def test_decomposition_sums(self, rng):
        """direct + indirect should approximately equal total."""
        n = 300
        t = rng.binomial(1, 0.5, n).astype(float)
        m = 0.5 * t + rng.normal(0, 1, n)
        y = 0.4 * t + 0.3 * m + rng.normal(0, 1, n)

        r = MediationAnalysis().fit(y, t, m).result()

        assert abs(r.total_effect - (r.direct_effect + r.indirect_effect)) < 0.1

    def test_with_covariates(self, rng):
        """Should handle covariates correctly."""
        n = 300
        x = rng.normal(0, 1, n)
        t = rng.binomial(1, 0.5, n).astype(float)
        m = 0.5 * t + 0.3 * x + rng.normal(0, 1, n)
        y = 0.4 * t + 0.6 * m + 0.2 * x + rng.normal(0, 1, n)

        r = MediationAnalysis().fit(y, t, m, covariates=x).result()

        assert r.indirect_effect > 0
        assert r.n == n

    def test_sobel_test_significant(self, rng):
        """Strong mediation should yield significant Sobel test."""
        n = 500
        t = rng.binomial(1, 0.5, n).astype(float)
        m = 1.0 * t + rng.normal(0, 0.5, n)
        y = 0.5 * t + 1.0 * m + rng.normal(0, 0.5, n)

        r = MediationAnalysis().fit(y, t, m).result()

        assert r.acme_pvalue < 0.05

    def test_acme_ci_contains_effect(self, rng):
        """ACME CI should contain the indirect effect."""
        n = 300
        t = rng.binomial(1, 0.5, n).astype(float)
        m = 0.6 * t + rng.normal(0, 1, n)
        y = 0.3 * t + 0.5 * m + rng.normal(0, 1, n)

        r = MediationAnalysis().fit(y, t, m).result()

        assert r.acme_ci[0] <= r.indirect_effect <= r.acme_ci[1]


class TestMediationValidation:
    """Validation and error handling tests."""

    def test_mismatched_lengths(self, rng):
        """Arrays must have the same length."""
        with pytest.raises(ValueError, match="same length"):
            MediationAnalysis().fit(
                rng.normal(0, 1, 10),
                rng.binomial(1, 0.5, 8).astype(float),
                rng.normal(0, 1, 10),
            )

    def test_too_few_observations(self, rng):
        """Should reject arrays with fewer than 5 elements."""
        with pytest.raises(ValueError, match="at least 5"):
            MediationAnalysis().fit(
                [1.0, 2.0, 3.0],
                [0.0, 1.0, 0.0],
                [0.5, 1.5, 0.3],
            )

    def test_result_before_fit(self):
        """Calling result() before fit() should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="fitted"):
            MediationAnalysis().result()

    def test_invalid_alpha(self):
        """Alpha outside (0, 1) should raise ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            MediationAnalysis(alpha=1.5)

    def test_covariates_wrong_rows(self, rng):
        """Covariates with wrong number of rows should raise ValueError."""
        n = 20
        with pytest.raises(ValueError, match="same number of rows"):
            MediationAnalysis().fit(
                rng.normal(0, 1, n),
                rng.binomial(1, 0.5, n).astype(float),
                rng.normal(0, 1, n),
                covariates=rng.normal(0, 1, (n + 5, 2)),
            )

    def test_to_dict(self, rng):
        """to_dict should return a plain dictionary."""
        n = 50
        t = rng.binomial(1, 0.5, n).astype(float)
        m = 0.5 * t + rng.normal(0, 1, n)
        y = 0.3 * t + 0.6 * m + rng.normal(0, 1, n)

        d = MediationAnalysis().fit(y, t, m).result().to_dict()
        assert isinstance(d, dict)
        assert "total_effect" in d
        assert "indirect_effect" in d

    def test_repr(self, rng):
        """repr should return a formatted string."""
        n = 50
        t = rng.binomial(1, 0.5, n).astype(float)
        r = MediationAnalysis().fit(
            rng.normal(0, 1, n), t, rng.normal(0, 1, n)
        ).result()
        assert "MediationResult" in repr(r)

    def test_chaining(self, rng):
        """fit() should return self for chaining."""
        ma = MediationAnalysis()
        n = 20
        t = rng.binomial(1, 0.5, n).astype(float)
        ret = ma.fit(rng.normal(0, 1, n), t, rng.normal(0, 1, n))
        assert ret is ma
