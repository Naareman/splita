"""Tests for EffectTransport (transportability)."""

from __future__ import annotations

import numpy as np
import pytest

from splita.causal.transportability import EffectTransport


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestTransportBasic:
    """Basic functionality tests."""

    def test_similar_populations(self, rng):
        """When populations are similar, transported ATE ~ naive ATE."""
        n = 300
        X_exp = rng.normal(0, 1, (n, 2))
        A = rng.binomial(1, 0.5, n)
        Y = 2.0 * A + X_exp[:, 0] + rng.normal(0, 1, n)
        X_tgt = rng.normal(0, 1, (500, 2))  # same distribution

        r = EffectTransport().transport(Y, A, X_exp, X_tgt)
        assert abs(r.transported_ate - 2.0) < 2.0

    def test_different_populations(self, rng):
        """When populations differ, transport should adjust."""
        n = 300
        X_exp = rng.normal(0, 1, (n, 2))
        A = rng.binomial(1, 0.5, n)
        Y = 2.0 * A + X_exp[:, 0] + rng.normal(0, 1, n)
        X_tgt = rng.normal(2, 1, (500, 2))  # shifted

        r = EffectTransport().transport(Y, A, X_exp, X_tgt)
        assert isinstance(r.transported_ate, float)

    def test_ci_order(self, rng):
        n = 200
        X_exp = rng.normal(0, 1, (n, 2))
        A = rng.binomial(1, 0.5, n)
        Y = A + rng.normal(0, 1, n)
        X_tgt = rng.normal(0, 1, (300, 2))

        r = EffectTransport().transport(Y, A, X_exp, X_tgt)
        assert r.ci_lower <= r.ci_upper

    def test_weight_diagnostics(self, rng):
        n = 200
        X_exp = rng.normal(0, 1, (n, 2))
        A = rng.binomial(1, 0.5, n)
        Y = A + rng.normal(0, 1, n)
        X_tgt = rng.normal(0, 1, (300, 2))

        r = EffectTransport().transport(Y, A, X_exp, X_tgt)
        assert "max_weight" in r.weight_diagnostics
        assert "mean_weight" in r.weight_diagnostics
        assert "effective_n" in r.weight_diagnostics

    def test_1d_covariates(self, rng):
        """Should work with 1-D covariates."""
        n = 200
        X_exp = rng.normal(0, 1, n)
        A = rng.binomial(1, 0.5, n)
        Y = 2.0 * A + rng.normal(0, 1, n)
        X_tgt = rng.normal(0, 1, 300)

        r = EffectTransport().transport(Y, A, X_exp, X_tgt)
        assert isinstance(r.transported_ate, float)

    def test_to_dict(self, rng):
        n = 100
        X_exp = rng.normal(0, 1, (n, 2))
        A = rng.binomial(1, 0.5, n)
        Y = A + rng.normal(0, 1, n)
        X_tgt = rng.normal(0, 1, (200, 2))

        r = EffectTransport().transport(Y, A, X_exp, X_tgt)
        d = r.to_dict()
        assert isinstance(d, dict)
        assert "transported_ate" in d

    def test_repr(self, rng):
        n = 100
        X_exp = rng.normal(0, 1, (n, 2))
        A = rng.binomial(1, 0.5, n)
        Y = A + rng.normal(0, 1, n)
        X_tgt = rng.normal(0, 1, (200, 2))

        r = EffectTransport().transport(Y, A, X_exp, X_tgt)
        assert "TransportResult" in repr(r)


class TestTransportValidation:
    """Validation and error tests."""

    def test_alpha_out_of_range(self):
        with pytest.raises(ValueError, match="alpha"):
            EffectTransport(alpha=1.5)

    def test_mismatched_covariates(self, rng):
        with pytest.raises(ValueError, match="same number of columns"):
            EffectTransport().transport(
                rng.normal(0, 1, 100),
                rng.binomial(1, 0.5, 100),
                rng.normal(0, 1, (100, 3)),
                rng.normal(0, 1, (200, 2)),
            )

    def test_mismatched_outcome_covariates(self, rng):
        with pytest.raises(ValueError, match="same number of rows"):
            EffectTransport().transport(
                rng.normal(0, 1, 50),
                rng.binomial(1, 0.5, 50),
                rng.normal(0, 1, (100, 2)),  # wrong rows
                rng.normal(0, 1, (200, 2)),
            )

    def test_all_treated(self, rng):
        """Should raise if no control units exist."""
        n = 100
        with pytest.raises(ValueError, match="at least 1"):
            EffectTransport().transport(
                rng.normal(0, 1, n),
                np.ones(n),  # all treated
                rng.normal(0, 1, (n, 2)),
                rng.normal(0, 1, (200, 2)),
            )
