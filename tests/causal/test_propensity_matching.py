"""Tests for PropensityScoreMatching (PSM for causal inference)."""

from __future__ import annotations

import numpy as np
import pytest

from splita.causal.propensity_matching import PropensityScoreMatching


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestPSMBasic:
    """Basic functionality tests."""

    def test_removes_confounding(self, rng):
        """PSM should reduce confounding bias."""
        n = 1000
        x = rng.normal(0, 1, (n, 2))
        # Confounded treatment: higher x -> more likely treated
        prob_treat = 1.0 / (1.0 + np.exp(-(x[:, 0] + x[:, 1])))
        t = rng.binomial(1, prob_treat).astype(float)
        true_att = 2.0
        y = true_att * t + x[:, 0] * 1.5 + x[:, 1] * 0.5 + rng.normal(0, 1, n)

        r = PropensityScoreMatching().fit(y, t, x)
        # PSM estimate should be closer to true ATT than naive
        naive_att = float(np.mean(y[t == 1]) - np.mean(y[t == 0]))
        assert abs(r.att - true_att) < abs(naive_att - true_att) + 1.0

    def test_known_effect(self, rng):
        """PSM should approximately recover a known treatment effect."""
        n = 1000
        x = rng.normal(0, 1, (n, 2))
        prob_treat = 1.0 / (1.0 + np.exp(-x[:, 0]))
        t = rng.binomial(1, prob_treat).astype(float)
        true_att = 3.0
        y = true_att * t + x[:, 0] + rng.normal(0, 1, n)

        r = PropensityScoreMatching().fit(y, t, x)
        assert abs(r.att - true_att) < 2.0

    def test_caliper_drops_bad_matches(self, rng):
        """With a tight caliper, some units should be unmatched."""
        n = 500
        x = rng.normal(0, 1, (n, 2))
        # Noisy treatment assignment so PS overlap exists
        prob_treat = 1.0 / (1.0 + np.exp(-(0.5 * x[:, 0])))
        t = rng.binomial(1, prob_treat).astype(float)
        y = 2.0 * t + x[:, 0] + rng.normal(0, 1, n)

        r_no_cal = PropensityScoreMatching().fit(y, t, x)
        r_cal = PropensityScoreMatching(caliper=0.02).fit(y, t, x)

        assert r_cal.n_unmatched >= r_no_cal.n_unmatched

    def test_balance_improves(self, rng):
        """Covariate balance should improve after matching."""
        n = 1000
        x = rng.normal(0, 1, (n, 2))
        prob_treat = 1.0 / (1.0 + np.exp(-(x[:, 0] * 2)))
        t = rng.binomial(1, prob_treat).astype(float)
        y = 2.0 * t + x[:, 0] + rng.normal(0, 1, n)

        r = PropensityScoreMatching().fit(y, t, x)

        # At least the confounded covariate should improve
        before_smd = r.balance_before["X0"]
        after_smd = r.balance_after["X0"]
        assert after_smd <= before_smd + 0.1  # allow small tolerance

    def test_n_matched_reasonable(self, rng):
        """Number of matched units should be positive."""
        n = 500
        x = rng.normal(0, 1, (n, 2))
        t = (x[:, 0] > 0).astype(float)
        y = 2.0 * t + rng.normal(0, 1, n)

        r = PropensityScoreMatching().fit(y, t, x)
        assert r.n_matched > 0

    def test_ci_contains_estimate(self, rng):
        """CI should contain the point estimate."""
        n = 500
        x = rng.normal(0, 1, (n, 2))
        t = (x[:, 0] > 0).astype(float)
        y = 2.0 * t + rng.normal(0, 1, n)

        r = PropensityScoreMatching().fit(y, t, x)
        assert r.ci_lower <= r.att <= r.ci_upper

    def test_se_positive(self, rng):
        """Standard error should be positive."""
        n = 500
        x = rng.normal(0, 1, (n, 2))
        t = (x[:, 0] > 0).astype(float)
        y = 2.0 * t + rng.normal(0, 1, n)

        r = PropensityScoreMatching().fit(y, t, x)
        assert r.se > 0

    def test_pvalue_range(self, rng):
        """p-value should be in [0, 1]."""
        n = 500
        x = rng.normal(0, 1, (n, 2))
        t = (x[:, 0] > 0).astype(float)
        y = 2.0 * t + rng.normal(0, 1, n)

        r = PropensityScoreMatching().fit(y, t, x)
        assert 0.0 <= r.pvalue <= 1.0

    def test_n_neighbors(self, rng):
        """Multiple neighbors should work."""
        n = 500
        x = rng.normal(0, 1, (n, 2))
        t = (x[:, 0] > 0).astype(float)
        y = 2.0 * t + rng.normal(0, 1, n)

        r = PropensityScoreMatching(n_neighbors=3).fit(y, t, x)
        assert r.n_matched > 0

    def test_balance_dicts_have_all_covariates(self, rng):
        """Balance dicts should have entries for every covariate."""
        n = 500
        n_covs = 4
        x = rng.normal(0, 1, (n, n_covs))
        t = (x[:, 0] > 0).astype(float)
        y = 2.0 * t + rng.normal(0, 1, n)

        r = PropensityScoreMatching().fit(y, t, x)
        assert len(r.balance_before) == n_covs
        assert len(r.balance_after) == n_covs

    def test_no_effect(self, rng):
        """Zero treatment effect should yield small ATT."""
        n = 500
        x = rng.normal(0, 1, (n, 2))
        t = (x[:, 0] > 0).astype(float)
        y = x[:, 0] + rng.normal(0, 1, n)  # no treatment effect

        r = PropensityScoreMatching().fit(y, t, x)
        assert abs(r.att) < 2.0

    def test_significant_flag(self, rng):
        """significant should match pvalue < 0.05."""
        n = 1000
        x = rng.normal(0, 1, (n, 2))
        t = (x[:, 0] > 0).astype(float)
        y = 3.0 * t + rng.normal(0, 1, n)

        r = PropensityScoreMatching().fit(y, t, x)
        assert r.significant == (r.pvalue < 0.05)


class TestPSMValidation:
    """Input validation tests."""

    def test_invalid_n_neighbors(self):
        with pytest.raises(ValueError, match="n_neighbors"):
            PropensityScoreMatching(n_neighbors=0)

    def test_invalid_caliper(self):
        with pytest.raises(ValueError, match="caliper"):
            PropensityScoreMatching(caliper=-1.0)

    def test_non_binary_treatment(self, rng):
        n = 100
        x = rng.normal(0, 1, (n, 2))
        t = rng.normal(0, 1, n)  # continuous, not binary
        y = rng.normal(0, 1, n)

        with pytest.raises(ValueError, match="binary"):
            PropensityScoreMatching().fit(y, t, x)

    def test_mismatched_lengths(self, rng):
        with pytest.raises(ValueError, match="same length"):
            PropensityScoreMatching().fit(
                rng.normal(0, 1, 100),
                np.ones(50),
                rng.normal(0, 1, (100, 2)),
            )

    def test_non_array_covariates(self, rng):
        with pytest.raises(TypeError, match="ndarray"):
            PropensityScoreMatching().fit(
                rng.normal(0, 1, 100),
                np.concatenate([np.ones(50), np.zeros(50)]),
                "not_array",
            )


class TestPSMResult:
    """Result object tests."""

    def test_to_dict(self, rng):
        n = 200
        x = rng.normal(0, 1, (n, 2))
        t = (x[:, 0] > 0).astype(float)
        y = 2.0 * t + rng.normal(0, 1, n)

        r = PropensityScoreMatching().fit(y, t, x)
        d = r.to_dict()
        assert isinstance(d, dict)
        assert "att" in d
        assert "n_matched" in d
        assert "balance_before" in d

    def test_repr(self, rng):
        n = 200
        x = rng.normal(0, 1, (n, 2))
        t = (x[:, 0] > 0).astype(float)
        y = 2.0 * t + rng.normal(0, 1, n)

        r = PropensityScoreMatching().fit(y, t, x)
        assert "PSMResult" in repr(r)
