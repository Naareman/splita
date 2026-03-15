"""Tests for MixedEffectsExperiment (repeated measures)."""

from __future__ import annotations

import numpy as np
import pytest

from splita.core.mixed_effects import MixedEffectsExperiment


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _make_repeated_data(rng, n_users=100, obs_per_user=5, effect=0.5, icc_target=0.3):
    """Helper to generate repeated-measures data."""
    user_ids = np.repeat(np.arange(n_users), obs_per_user)
    treatment = np.repeat(
        rng.binomial(1, 0.5, n_users), obs_per_user
    ).astype(float)
    user_effects = rng.normal(0, np.sqrt(icc_target), n_users)
    noise = rng.normal(0, np.sqrt(1 - icc_target), n_users * obs_per_user)
    outcome = np.repeat(user_effects, obs_per_user) + effect * treatment + noise
    return outcome, treatment, user_ids


class TestMixedEffectsBasic:
    """Basic functionality tests."""

    def test_known_effect(self, rng):
        """Should recover a known treatment effect."""
        outcome, treatment, user_ids = _make_repeated_data(
            rng, n_users=200, obs_per_user=5, effect=1.0
        )

        r = MixedEffectsExperiment().fit(outcome, treatment, user_ids).result()

        assert abs(r.ate - 1.0) < 0.5
        assert r.significant is True

    def test_no_effect(self, rng):
        """No treatment effect should yield non-significant result."""
        outcome, treatment, user_ids = _make_repeated_data(
            rng, n_users=100, obs_per_user=5, effect=0.0
        )

        r = MixedEffectsExperiment().fit(outcome, treatment, user_ids).result()

        assert abs(r.ate) < 0.5
        assert r.pvalue > 0.01

    def test_icc_positive(self, rng):
        """ICC should be positive when there are user-level effects."""
        outcome, treatment, user_ids = _make_repeated_data(
            rng, n_users=100, obs_per_user=10, icc_target=0.5
        )

        r = MixedEffectsExperiment().fit(outcome, treatment, user_ids).result()

        assert r.icc > 0.1

    def test_icc_near_zero(self, rng):
        """ICC should be near zero when no user-level effects."""
        n_users = 100
        obs_per = 5
        user_ids = np.repeat(np.arange(n_users), obs_per)
        treatment = np.repeat(
            rng.binomial(1, 0.5, n_users), obs_per
        ).astype(float)
        outcome = rng.normal(0, 1, n_users * obs_per)

        r = MixedEffectsExperiment().fit(outcome, treatment, user_ids).result()

        assert r.icc < 0.15

    def test_n_users_and_observations(self, rng):
        """Should correctly report user and observation counts."""
        n_users, obs_per = 80, 4
        outcome, treatment, user_ids = _make_repeated_data(
            rng, n_users=n_users, obs_per_user=obs_per
        )

        r = MixedEffectsExperiment().fit(outcome, treatment, user_ids).result()

        assert r.n_users == n_users
        assert r.n_observations == n_users * obs_per

    def test_ci_contains_ate(self, rng):
        """CI should contain the ATE estimate."""
        outcome, treatment, user_ids = _make_repeated_data(rng)
        r = MixedEffectsExperiment().fit(outcome, treatment, user_ids).result()

        assert r.ci_lower <= r.ate <= r.ci_upper

    def test_unequal_observations_per_user(self, rng):
        """Should handle different numbers of observations per user."""
        user_ids = []
        for uid in range(50):
            n_obs = rng.integers(2, 10)
            user_ids.extend([uid] * n_obs)
        user_ids = np.array(user_ids, dtype=float)
        n_total = len(user_ids)

        treatment = np.zeros(n_total)
        for uid in range(50):
            mask = user_ids == uid
            treatment[mask] = float(rng.binomial(1, 0.5))

        outcome = 0.5 * treatment + rng.normal(0, 1, n_total)

        r = MixedEffectsExperiment().fit(outcome, treatment, user_ids).result()

        assert r.n_users == 50
        assert r.n_observations == n_total


class TestMixedEffectsValidation:
    """Validation and error handling tests."""

    def test_too_few_observations(self):
        """Should reject arrays with fewer than 4 elements."""
        with pytest.raises(ValueError, match="at least 4"):
            MixedEffectsExperiment().fit(
                [1.0, 2.0], [0.0, 1.0], [0.0, 1.0],
            )

    def test_single_user(self, rng):
        """Should reject data with only 1 unique user."""
        with pytest.raises(ValueError, match="at least 2"):
            MixedEffectsExperiment().fit(
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            )

    def test_mismatched_lengths(self, rng):
        """Arrays must have the same length."""
        with pytest.raises(ValueError, match="same length"):
            MixedEffectsExperiment().fit(
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
            )

    def test_result_before_fit(self):
        """Calling result() before fit() should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="fitted"):
            MixedEffectsExperiment().result()

    def test_invalid_alpha(self):
        """Alpha outside (0, 1) should raise ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            MixedEffectsExperiment(alpha=0.0)

    def test_to_dict(self, rng):
        """to_dict should return a plain dictionary."""
        outcome, treatment, user_ids = _make_repeated_data(rng, n_users=30)
        d = MixedEffectsExperiment().fit(
            outcome, treatment, user_ids
        ).result().to_dict()

        assert isinstance(d, dict)
        assert "ate" in d
        assert "icc" in d

    def test_repr(self, rng):
        """repr should return a formatted string."""
        outcome, treatment, user_ids = _make_repeated_data(rng, n_users=30)
        r = MixedEffectsExperiment().fit(outcome, treatment, user_ids).result()
        assert "MixedEffectsResult" in repr(r)

    def test_chaining(self, rng):
        """fit() should return self for chaining."""
        me = MixedEffectsExperiment()
        outcome, treatment, user_ids = _make_repeated_data(rng, n_users=30)
        ret = me.fit(outcome, treatment, user_ids)
        assert ret is me

    def test_all_treatment_one_group(self, rng):
        """Should raise when all users are in the same group."""
        n_users, obs_per = 20, 3
        user_ids = np.repeat(np.arange(n_users), obs_per)
        treatment = np.ones(n_users * obs_per)  # all treatment
        outcome = rng.normal(0, 1, n_users * obs_per)

        with pytest.raises(ValueError, match="at least 1 user"):
            MixedEffectsExperiment().fit(outcome, treatment, user_ids)
