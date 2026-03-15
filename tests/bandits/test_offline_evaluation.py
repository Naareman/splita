"""Tests for OfflineEvaluator (offline policy evaluation)."""

from __future__ import annotations

import numpy as np
import pytest

from splita.bandits.offline_evaluation import OfflineEvaluator


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _make_logged_data(rng, n=500, n_actions=3):
    """Generate logged bandit data."""
    contexts = rng.normal(0, 1, (n, 2))
    actions = rng.choice(n_actions, n)
    rewards = rng.binomial(1, 0.3, n).astype(float)
    log_probs = np.full(n, 1.0 / n_actions)
    return rewards, actions, contexts, log_probs


class TestOfflineBasic:
    """Basic functionality tests."""

    def test_same_policy_ips(self, rng):
        """When target = logging policy, estimate should equal mean reward."""
        rewards, actions, contexts, log_probs = _make_logged_data(rng)
        target_probs = log_probs.copy()

        r = OfflineEvaluator(method="ips").evaluate(
            rewards, actions, contexts, log_probs, target_probs
        )

        assert abs(r.estimated_value - np.mean(rewards)) < 0.1
        assert r.method == "ips"

    def test_same_policy_dr(self, rng):
        """Doubly robust with same policy should estimate mean reward."""
        rewards, actions, contexts, log_probs = _make_logged_data(rng)
        target_probs = log_probs.copy()

        r = OfflineEvaluator(method="doubly_robust").evaluate(
            rewards, actions, contexts, log_probs, target_probs
        )

        assert abs(r.estimated_value - np.mean(rewards)) < 0.15
        assert r.method == "doubly_robust"

    def test_ess_with_uniform_policy(self, rng):
        """With uniform weights, ESS should be near n."""
        rewards, actions, contexts, log_probs = _make_logged_data(rng)
        target_probs = log_probs.copy()

        r = OfflineEvaluator().evaluate(
            rewards, actions, contexts, log_probs, target_probs
        )

        assert r.effective_sample_size > 0
        assert r.n == len(rewards)

    def test_ci_contains_estimate(self, rng):
        """CI should contain the estimated value."""
        rewards, actions, contexts, log_probs = _make_logged_data(rng)
        target_probs = log_probs.copy()

        r = OfflineEvaluator().evaluate(
            rewards, actions, contexts, log_probs, target_probs
        )

        assert r.ci_lower <= r.estimated_value <= r.ci_upper

    def test_different_policy(self, rng):
        """Different target policy should produce a different estimate."""
        n = 500
        n_actions = 3
        rewards, actions, contexts, log_probs = _make_logged_data(rng, n, n_actions)

        # Target policy that prefers action 0
        target_probs = np.where(actions == 0, 0.8, 0.1)

        r = OfflineEvaluator().evaluate(
            rewards, actions, contexts, log_probs, target_probs
        )

        assert r.estimated_value is not None
        assert r.se > 0

    def test_se_positive(self, rng):
        """Standard error should be positive."""
        rewards, actions, contexts, log_probs = _make_logged_data(rng)
        target_probs = log_probs.copy()

        r = OfflineEvaluator().evaluate(
            rewards, actions, contexts, log_probs, target_probs
        )

        assert r.se > 0

    def test_clipping_reduces_variance(self, rng):
        """Clipping should reduce variance of the estimate."""
        n = 500
        rewards = rng.binomial(1, 0.3, n).astype(float)
        actions = rng.choice(3, n)
        contexts = rng.normal(0, 1, (n, 2))

        # Very unbalanced probabilities
        log_probs = np.where(actions == 0, 0.01, 0.495)
        target_probs = np.where(actions == 0, 0.8, 0.1)

        r_clipped = OfflineEvaluator(clip=5.0).evaluate(
            rewards, actions, contexts, log_probs, target_probs
        )
        r_unclipped = OfflineEvaluator(clip=1000.0).evaluate(
            rewards, actions, contexts, log_probs, target_probs
        )

        assert r_clipped.se <= r_unclipped.se * 1.5  # Clipped should have less variance


class TestOfflineValidation:
    """Validation and error handling tests."""

    def test_zero_logging_probs(self, rng):
        """Zero logging probabilities should raise ValueError."""
        n = 20
        rewards = rng.normal(0, 1, n)
        actions = rng.choice(3, n)
        contexts = rng.normal(0, 1, (n, 2))
        log_probs = np.zeros(n)
        target_probs = np.full(n, 1.0 / 3)

        with pytest.raises(ValueError, match="strictly positive"):
            OfflineEvaluator().evaluate(
                rewards, actions, contexts, log_probs, target_probs
            )

    def test_negative_target_probs(self, rng):
        """Negative target probabilities should raise ValueError."""
        n = 20
        rewards = rng.normal(0, 1, n)
        actions = rng.choice(3, n)
        contexts = rng.normal(0, 1, (n, 2))
        log_probs = np.full(n, 1.0 / 3)
        target_probs = np.full(n, -0.1)

        with pytest.raises(ValueError, match="non-negative"):
            OfflineEvaluator().evaluate(
                rewards, actions, contexts, log_probs, target_probs
            )

    def test_mismatched_lengths(self, rng):
        """Arrays must have the same length."""
        with pytest.raises(ValueError, match="same length"):
            OfflineEvaluator().evaluate(
                rng.normal(0, 1, 20),
                rng.choice(3, 15),
                rng.normal(0, 1, (20, 2)),
                np.full(20, 1.0 / 3),
                np.full(20, 1.0 / 3),
            )

    def test_invalid_method(self):
        """Invalid method should raise ValueError."""
        with pytest.raises(ValueError, match="method"):
            OfflineEvaluator(method="invalid")

    def test_invalid_alpha(self):
        """Alpha outside (0, 1) should raise ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            OfflineEvaluator(alpha=0.0)

    def test_negative_clip(self):
        """Negative clip should raise ValueError."""
        with pytest.raises(ValueError, match="clip"):
            OfflineEvaluator(clip=-1.0)

    def test_to_dict(self, rng):
        """to_dict should return a plain dictionary."""
        rewards, actions, contexts, log_probs = _make_logged_data(rng)
        r = OfflineEvaluator().evaluate(
            rewards, actions, contexts, log_probs, log_probs.copy()
        )

        d = r.to_dict()
        assert isinstance(d, dict)
        assert "estimated_value" in d
        assert "effective_sample_size" in d

    def test_repr(self, rng):
        """repr should return a formatted string."""
        rewards, actions, contexts, log_probs = _make_logged_data(rng)
        r = OfflineEvaluator().evaluate(
            rewards, actions, contexts, log_probs, log_probs.copy()
        )
        assert "OfflineResult" in repr(r)

    def test_too_few_observations(self):
        """Should reject arrays with fewer than 2 elements."""
        with pytest.raises(ValueError, match="at least 2"):
            OfflineEvaluator().evaluate(
                [1.0], [0], [[1.0]], [0.5], [0.5],
            )
