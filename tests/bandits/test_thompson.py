"""Tests for ThompsonSampler."""

from __future__ import annotations

import numpy as np
import pytest

from splita.bandits.thompson import ThompsonSampler

# ─── Basic: correct arm identification ───────────────────────────


class TestBernoulli:
    """Bernoulli likelihood identifies the better arm."""

    def test_two_arms_identifies_best(self):
        """Arm 0 at 80% vs arm 1 at 20% → arm 0 wins."""
        rng = np.random.default_rng(42)
        ts = ThompsonSampler(2, random_state=0)
        for _ in range(500):
            # arm 0: 80% conversion
            ts.update(0, int(rng.random() < 0.8))
            # arm 1: 20% conversion
            ts.update(1, int(rng.random() < 0.2))

        result = ts.result()
        assert result.current_best_arm == 0
        assert result.arm_means[0] > result.arm_means[1]


class TestGaussian:
    """Gaussian likelihood identifies the better arm."""

    def test_two_arms_identifies_best(self):
        """Arm 0 mean=10 vs arm 1 mean=5 → arm 0 wins."""
        rng = np.random.default_rng(42)
        ts = ThompsonSampler(2, likelihood="gaussian", random_state=0)
        for _ in range(300):
            ts.update(0, rng.normal(10.0, 1.0))
            ts.update(1, rng.normal(5.0, 1.0))

        result = ts.result()
        assert result.current_best_arm == 0


class TestPoisson:
    """Poisson likelihood identifies the better arm."""

    def test_two_arms_identifies_best(self):
        """Arm 0 rate=8 vs arm 1 rate=2 → arm 0 wins."""
        rng = np.random.default_rng(42)
        ts = ThompsonSampler(2, likelihood="poisson", random_state=0)
        for _ in range(300):
            ts.update(0, int(rng.poisson(8)))
            ts.update(1, int(rng.poisson(2)))

        result = ts.result()
        assert result.current_best_arm == 0


class TestMultipleArms:
    """Works with more than 2 arms."""

    def test_three_arms(self):
        rng = np.random.default_rng(42)
        ts = ThompsonSampler(3, random_state=0)
        rates = [0.9, 0.5, 0.1]
        for _ in range(400):
            for arm, rate in enumerate(rates):
                ts.update(arm, int(rng.random() < rate))

        result = ts.result()
        assert result.current_best_arm == 0


# ─── Recommend ───────────────────────────────────────────────────


class TestRecommend:
    """Tests for the recommend() method."""

    def test_returns_valid_arm_index(self):
        ts = ThompsonSampler(4, random_state=0)
        for _ in range(20):
            arm = ts.recommend()
            assert 0 <= arm < 4

    def test_exploration_early(self):
        """With few samples, both arms should get recommended."""
        ts = ThompsonSampler(2, random_state=42)
        # Give each arm one pull so posteriors are similar
        ts.update(0, 1)
        ts.update(1, 1)
        arms_seen = set()
        for _ in range(100):
            arms_seen.add(ts.recommend())
        assert len(arms_seen) == 2

    def test_exploitation_later(self):
        """With many samples, best arm recommended most often."""
        rng = np.random.default_rng(42)
        ts = ThompsonSampler(2, random_state=0)
        for _ in range(500):
            ts.update(0, int(rng.random() < 0.9))
            ts.update(1, int(rng.random() < 0.1))

        counts = {0: 0, 1: 0}
        for _ in range(200):
            counts[ts.recommend()] += 1
        assert counts[0] > counts[1]


# ─── Result ──────────────────────────────────────────────────────


class TestResult:
    """Tests for the result() method outputs."""

    def _make_trained(self) -> ThompsonSampler:
        rng = np.random.default_rng(42)
        ts = ThompsonSampler(2, random_state=0)
        for _ in range(300):
            ts.update(0, int(rng.random() < 0.7))
            ts.update(1, int(rng.random() < 0.3))
        return ts

    def test_prob_best_sums_to_one(self):
        result = self._make_trained().result()
        assert abs(sum(result.prob_best) - 1.0) < 0.05

    def test_expected_loss_non_negative(self):
        result = self._make_trained().result()
        for loss in result.expected_loss:
            assert loss >= 0.0

    def test_best_arm_has_lowest_expected_loss(self):
        result = self._make_trained().result()
        best = result.current_best_arm
        assert result.expected_loss[best] == min(result.expected_loss)

    def test_arm_means_reasonable(self):
        """Posterior means should be close to true values after many pulls."""
        result = self._make_trained().result()
        assert abs(result.arm_means[0] - 0.7) < 0.1
        assert abs(result.arm_means[1] - 0.3) < 0.1

    def test_credible_intervals_contain_true_value(self):
        """95% CI should contain the true conversion rate."""
        result = self._make_trained().result()
        lo0, hi0 = result.arm_credible_intervals[0]
        lo1, hi1 = result.arm_credible_intervals[1]
        assert lo0 < 0.7 < hi0
        assert lo1 < 0.3 < hi1


# ─── Stopping ────────────────────────────────────────────────────


class TestStopping:
    """Tests for stopping rules."""

    def test_expected_loss_stopping(self):
        """Converges and stops with expected_loss rule."""
        rng = np.random.default_rng(42)
        ts = ThompsonSampler(
            2,
            stopping_rule="expected_loss",
            stopping_threshold=0.01,
            min_samples=50,
            random_state=0,
        )
        for _ in range(500):
            ts.update(0, int(rng.random() < 0.8))
            ts.update(1, int(rng.random() < 0.2))

        assert ts.result().should_stop is True

    def test_prob_best_stopping(self):
        """Converges and stops with prob_best rule."""
        rng = np.random.default_rng(42)
        ts = ThompsonSampler(
            2,
            stopping_rule="prob_best",
            stopping_threshold=0.95,
            min_samples=50,
            random_state=0,
        )
        for _ in range(500):
            ts.update(0, int(rng.random() < 0.8))
            ts.update(1, int(rng.random() < 0.2))

        assert ts.result().should_stop is True

    def test_n_samples_stopping(self):
        """Stops after threshold total observations."""
        ts = ThompsonSampler(
            2,
            stopping_rule="n_samples",
            stopping_threshold=10,
            min_samples=0,
            random_state=0,
        )
        for _i in range(5):
            ts.update(0, 1)
            ts.update(1, 0)
        assert ts.result().should_stop is True

    def test_min_samples_respected(self):
        """Does not stop before min_samples even if threshold met."""
        rng = np.random.default_rng(42)
        ts = ThompsonSampler(
            2,
            stopping_rule="expected_loss",
            stopping_threshold=0.5,  # very easy threshold
            min_samples=200,
            random_state=0,
        )
        for _ in range(50):
            ts.update(0, int(rng.random() < 0.9))
            ts.update(1, int(rng.random() < 0.1))

        # only 100 total samples, min is 200
        assert ts.result().should_stop is False

    def test_should_stop_initially_false(self):
        """Before any updates, should_stop is False."""
        ts = ThompsonSampler(2, random_state=0)
        assert ts.result().should_stop is False


# ─── Validation ──────────────────────────────────────────────────


class TestValidation:
    """Input validation tests."""

    def test_n_arms_less_than_2(self):
        with pytest.raises(ValueError, match="n_arms"):
            ThompsonSampler(1)

    def test_invalid_arm_index(self):
        ts = ThompsonSampler(2, random_state=0)
        with pytest.raises(ValueError, match="arm"):
            ts.update(5, 1.0)

    def test_invalid_arm_index_negative(self):
        ts = ThompsonSampler(2, random_state=0)
        with pytest.raises(ValueError, match="arm"):
            ts.update(-1, 1.0)

    def test_invalid_reward_bernoulli(self):
        ts = ThompsonSampler(2, random_state=0)
        with pytest.raises(ValueError, match="reward"):
            ts.update(0, 0.5)

    def test_invalid_arm_type(self):
        ts = ThompsonSampler(2, random_state=0)
        with pytest.raises(TypeError, match="arm"):
            ts.update("a", 1.0)  # type: ignore[arg-type]

    def test_invalid_reward_poisson_negative(self):
        ts = ThompsonSampler(2, likelihood="poisson", random_state=0)
        with pytest.raises(ValueError, match="reward"):
            ts.update(0, -1)

    def test_invalid_reward_poisson_non_integer(self):
        ts = ThompsonSampler(2, likelihood="poisson", random_state=0)
        with pytest.raises(ValueError, match="reward"):
            ts.update(0, 2.5)

    def test_invalid_likelihood(self):
        with pytest.raises(ValueError, match="likelihood"):
            ThompsonSampler(2, likelihood="beta")

    def test_invalid_stopping_rule(self):
        with pytest.raises(ValueError, match="stopping_rule"):
            ThompsonSampler(2, stopping_rule="magic")


# ─── Reproducibility ─────────────────────────────────────────────


class TestReproducibility:
    """Same seed produces same results."""

    def test_same_seed_same_recommendations(self):
        arms_a = []
        arms_b = []
        for seed_list in [arms_a, arms_b]:
            ts = ThompsonSampler(3, random_state=123)
            ts.update(0, 1)
            ts.update(1, 0)
            ts.update(2, 1)
            for _ in range(20):
                seed_list.append(ts.recommend())
        assert arms_a == arms_b


# ─── Edge cases ──────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases that should not crash."""

    def test_no_updates_result(self):
        """result() works before any updates."""
        ts = ThompsonSampler(2, random_state=0)
        result = ts.result()
        assert len(result.prob_best) == 2
        assert sum(result.n_pulls_per_arm) == 0
        assert result.total_reward == 0.0

    def test_single_pull_per_arm(self):
        """Works with just one pull per arm."""
        ts = ThompsonSampler(2, random_state=0)
        ts.update(0, 1)
        ts.update(1, 0)
        result = ts.result()
        assert len(result.arm_means) == 2
        assert result.n_pulls_per_arm == [1, 1]


# ─── Custom prior ──────────────────────────────────────────────────


class TestCustomPrior:
    """Tests for explicit prior hyperparameters."""

    def test_bernoulli_custom_prior(self):
        """Bernoulli with informative prior (alpha=10, beta=1) favours success."""
        ts = ThompsonSampler(
            2,
            likelihood="bernoulli",
            prior={"alpha": 10.0, "beta": 1.0},
            random_state=42,
        )
        # With strong prior towards success, means should start high
        result = ts.result()
        for mean in result.arm_means:
            assert mean > 0.8

    def test_bernoulli_custom_prior_still_learns(self):
        """Even with strong prior, enough data shifts the posterior."""
        rng = np.random.default_rng(42)
        ts = ThompsonSampler(
            2,
            likelihood="bernoulli",
            prior={"alpha": 10.0, "beta": 1.0},
            random_state=0,
        )
        # Arm 0: 20% (against the prior), arm 1: 80% (with the prior)
        for _ in range(500):
            ts.update(0, int(rng.random() < 0.2))
            ts.update(1, int(rng.random() < 0.8))

        result = ts.result()
        assert result.current_best_arm == 1


# ─── Gaussian accepts any float ───────────────────────────────────


class TestGaussianRewardRange:
    """Gaussian likelihood accepts any finite float reward."""

    def test_negative_reward(self):
        ts = ThompsonSampler(2, likelihood="gaussian", random_state=0)
        ts.update(0, -5.0)
        assert ts.result().total_reward == -5.0

    def test_large_reward(self):
        ts = ThompsonSampler(2, likelihood="gaussian", random_state=0)
        ts.update(0, 1e6)
        assert ts.result().total_reward == 1e6

    def test_fractional_reward(self):
        ts = ThompsonSampler(2, likelihood="gaussian", random_state=0)
        ts.update(0, 3.14159)
        assert abs(ts.result().total_reward - 3.14159) < 1e-10

    def test_nan_reward_rejected(self):
        ts = ThompsonSampler(2, likelihood="gaussian", random_state=0)
        with pytest.raises(ValueError, match="reward"):
            ts.update(0, float("nan"))

    def test_inf_reward_rejected(self):
        ts = ThompsonSampler(2, likelihood="gaussian", random_state=0)
        with pytest.raises(ValueError, match="reward"):
            ts.update(0, float("inf"))

    def test_neg_inf_reward_rejected(self):
        ts = ThompsonSampler(2, likelihood="gaussian", random_state=0)
        with pytest.raises(ValueError, match="reward"):
            ts.update(0, float("-inf"))


# ─── Stopping threshold validation ───────────────────────────────


class TestStoppingThresholdValidation:
    """Stopping threshold must be valid for the chosen rule."""

    def test_expected_loss_threshold_must_be_positive(self):
        with pytest.raises(ValueError, match="stopping_threshold"):
            ThompsonSampler(2, stopping_rule="expected_loss", stopping_threshold=0)

    def test_expected_loss_threshold_negative(self):
        with pytest.raises(ValueError, match="stopping_threshold"):
            ThompsonSampler(2, stopping_rule="expected_loss", stopping_threshold=-0.1)

    def test_prob_best_threshold_zero(self):
        with pytest.raises(ValueError, match="stopping_threshold"):
            ThompsonSampler(2, stopping_rule="prob_best", stopping_threshold=0)

    def test_prob_best_threshold_one(self):
        with pytest.raises(ValueError, match="stopping_threshold"):
            ThompsonSampler(2, stopping_rule="prob_best", stopping_threshold=1.0)

    def test_prob_best_threshold_above_one(self):
        with pytest.raises(ValueError, match="stopping_threshold"):
            ThompsonSampler(2, stopping_rule="prob_best", stopping_threshold=1.5)

    def test_n_samples_threshold_must_be_positive(self):
        with pytest.raises(ValueError, match="stopping_threshold"):
            ThompsonSampler(2, stopping_rule="n_samples", stopping_threshold=0)

    def test_n_samples_threshold_negative(self):
        with pytest.raises(ValueError, match="stopping_threshold"):
            ThompsonSampler(2, stopping_rule="n_samples", stopping_threshold=-10)
