"""Tests for LinTS (Linear Thompson Sampling) contextual bandit."""

from __future__ import annotations

import numpy as np
import pytest

from splita.bandits.lints import LinTS

# ─── Basic tests ────────────────────────────────────────────────────


class TestPersonalizedRecommendation:
    """Arm 0 is better for context type A, arm 1 for context type B."""

    def test_recommends_correctly_after_training(self):
        rng = np.random.default_rng(42)
        lints = LinTS(2, n_features=5, random_state=0)

        ctx_a = np.array([1.0, 0.0, 0.0, 1.0, 0.0])
        ctx_b = np.array([0.0, 1.0, 1.0, 0.0, 1.0])

        # Train: arm 0 rewards high for ctx_a, arm 1 for ctx_b
        for _ in range(200):
            lints.update(0, ctx_a, reward=1.0 + rng.normal(0, 0.1))
            lints.update(1, ctx_a, reward=0.1 + rng.normal(0, 0.1))
            lints.update(0, ctx_b, reward=0.1 + rng.normal(0, 0.1))
            lints.update(1, ctx_b, reward=1.0 + rng.normal(0, 0.1))

        # After training, should recommend arm 0 for ctx_a, arm 1 for ctx_b
        # Check majority over multiple samples to account for stochasticity
        recs_a = [lints.recommend(ctx_a) for _ in range(20)]
        recs_b = [lints.recommend(ctx_b) for _ in range(20)]

        assert recs_a.count(0) > 15, f"Expected arm 0 for ctx_a, got {recs_a}"
        assert recs_b.count(1) > 15, f"Expected arm 1 for ctx_b, got {recs_b}"


class TestLearningConverges:
    """Recommendations stabilise after many updates."""

    def test_stable_recommendations(self):
        lints = LinTS(2, n_features=3, random_state=123)
        ctx = np.array([1.0, 0.5, -0.5])
        rng = np.random.default_rng(99)

        # Arm 0 is consistently better
        for _ in range(300):
            lints.update(0, ctx, reward=1.0 + rng.normal(0, 0.1))
            lints.update(1, ctx, reward=0.2 + rng.normal(0, 0.1))

        recs = [lints.recommend(ctx) for _ in range(30)]
        assert recs.count(0) > 25, f"Expected stable arm 0, got {recs}"


class TestThreeArms:
    """Works with 3 arms."""

    def test_three_arms(self):
        lints = LinTS(3, n_features=2, random_state=7)
        ctx = np.array([1.0, 0.0])
        rng = np.random.default_rng(7)

        for _ in range(200):
            lints.update(0, ctx, reward=0.1 + rng.normal(0, 0.1))
            lints.update(1, ctx, reward=0.5 + rng.normal(0, 0.1))
            lints.update(2, ctx, reward=1.0 + rng.normal(0, 0.1))

        recs = [lints.recommend(ctx) for _ in range(20)]
        assert recs.count(2) > 15, f"Expected arm 2 dominant, got {recs}"


# ─── Update / Recommend ────────────────────────────────────────────


class TestUpdateInternalState:
    """Internal state changes after an update."""

    def test_b_matrix_changes(self):
        lints = LinTS(2, n_features=3, random_state=0)
        b_before = lints._B[0].copy()
        lints.update(0, [1.0, 0.0, 0.0], reward=1.0)
        assert not np.array_equal(lints._B[0], b_before)


class TestRecommendReturnsValidArm:
    """recommend() always returns an arm in [0, n_arms)."""

    def test_valid_range(self):
        lints = LinTS(4, n_features=2, random_state=0)
        ctx = np.array([0.5, -0.5])
        for _ in range(50):
            arm = lints.recommend(ctx)
            assert 0 <= arm < 4


class TestDifferentContextsDifferentRecommendations:
    """After training, different contexts can yield different arms."""

    def test_context_dependent(self):
        lints = LinTS(2, n_features=2, random_state=0)
        ctx_a = np.array([1.0, 0.0])
        ctx_b = np.array([0.0, 1.0])
        rng = np.random.default_rng(0)

        for _ in range(200):
            lints.update(0, ctx_a, reward=1.0 + rng.normal(0, 0.05))
            lints.update(1, ctx_a, reward=0.0 + rng.normal(0, 0.05))
            lints.update(0, ctx_b, reward=0.0 + rng.normal(0, 0.05))
            lints.update(1, ctx_b, reward=1.0 + rng.normal(0, 0.05))

        recs_a = [lints.recommend(ctx_a) for _ in range(20)]
        recs_b = [lints.recommend(ctx_b) for _ in range(20)]

        # At least some difference
        assert recs_a.count(0) > recs_b.count(0)


# ─── Validation ─────────────────────────────────────────────────────


class TestValidation:
    """Constructor and method input validation."""

    def test_n_arms_too_small(self):
        with pytest.raises(ValueError, match="n_arms"):
            LinTS(1, n_features=2)

    def test_n_features_too_small(self):
        with pytest.raises(ValueError, match="n_features"):
            LinTS(2, n_features=0)

    def test_lambda_not_positive(self):
        with pytest.raises(ValueError, match="lambda_"):
            LinTS(2, n_features=2, lambda_=0.0)

    def test_lambda_negative(self):
        with pytest.raises(ValueError, match="lambda_"):
            LinTS(2, n_features=2, lambda_=-1.0)

    def test_noise_var_not_positive(self):
        with pytest.raises(ValueError, match="noise_var"):
            LinTS(2, n_features=2, noise_var=0.0)

    def test_noise_var_negative(self):
        with pytest.raises(ValueError, match="noise_var"):
            LinTS(2, n_features=2, noise_var=-0.5)

    def test_wrong_context_shape(self):
        lints = LinTS(2, n_features=3, random_state=0)
        with pytest.raises(ValueError, match="context"):
            lints.recommend(np.array([1.0, 2.0]))

    def test_wrong_context_shape_2d(self):
        lints = LinTS(2, n_features=3, random_state=0)
        with pytest.raises(ValueError, match="context"):
            lints.recommend(np.array([[1.0, 2.0, 3.0]]))

    def test_invalid_arm_too_high(self):
        lints = LinTS(2, n_features=2, random_state=0)
        with pytest.raises(ValueError, match="arm"):
            lints.update(2, [1.0, 0.0], reward=1.0)

    def test_invalid_arm_negative(self):
        lints = LinTS(2, n_features=2, random_state=0)
        with pytest.raises(ValueError, match="arm"):
            lints.update(-1, [1.0, 0.0], reward=1.0)

    def test_invalid_arm_type(self):
        lints = LinTS(2, n_features=2, random_state=0)
        with pytest.raises(TypeError, match="arm"):
            lints.update("a", [1.0, 0.0], reward=1.0)  # type: ignore[arg-type]


# ─── Reproducibility ───────────────────────────────────────────────


class TestReproducibility:
    """Same seed produces same recommendations."""

    def test_same_seed_same_results(self):
        ctx = np.array([1.0, 0.5, -0.3])

        lints_a = LinTS(3, n_features=3, random_state=42)
        lints_b = LinTS(3, n_features=3, random_state=42)

        # Same updates
        for lints in (lints_a, lints_b):
            lints.update(0, ctx, reward=1.0)
            lints.update(1, ctx, reward=0.5)
            lints.update(2, ctx, reward=0.2)

        recs_a = [lints_a.recommend(ctx) for _ in range(10)]
        recs_b = [lints_b.recommend(ctx) for _ in range(10)]

        assert recs_a == recs_b


# ─── Edge cases ─────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases: single feature, no updates, large feature space."""

    def test_single_feature(self):
        lints = LinTS(2, n_features=1, random_state=0)
        ctx = np.array([1.0])
        arm = lints.recommend(ctx)
        assert 0 <= arm < 2

        lints.update(0, ctx, reward=1.0)
        arm = lints.recommend(ctx)
        assert 0 <= arm < 2

    def test_no_updates_recommend_works(self):
        """Before any updates, recommend still works (from prior)."""
        lints = LinTS(2, n_features=3, random_state=0)
        ctx = np.array([1.0, 0.0, -1.0])
        arm = lints.recommend(ctx)
        assert 0 <= arm < 2

    def test_large_n_features(self):
        """50 features should work without issue."""
        d = 50
        lints = LinTS(2, n_features=d, random_state=0)
        rng = np.random.default_rng(0)
        ctx = rng.standard_normal(d)

        # A few updates
        for _ in range(10):
            lints.update(0, ctx, reward=1.0)
            lints.update(1, ctx, reward=0.5)

        arm = lints.recommend(ctx)
        assert 0 <= arm < 2


# ─── Result ──────────────────────────────────────────────────────────


class TestResult:
    """Tests for the result() method."""

    def test_properties(self):
        """n_arms and n_features properties return correct values."""
        lints = LinTS(4, n_features=7, random_state=0)
        assert lints.n_arms == 4
        assert lints.n_features == 7

    def test_result_before_updates(self):
        """result() works before any updates."""
        lints = LinTS(2, n_features=3, random_state=0)
        result = lints.result()
        assert len(result.prob_best) == 2
        assert sum(result.n_pulls_per_arm) == 0
        assert result.total_reward == 0.0
        assert result.should_stop is False
        assert result.current_best_arm == 0

    def test_result_after_training(self):
        """result() reflects training data."""
        lints = LinTS(2, n_features=3, random_state=0)
        rng = np.random.default_rng(42)
        ctx = np.array([1.0, 0.5, -0.5])

        for _ in range(100):
            lints.update(0, ctx, reward=1.0 + rng.normal(0, 0.1))
            lints.update(1, ctx, reward=0.2 + rng.normal(0, 0.1))

        result = lints.result()
        assert result.n_pulls_per_arm == [100, 100]
        assert result.current_best_arm == 0
        assert result.arm_means[0] > result.arm_means[1]
        assert result.total_reward > 0.0
        assert result.cumulative_regret is None
        assert result.should_stop is False

    def test_result_credible_intervals(self):
        """Credible intervals are present for trained arms."""
        lints = LinTS(2, n_features=2, random_state=0)
        ctx = np.array([1.0, 0.0])

        for _ in range(50):
            lints.update(0, ctx, reward=1.0)
            lints.update(1, ctx, reward=0.5)

        result = lints.result()
        for lo, hi in result.arm_credible_intervals:
            assert lo < hi

    def test_result_to_dict(self):
        """result().to_dict() produces a plain dict."""
        lints = LinTS(2, n_features=2, random_state=0)
        lints.update(0, [1.0, 0.0], reward=1.0)
        d = lints.result().to_dict()
        assert isinstance(d, dict)
        assert "n_pulls_per_arm" in d
