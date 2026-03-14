"""Tests for LinUCB contextual bandit."""

from __future__ import annotations

import numpy as np
import pytest

from splita.bandits.linucb import LinUCB
from splita._types import BanditResult


# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ─── Basic functionality ─────────────────────────────────────────────


class TestBasic:
    def test_recommend_returns_int(self):
        ucb = LinUCB(2, n_features=3, random_state=42)
        ctx = np.array([1.0, 0.0, 0.5])
        arm = ucb.recommend(ctx)
        assert isinstance(arm, int)
        assert 0 <= arm < 2

    def test_update_and_recommend(self):
        ucb = LinUCB(2, n_features=3, random_state=42)
        ctx = np.array([1.0, 0.0, 0.5])
        arm = ucb.recommend(ctx)
        ucb.update(arm, ctx, reward=1.0)
        # Should work without error
        arm2 = ucb.recommend(ctx)
        assert isinstance(arm2, int)

    def test_result_returns_bandit_result(self):
        ucb = LinUCB(2, n_features=3, random_state=42)
        ctx = np.array([1.0, 0.0, 0.5])
        ucb.update(0, ctx, 1.0)
        result = ucb.result()
        assert isinstance(result, BanditResult)

    def test_n_pulls_tracked(self):
        ucb = LinUCB(3, n_features=2, random_state=42)
        ctx = np.array([1.0, 0.5])
        ucb.update(0, ctx, 1.0)
        ucb.update(1, ctx, 0.5)
        ucb.update(0, ctx, 0.8)
        result = ucb.result()
        assert result.n_pulls_per_arm == [2, 1, 0]


# ─── UCB exploration ─────────────────────────────────────────────────


class TestExploration:
    def test_explores_unvisited_arms(self, rng):
        ucb = LinUCB(3, n_features=2, alpha=10.0, random_state=42)
        ctx = np.array([1.0, 0.5])
        # Pull arm 0 many times
        for _ in range(50):
            ucb.update(0, ctx, 0.5)
        # With high alpha, unexplored arms should be preferred
        arm = ucb.recommend(ctx)
        assert arm in (1, 2)

    def test_alpha_controls_exploration(self, rng):
        # Low alpha should exploit more
        ucb_low = LinUCB(2, n_features=2, alpha=0.01, random_state=42)
        ucb_high = LinUCB(2, n_features=2, alpha=10.0, random_state=42)

        ctx = np.array([1.0, 0.5])
        # Give arm 0 high reward
        for _ in range(20):
            ucb_low.update(0, ctx, 1.0)
            ucb_high.update(0, ctx, 1.0)

        # Low alpha: should recommend arm 0 (exploit)
        assert ucb_low.recommend(ctx) == 0


# ─── Properties ──────────────────────────────────────────────────────


class TestProperties:
    def test_n_arms_property(self):
        ucb = LinUCB(4, n_features=3)
        assert ucb.n_arms == 4

    def test_n_features_property(self):
        ucb = LinUCB(2, n_features=5)
        assert ucb.n_features == 5


# ─── Result fields ──────────────────────────────────────────────────


class TestResultFields:
    def test_arm_means(self):
        ucb = LinUCB(2, n_features=2, random_state=42)
        ctx = np.array([1.0, 0.5])
        ucb.update(0, ctx, 1.0)
        ucb.update(0, ctx, 0.5)
        ucb.update(1, ctx, 0.2)
        result = ucb.result()
        assert result.arm_means[0] == pytest.approx(0.75)
        assert result.arm_means[1] == pytest.approx(0.2)

    def test_total_reward(self):
        ucb = LinUCB(2, n_features=2, random_state=42)
        ctx = np.array([1.0, 0.5])
        ucb.update(0, ctx, 1.0)
        ucb.update(1, ctx, 0.5)
        result = ucb.result()
        assert result.total_reward == pytest.approx(1.5)

    def test_arm_credible_intervals(self):
        ucb = LinUCB(2, n_features=2, random_state=42)
        ctx = np.array([1.0, 0.5])
        for _ in range(10):
            ucb.update(0, ctx, 1.0)
        result = ucb.result()
        ci = result.arm_credible_intervals[0]
        assert ci[0] < ci[1]

    def test_unpulled_arm_ci(self):
        ucb = LinUCB(2, n_features=2, random_state=42)
        result = ucb.result()
        assert result.arm_credible_intervals[0] == (0.0, 0.0)


# ─── Validation ──────────────────────────────────────────────────────


class TestValidation:
    def test_n_arms_less_than_2(self):
        with pytest.raises(ValueError, match="n_arms"):
            LinUCB(1, n_features=2)

    def test_n_features_less_than_1(self):
        with pytest.raises(ValueError, match="n_features"):
            LinUCB(2, n_features=0)

    def test_alpha_not_positive(self):
        with pytest.raises(ValueError, match="alpha"):
            LinUCB(2, n_features=2, alpha=0.0)

    def test_invalid_arm_type(self):
        ucb = LinUCB(2, n_features=2)
        with pytest.raises(TypeError, match="arm"):
            ucb.update("a", [1.0, 0.5], 1.0)

    def test_arm_out_of_range(self):
        ucb = LinUCB(2, n_features=2)
        with pytest.raises(ValueError, match="arm"):
            ucb.update(5, [1.0, 0.5], 1.0)

    def test_context_wrong_shape(self):
        ucb = LinUCB(2, n_features=3)
        with pytest.raises(ValueError, match="context"):
            ucb.recommend([1.0, 0.5])  # expects 3 features

    def test_context_2d_raises(self):
        ucb = LinUCB(2, n_features=2)
        with pytest.raises(ValueError, match="context"):
            ucb.recommend([[1.0, 0.5]])


# ─── Serialization ──────────────────────────────────────────────────


class TestSerialization:
    def test_result_to_dict(self):
        ucb = LinUCB(2, n_features=2, random_state=42)
        ucb.update(0, [1.0, 0.5], 1.0)
        d = ucb.result().to_dict()
        assert "n_pulls_per_arm" in d
        assert "arm_means" in d
