"""Tests for ResponseAdaptiveRandomization (RAR)."""

from __future__ import annotations

import numpy as np
import pytest

from splita.design.response_adaptive import ResponseAdaptiveRandomization


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestRARBasic:
    """Basic functionality tests."""

    def test_shifts_allocation_toward_better_arm(self):
        """Better arm should get higher allocation over time."""
        rar = ResponseAdaptiveRandomization(n_arms=2, random_state=42)

        # Arm 0 always succeeds, arm 1 always fails
        for _ in range(50):
            rar.update(0, 1.0)
            rar.update(1, 0.0)

        alloc = rar.get_allocation()
        assert alloc[0] > alloc[1]

    def test_min_allocation_respected(self):
        """Even the worst arm should get at least min_allocation."""
        min_alloc = 0.1
        rar = ResponseAdaptiveRandomization(
            n_arms=2, min_allocation=min_alloc, random_state=42
        )

        for _ in range(100):
            rar.update(0, 1.0)
            rar.update(1, 0.0)

        alloc = rar.get_allocation()
        assert alloc[1] >= min_alloc - 0.01  # small numerical tolerance

    def test_equal_rewards_equal_allocation(self):
        """Equal arms should get roughly equal allocation."""
        rar = ResponseAdaptiveRandomization(n_arms=2, random_state=42)

        for _ in range(100):
            rar.update(0, 1.0)
            rar.update(1, 1.0)

        alloc = rar.get_allocation()
        assert abs(alloc[0] - alloc[1]) < 0.2

    def test_three_arms(self):
        """RAR should work with 3+ arms."""
        rar = ResponseAdaptiveRandomization(n_arms=3, random_state=42)

        for _ in range(50):
            rar.update(0, 1.0)
            rar.update(1, 0.5)
            rar.update(2, 0.0)

        alloc = rar.get_allocation()
        assert len(alloc) == 3
        assert alloc[0] > alloc[2]

    def test_bayesian_method(self):
        """Bayesian method should produce valid allocations."""
        rar = ResponseAdaptiveRandomization(
            n_arms=2, method="bayesian", random_state=42
        )
        rar.update(0, 1.0)
        rar.update(1, 0.0)

        alloc = rar.get_allocation()
        assert abs(sum(alloc) - 1.0) < 1e-6

    def test_urn_method(self):
        """Urn method should produce valid allocations."""
        rar = ResponseAdaptiveRandomization(
            n_arms=2, method="urn", random_state=42
        )
        rar.update(0, 1.0)
        rar.update(1, 0.0)

        alloc = rar.get_allocation()
        assert abs(sum(alloc) - 1.0) < 1e-6

    def test_recommend_returns_valid_arm(self):
        """recommend() should return a valid arm index."""
        rar = ResponseAdaptiveRandomization(n_arms=3, random_state=42)
        rar.update(0, 1.0)

        arm = rar.recommend()
        assert 0 <= arm < 3

    def test_result_counts(self):
        """result() should reflect correct observation counts."""
        rar = ResponseAdaptiveRandomization(n_arms=2, random_state=42)

        for _ in range(10):
            rar.update(0, 1.0)
        for _ in range(5):
            rar.update(1, 0.5)

        r = rar.result()
        assert r.n_per_arm[0] == 10
        assert r.n_per_arm[1] == 5
        assert r.total_observations == 15

    def test_result_best_arm(self):
        """best_arm should be the arm with highest allocation."""
        rar = ResponseAdaptiveRandomization(n_arms=2, random_state=42)

        for _ in range(50):
            rar.update(0, 1.0)
            rar.update(1, 0.0)

        r = rar.result()
        assert r.best_arm == 0

    def test_allocations_sum_to_one(self):
        """Allocations should sum to 1.0."""
        rar = ResponseAdaptiveRandomization(n_arms=4, random_state=42)
        for arm in range(4):
            rar.update(arm, float(arm) / 3.0)

        alloc = rar.get_allocation()
        assert abs(sum(alloc) - 1.0) < 1e-6


class TestRARValidation:
    """Input validation tests."""

    def test_invalid_n_arms(self):
        with pytest.raises(ValueError, match="n_arms"):
            ResponseAdaptiveRandomization(n_arms=1)

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method"):
            ResponseAdaptiveRandomization(method="invalid")

    def test_invalid_min_allocation(self):
        with pytest.raises(ValueError, match="min_allocation"):
            ResponseAdaptiveRandomization(min_allocation=-0.1)

    def test_min_allocation_too_high(self):
        """min_allocation * n_arms > 1 should fail."""
        with pytest.raises(ValueError, match="min_allocation"):
            ResponseAdaptiveRandomization(n_arms=3, min_allocation=0.4)

    def test_invalid_arm_index(self):
        rar = ResponseAdaptiveRandomization(n_arms=2)
        with pytest.raises(ValueError, match="arm"):
            rar.update(5, 1.0)


class TestRARResult:
    """Result object tests."""

    def test_to_dict(self):
        rar = ResponseAdaptiveRandomization(n_arms=2, random_state=42)
        rar.update(0, 1.0)
        rar.update(1, 0.0)

        r = rar.result()
        d = r.to_dict()
        assert isinstance(d, dict)
        assert "allocations" in d
        assert "n_per_arm" in d
        assert "best_arm" in d
        assert "total_observations" in d

    def test_repr(self):
        rar = ResponseAdaptiveRandomization(n_arms=2, random_state=42)
        rar.update(0, 1.0)

        r = rar.result()
        assert "RARResult" in repr(r)
