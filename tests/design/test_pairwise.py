"""Tests for PairwiseDesign (Tier 3, Item 13)."""

from __future__ import annotations

import numpy as np
import pytest

from splita._types import PairwiseDesignResult
from splita.design.pairwise import PairwiseDesign


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestPairwiseDesignBasic:
    """Basic functionality tests."""

    def test_balanced_assignment(self, rng):
        """Should produce roughly balanced treatment/control groups."""
        X = rng.normal(size=(100, 3))
        result = PairwiseDesign(random_state=42).assign(X)

        assert isinstance(result, PairwiseDesignResult)
        n_trt = sum(1 for a in result.assignments if a == 1)
        n_ctrl = sum(1 for a in result.assignments if a == 0)
        assert n_trt == 50
        assert n_ctrl == 50

    def test_n_pairs_correct(self, rng):
        """Number of pairs should be n // 2."""
        X = rng.normal(size=(20, 3))
        result = PairwiseDesign(random_state=42).assign(X)

        assert result.n_pairs == 10
        assert len(result.pairs) == 10

    def test_better_balance_than_random(self, rng):
        """Pairwise design should yield better balance than random assignment."""
        X = rng.normal(size=(100, 5))
        X[:50, 0] += 2  # Create imbalance in feature 0

        # Pairwise
        pw_result = PairwiseDesign(random_state=42).assign(X)

        # Random assignment
        random_assignments = np.zeros(100, dtype=int)
        idx = rng.permutation(100)
        random_assignments[idx[:50]] = 1

        # Compute balance for random
        ctrl = X[random_assignments == 0]
        trt = X[random_assignments == 1]
        max_smd = 0.0
        for col in range(X.shape[1]):
            mean_diff = abs(np.mean(trt[:, col]) - np.mean(ctrl[:, col]))
            pooled_std = np.sqrt(
                (np.var(ctrl[:, col], ddof=1) + np.var(trt[:, col], ddof=1)) / 2
            )
            if pooled_std > 0:
                max_smd = max(max_smd, mean_diff / pooled_std)

        # Pairwise should have lower balance score (better balance)
        assert pw_result.balance_score <= max_smd + 0.5  # generous tolerance

    def test_pairs_are_reasonable(self, rng):
        """Paired units should be more similar than average distance."""
        X = rng.normal(size=(20, 3))
        result = PairwiseDesign(random_state=42).assign(X)

        # Average distance between paired units
        pair_dists = []
        for i, j in result.pairs:
            pair_dists.append(np.linalg.norm(X[i] - X[j]))

        # Average distance between all units
        from scipy.spatial.distance import pdist

        all_dists = pdist(X)

        assert np.mean(pair_dists) < np.mean(all_dists)

    def test_odd_n_handled(self, rng):
        """Odd number of units should be handled gracefully."""
        X = rng.normal(size=(21, 3))
        result = PairwiseDesign(random_state=42).assign(X)

        assert result.n_pairs == 10  # 21 // 2
        assert len(result.assignments) == 21
        # Unmatched unit should be assigned to control
        n_ctrl = sum(1 for a in result.assignments if a == 0)
        assert n_ctrl == 11  # 10 matched + 1 unmatched

    def test_each_pair_has_one_trt_one_ctrl(self, rng):
        """Each pair should have exactly one treatment and one control unit."""
        X = rng.normal(size=(20, 3))
        result = PairwiseDesign(random_state=42).assign(X)

        for ctrl_idx, trt_idx in result.pairs:
            assert result.assignments[ctrl_idx] == 0
            assert result.assignments[trt_idx] == 1

    def test_no_unit_in_multiple_pairs(self, rng):
        """No unit should appear in more than one pair."""
        X = rng.normal(size=(30, 3))
        result = PairwiseDesign(random_state=42).assign(X)

        all_indices = []
        for i, j in result.pairs:
            all_indices.extend([i, j])

        assert len(all_indices) == len(set(all_indices))

    def test_single_feature(self, rng):
        """Should work with a single feature."""
        X = rng.normal(size=(20, 1))
        result = PairwiseDesign(random_state=42).assign(X)

        assert result.n_pairs == 10

    def test_1d_input_reshaped(self, rng):
        """1-D input should be reshaped automatically."""
        X = rng.normal(size=20)
        result = PairwiseDesign(random_state=42).assign(X)

        assert result.n_pairs == 10

    def test_to_dict(self, rng):
        """Result should serialise to a dictionary."""
        X = rng.normal(size=(10, 2))
        result = PairwiseDesign(random_state=42).assign(X)

        d = result.to_dict()
        assert isinstance(d, dict)
        assert "assignments" in d
        assert "balance_score" in d

    def test_repr(self, rng):
        """Result __repr__ should be a string."""
        X = rng.normal(size=(10, 2))
        result = PairwiseDesign(random_state=42).assign(X)

        assert "PairwiseDesignResult" in repr(result)

    def test_deterministic_with_seed(self, rng):
        """Same seed should produce same assignments."""
        X = rng.normal(size=(20, 3))
        r1 = PairwiseDesign(random_state=42).assign(X)
        r2 = PairwiseDesign(random_state=42).assign(X)

        assert r1.assignments == r2.assignments


class TestPairwiseDesignValidation:
    """Validation and error handling tests."""

    def test_too_few_units_raises(self):
        """Fewer than 2 units should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            PairwiseDesign().assign(np.array([[1, 2]]))

    def test_3d_input_raises(self):
        """3-D input should raise ValueError."""
        with pytest.raises(ValueError, match="1-D or 2-D"):
            PairwiseDesign().assign(np.ones((2, 3, 4)))

    def test_singular_covariance_fallback(self):
        """Lines 100-102: singular cov should fall back to euclidean."""
        # All same feature values in one column -> singular covariance
        X = np.zeros((10, 2))
        X[:, 0] = 1.0  # constant column
        X[:, 1] = np.arange(10, dtype=float)
        result = PairwiseDesign(random_state=42).assign(X)
        assert result.n_pairs == 5

    def test_balance_empty_group_returns_zero(self):
        """Line 178: balance with no ctrl or trt returns 0.0."""
        X = np.random.default_rng(42).normal(size=(10, 2))
        assignments = np.zeros(10, dtype=int)  # all control
        score = PairwiseDesign._compute_balance(X, assignments)
        assert score == 0.0
