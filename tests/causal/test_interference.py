"""Tests for InterferenceExperiment (Tier 3, Item 10)."""

from __future__ import annotations

import numpy as np
import pytest

from splita._types import InterferenceResult
from splita.causal.interference import InterferenceExperiment


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestInterferenceBasic:
    """Basic functionality tests."""

    def test_significant_treatment_effect(self, rng):
        """Large treatment effect should be detected."""
        n_clusters = 20
        n_per = 20
        outcomes, treatments, clusters = [], [], []
        for c in range(n_clusters):
            for i in range(n_per):
                t = 1 if i < n_per // 2 else 0
                y = rng.normal(10 + t * 3.0, 1)
                outcomes.append(y)
                treatments.append(t)
                clusters.append(c)

        result = InterferenceExperiment(
            outcomes, treatments, clusters
        ).run()

        assert isinstance(result, InterferenceResult)
        assert result.significant is True
        assert result.ate > 0
        assert result.pvalue < 0.05

    def test_no_effect_not_significant(self, rng):
        """No treatment effect should yield non-significant result."""
        n_clusters = 20
        n_per = 20
        outcomes, treatments, clusters = [], [], []
        for c in range(n_clusters):
            for i in range(n_per):
                t = 1 if i < n_per // 2 else 0
                y = rng.normal(10, 1)
                outcomes.append(y)
                treatments.append(t)
                clusters.append(c)

        result = InterferenceExperiment(
            outcomes, treatments, clusters
        ).run()

        assert result.pvalue > 0.01

    def test_design_effect_greater_than_one_with_correlation(self, rng):
        """Design effect should exceed 1 with correlated clusters."""
        n_clusters = 30
        n_per = 20
        outcomes, treatments, clusters = [], [], []
        for c in range(n_clusters):
            # Add a large cluster-level random effect to create correlation
            cluster_shift = rng.normal(0, 5)
            for i in range(n_per):
                t = 1 if i < n_per // 2 else 0
                y = rng.normal(10 + cluster_shift + t * 0.5, 0.5)
                outcomes.append(y)
                treatments.append(t)
                clusters.append(c)

        result = InterferenceExperiment(
            outcomes, treatments, clusters
        ).run()

        assert result.design_effect > 1.0

    def test_matches_naive_when_no_interference(self, rng):
        """Design effect should be near 1 when clusters are homogeneous."""
        n_clusters = 30
        n_per = 20
        outcomes, treatments, clusters = [], [], []
        for c in range(n_clusters):
            for i in range(n_per):
                t = 1 if i < n_per // 2 else 0
                y = rng.normal(10 + t * 1.0, 2)
                outcomes.append(y)
                treatments.append(t)
                clusters.append(c)

        result = InterferenceExperiment(
            outcomes, treatments, clusters
        ).run()

        # Design effect should be close to 1 (no cluster correlation)
        assert result.design_effect < 3.0

    def test_n_clusters_correct(self, rng):
        """n_clusters should match the input."""
        n_clusters = 15
        n_per = 10
        outcomes, treatments, clusters = [], [], []
        for c in range(n_clusters):
            for i in range(n_per):
                t = 1 if i < n_per // 2 else 0
                outcomes.append(rng.normal(10, 1))
                treatments.append(t)
                clusters.append(c)

        result = InterferenceExperiment(
            outcomes, treatments, clusters
        ).run()

        assert result.n_clusters == n_clusters

    def test_ci_contains_ate(self, rng):
        """CI should contain the point estimate."""
        n_clusters = 20
        n_per = 20
        outcomes, treatments, clusters = [], [], []
        for c in range(n_clusters):
            for i in range(n_per):
                t = 1 if i < n_per // 2 else 0
                outcomes.append(rng.normal(10 + t * 1.0, 1))
                treatments.append(t)
                clusters.append(c)

        result = InterferenceExperiment(
            outcomes, treatments, clusters
        ).run()

        assert result.ci_lower <= result.ate <= result.ci_upper

    def test_to_dict(self, rng):
        """Result should serialise to a dictionary."""
        outcomes = [10, 11, 10, 12, 10, 11, 10, 12, 10, 11, 10, 12]
        treatments = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        clusters = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

        result = InterferenceExperiment(
            outcomes, treatments, clusters
        ).run()

        d = result.to_dict()
        assert isinstance(d, dict)
        assert "ate" in d
        assert "design_effect" in d

    def test_repr(self, rng):
        """Result __repr__ should be a string."""
        outcomes = [10, 11, 10, 12, 10, 11, 10, 12, 10, 11, 10, 12]
        treatments = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        clusters = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

        result = InterferenceExperiment(
            outcomes, treatments, clusters
        ).run()

        assert "InterferenceResult" in repr(result)


class TestInterferenceValidation:
    """Validation and error handling tests."""

    def test_non_binary_treatments_raises(self):
        """Non-binary treatments should raise ValueError."""
        with pytest.raises(ValueError, match="must contain only 0 and 1"):
            InterferenceExperiment(
                [1, 2, 3, 4], [0, 1, 2, 3], [0, 0, 1, 1]
            )

    def test_mismatched_lengths_raises(self):
        """Mismatched array lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            InterferenceExperiment(
                [1, 2, 3], [0, 1], [0, 0, 1]
            )

    def test_single_cluster_raises(self):
        """Single cluster should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2 clusters"):
            InterferenceExperiment(
                [1, 2, 3, 4], [0, 1, 0, 1], [0, 0, 0, 0]
            )

    def test_invalid_alpha_raises(self):
        """Alpha outside (0, 1) should raise ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            InterferenceExperiment(
                [1, 2, 3, 4], [0, 1, 0, 1], [0, 0, 1, 1],
                alpha=1.5,
            )

    def test_clusters_with_single_arm_handled(self, rng):
        """Clusters with only one arm should be skipped gracefully."""
        # All units in cluster 0 are control, cluster 1 are treatment
        # Clusters 2, 3 have both
        outcomes = [10, 10, 11, 11, 10, 11, 10, 11]
        treatments = [0, 0, 1, 1, 0, 1, 0, 1]
        clusters = [0, 0, 1, 1, 2, 2, 3, 3]

        result = InterferenceExperiment(
            outcomes, treatments, clusters
        ).run()

        assert isinstance(result, InterferenceResult)

    def test_2d_clusters_rejected(self):
        """Line 90: 2-D clusters array should raise ValueError."""
        with pytest.raises(ValueError, match="1-D"):
            InterferenceExperiment(
                [1, 2, 3, 4], [0, 1, 0, 1],
                np.zeros((4, 2)),
            )

    def test_icc_single_cluster_returns_zero(self, rng):
        """Line 135: ICC with <2 clusters returns 0.0."""
        values = rng.normal(0, 1, 20)
        clusters = np.zeros(20, dtype=int)
        icc = InterferenceExperiment._compute_icc(values, clusters)
        assert icc == 0.0

    def test_icc_zero_variance_returns_zero(self):
        """Line 166: ICC denom<=0 when all values identical across clusters."""
        values = np.array([5.0, 5.0, 5.0, 5.0])
        clusters = np.array([0, 0, 1, 1])
        icc = InterferenceExperiment._compute_icc(values, clusters)
        assert icc == 0.0

    def test_too_few_valid_clusters_raises(self):
        """Line 216: <2 clusters with both arms raises ValueError."""
        # Only cluster 0 has both arms, clusters 1 and 2 are single-arm
        outcomes = [10, 11, 10, 10, 11, 11]
        treatments = [0, 1, 0, 0, 1, 1]
        clusters = [0, 0, 1, 1, 2, 2]
        with pytest.raises(ValueError, match="at least 2 clusters"):
            InterferenceExperiment(outcomes, treatments, clusters).run()

    def test_zero_se_branch(self):
        """Lines 250-252: when all cluster effects identical, se==0."""
        # Create clusters where each has identical trt-ctrl diff
        outcomes = [5.0, 7.0, 5.0, 7.0, 5.0, 7.0, 5.0, 7.0]
        treatments = [0, 1, 0, 1, 0, 1, 0, 1]
        clusters = [0, 0, 1, 1, 2, 2, 3, 3]
        result = InterferenceExperiment(outcomes, treatments, clusters).run()
        # All cluster effects are 2.0, so se==0
        assert result.pvalue == 0.0 or result.se == 0.0
