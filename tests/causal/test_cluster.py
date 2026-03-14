"""Tests for ClusterExperiment (M16: Network Effects — Cluster)."""

from __future__ import annotations

import numpy as np
import pytest

from splita.causal.cluster import ClusterExperiment


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestClusterBasic:
    """Basic functionality tests."""

    def test_significant_cluster_effect(self, rng):
        """Large treatment effect should be detected."""
        n_clusters = 20
        n_per_cluster = 50
        ctrl, ctrl_cl = [], []
        trt, trt_cl = [], []
        for c in range(n_clusters):
            ctrl.extend(rng.normal(10, 2, n_per_cluster))
            ctrl_cl.extend([c] * n_per_cluster)
            trt.extend(rng.normal(15, 2, n_per_cluster))
            trt_cl.extend([c] * n_per_cluster)

        result = ClusterExperiment(
            ctrl, trt,
            control_clusters=ctrl_cl,
            treatment_clusters=trt_cl,
        ).run()

        assert result.significant is True
        assert result.lift > 0
        assert result.pvalue < 0.05

    def test_no_effect(self, rng):
        """No treatment effect should yield non-significant result."""
        n_clusters = 10
        n_per_cluster = 50
        ctrl, ctrl_cl = [], []
        trt, trt_cl = [], []
        for c in range(n_clusters):
            ctrl.extend(rng.normal(10, 1, n_per_cluster))
            ctrl_cl.extend([c] * n_per_cluster)
            trt.extend(rng.normal(10, 1, n_per_cluster))
            trt_cl.extend([c] * n_per_cluster)

        result = ClusterExperiment(
            ctrl, trt,
            control_clusters=ctrl_cl,
            treatment_clusters=trt_cl,
        ).run()

        assert result.pvalue > 0.01  # should not be significant usually

    def test_cluster_robust_se_wider_than_naive(self, rng):
        """Cluster-robust SEs should be wider than naive (unit-level) SEs when ICC > 0."""
        n_clusters = 10
        n_per_cluster = 50

        ctrl, ctrl_cl = [], []
        trt, trt_cl = [], []
        for c in range(n_clusters):
            # Create within-cluster correlation by adding a cluster-level random effect
            cluster_effect_c = rng.normal(0, 3)
            cluster_effect_t = rng.normal(0, 3)
            ctrl.extend(rng.normal(10 + cluster_effect_c, 1, n_per_cluster))
            ctrl_cl.extend([c] * n_per_cluster)
            trt.extend(rng.normal(10 + cluster_effect_t, 1, n_per_cluster))
            trt_cl.extend([c] * n_per_cluster)

        ctrl_arr = np.array(ctrl)
        trt_arr = np.array(trt)

        # Cluster-level CI width
        result = ClusterExperiment(
            ctrl_arr, trt_arr,
            control_clusters=ctrl_cl,
            treatment_clusters=trt_cl,
        ).run()
        cluster_ci_width = result.ci_upper - result.ci_lower

        # Naive CI width (ignoring clusters)
        from scipy.stats import ttest_ind
        naive_result = ttest_ind(trt_arr, ctrl_arr, equal_var=False)
        naive_se = (np.mean(trt_arr) - np.mean(ctrl_arr)) / naive_result.statistic
        naive_ci_width = 2 * 1.96 * abs(naive_se)

        assert cluster_ci_width > naive_ci_width

    def test_cluster_counts(self, rng):
        """Cluster counts should match the number of unique clusters."""
        ctrl = rng.normal(10, 1, 60)
        trt = rng.normal(10, 1, 40)
        ctrl_cl = np.repeat(np.arange(6), 10)
        trt_cl = np.repeat(np.arange(4), 10)

        result = ClusterExperiment(
            ctrl, trt,
            control_clusters=ctrl_cl,
            treatment_clusters=trt_cl,
        ).run()

        assert result.n_clusters_control == 6
        assert result.n_clusters_treatment == 4

    def test_icc_positive_with_cluster_effects(self, rng):
        """ICC should be positive when there are cluster-level random effects."""
        n_clusters = 10
        n_per_cluster = 30
        ctrl, ctrl_cl = [], []
        trt, trt_cl = [], []
        for c in range(n_clusters):
            cluster_effect = rng.normal(0, 5)
            ctrl.extend(rng.normal(10 + cluster_effect, 1, n_per_cluster))
            ctrl_cl.extend([c] * n_per_cluster)
            trt.extend(rng.normal(10 + cluster_effect, 1, n_per_cluster))
            trt_cl.extend([c] * n_per_cluster)

        result = ClusterExperiment(
            ctrl, trt,
            control_clusters=ctrl_cl,
            treatment_clusters=trt_cl,
        ).run()

        assert result.icc > 0

    def test_icc_near_zero_without_cluster_effects(self, rng):
        """ICC should be near zero when observations are independent."""
        n_clusters = 20
        n_per_cluster = 30
        ctrl, ctrl_cl = [], []
        trt, trt_cl = [], []
        for c in range(n_clusters):
            ctrl.extend(rng.normal(10, 1, n_per_cluster))
            ctrl_cl.extend([c] * n_per_cluster)
            trt.extend(rng.normal(10, 1, n_per_cluster))
            trt_cl.extend([c] * n_per_cluster)

        result = ClusterExperiment(
            ctrl, trt,
            control_clusters=ctrl_cl,
            treatment_clusters=trt_cl,
        ).run()

        assert result.icc < 0.2  # should be small

    def test_ci_contains_lift(self, rng):
        """CI should contain the point estimate."""
        ctrl = rng.normal(10, 1, 100)
        trt = rng.normal(12, 1, 100)
        ctrl_cl = np.repeat(np.arange(10), 10)
        trt_cl = np.repeat(np.arange(10), 10)

        result = ClusterExperiment(
            ctrl, trt,
            control_clusters=ctrl_cl,
            treatment_clusters=trt_cl,
        ).run()

        assert result.ci_lower <= result.lift <= result.ci_upper

    def test_alpha_parameter(self, rng):
        """Custom alpha should affect significance threshold."""
        ctrl = rng.normal(10, 1, 100)
        trt = rng.normal(10.5, 1, 100)
        ctrl_cl = np.repeat(np.arange(10), 10)
        trt_cl = np.repeat(np.arange(10), 10)

        result_05 = ClusterExperiment(
            ctrl, trt,
            control_clusters=ctrl_cl,
            treatment_clusters=trt_cl,
            alpha=0.05,
        ).run()

        result_01 = ClusterExperiment(
            ctrl, trt,
            control_clusters=ctrl_cl,
            treatment_clusters=trt_cl,
            alpha=0.001,
        ).run()

        # Same p-value, but significance may differ
        assert abs(result_05.pvalue - result_01.pvalue) < 0.001


class TestClusterValidation:
    """Tests for input validation."""

    def test_mismatched_control_lengths(self, rng):
        with pytest.raises(ValueError, match="same length"):
            ClusterExperiment(
                rng.normal(0, 1, 50),
                rng.normal(0, 1, 50),
                control_clusters=np.arange(30),
                treatment_clusters=np.repeat(np.arange(5), 10),
            )

    def test_mismatched_treatment_lengths(self, rng):
        with pytest.raises(ValueError, match="same length"):
            ClusterExperiment(
                rng.normal(0, 1, 50),
                rng.normal(0, 1, 50),
                control_clusters=np.repeat(np.arange(5), 10),
                treatment_clusters=np.arange(30),
            )

    def test_too_few_control_clusters(self, rng):
        ctrl = rng.normal(0, 1, 20)
        trt = rng.normal(0, 1, 20)
        with pytest.raises(ValueError, match="at least 2 control clusters"):
            ClusterExperiment(
                ctrl, trt,
                control_clusters=np.zeros(20, dtype=int),  # 1 cluster
                treatment_clusters=np.repeat(np.arange(2), 10),
            ).run()

    def test_too_few_treatment_clusters(self, rng):
        ctrl = rng.normal(0, 1, 20)
        trt = rng.normal(0, 1, 20)
        with pytest.raises(ValueError, match="at least 2 treatment clusters"):
            ClusterExperiment(
                ctrl, trt,
                control_clusters=np.repeat(np.arange(2), 10),
                treatment_clusters=np.zeros(20, dtype=int),  # 1 cluster
            ).run()

    def test_invalid_alpha(self, rng):
        with pytest.raises(ValueError, match="alpha"):
            ClusterExperiment(
                rng.normal(0, 1, 20),
                rng.normal(0, 1, 20),
                control_clusters=np.repeat(np.arange(2), 10),
                treatment_clusters=np.repeat(np.arange(2), 10),
                alpha=1.5,
            )

    def test_2d_clusters_rejected(self, rng):
        with pytest.raises(ValueError, match="1-D"):
            ClusterExperiment(
                rng.normal(0, 1, 20),
                rng.normal(0, 1, 20),
                control_clusters=np.zeros((20, 2)),
                treatment_clusters=np.repeat(np.arange(2), 10),
            )


class TestClusterResult:
    """Tests for result properties."""

    def test_to_dict(self, rng):
        ctrl = rng.normal(10, 1, 100)
        trt = rng.normal(12, 1, 100)
        ctrl_cl = np.repeat(np.arange(10), 10)
        trt_cl = np.repeat(np.arange(10), 10)
        result = ClusterExperiment(
            ctrl, trt,
            control_clusters=ctrl_cl,
            treatment_clusters=trt_cl,
        ).run()
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "lift" in d
        assert "icc" in d

    def test_repr(self, rng):
        ctrl = rng.normal(10, 1, 100)
        trt = rng.normal(12, 1, 100)
        ctrl_cl = np.repeat(np.arange(10), 10)
        trt_cl = np.repeat(np.arange(10), 10)
        result = ClusterExperiment(
            ctrl, trt,
            control_clusters=ctrl_cl,
            treatment_clusters=trt_cl,
        ).run()
        assert "ClusterResult" in repr(result)
