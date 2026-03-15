"""Tests for ClusterBootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from splita._types import ClusterBootstrapResult
from splita.variance.cluster_bootstrap import ClusterBootstrap


# ── helpers ──────────────────────────────────────────────────────────


def _make_cluster_data(
    n_clusters: int = 50,
    obs_per_cluster: int = 10,
    effect: float = 1.0,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    ctrl_vals, trt_vals = [], []
    ctrl_ids, trt_ids = [], []

    for c in range(n_clusters):
        mu_c = rng.normal(10, 2)
        ctrl_vals.append(rng.normal(mu_c, 1, obs_per_cluster))
        ctrl_ids.extend([c] * obs_per_cluster)

        mu_t = rng.normal(10 + effect, 2)
        trt_vals.append(rng.normal(mu_t, 1, obs_per_cluster))
        trt_ids.extend([c] * obs_per_cluster)

    return (
        np.concatenate(ctrl_vals),
        np.concatenate(trt_vals),
        np.array(ctrl_ids),
        np.array(trt_ids),
    )


# ── Basic behaviour ──────────────────────────────────────────────────


class TestBasic:
    """Basic ClusterBootstrap behaviour."""

    def test_detects_positive_effect(self):
        """Detects a clear cluster-level effect."""
        ctrl, trt, cc, tc = _make_cluster_data(effect=2.0)
        result = ClusterBootstrap(n_bootstrap=1000, random_state=42).run(ctrl, trt, cc, tc)
        assert isinstance(result, ClusterBootstrapResult)
        assert result.significant
        assert result.ate > 0

    def test_no_effect(self):
        """Zero effect is not significant."""
        ctrl, trt, cc, tc = _make_cluster_data(effect=0.0)
        result = ClusterBootstrap(n_bootstrap=1000, random_state=42).run(ctrl, trt, cc, tc)
        assert not result.significant

    def test_result_frozen(self):
        """Result is frozen."""
        ctrl, trt, cc, tc = _make_cluster_data()
        result = ClusterBootstrap(n_bootstrap=200, random_state=42).run(ctrl, trt, cc, tc)
        with pytest.raises(AttributeError):
            result.ate = 999.0  # type: ignore[misc]

    def test_to_dict(self):
        """Result serializes to dict."""
        ctrl, trt, cc, tc = _make_cluster_data()
        result = ClusterBootstrap(n_bootstrap=200, random_state=42).run(ctrl, trt, cc, tc)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "ate" in d
        assert "n_clusters" in d

    def test_repr(self):
        """Result has string representation."""
        ctrl, trt, cc, tc = _make_cluster_data()
        result = ClusterBootstrap(n_bootstrap=200, random_state=42).run(ctrl, trt, cc, tc)
        s = repr(result)
        assert "ClusterBootstrapResult" in s


# ── Cluster counting ─────────────────────────────────────────────────


class TestClusters:
    """Cluster count correctness."""

    def test_cluster_count(self):
        """Reports correct total cluster count."""
        ctrl, trt, cc, tc = _make_cluster_data(n_clusters=30)
        result = ClusterBootstrap(n_bootstrap=200, random_state=42).run(ctrl, trt, cc, tc)
        assert result.n_clusters == 60  # 30 ctrl + 30 trt

    def test_unequal_clusters(self):
        """Works with different numbers of clusters per group."""
        rng = np.random.default_rng(42)
        ctrl = rng.normal(10, 1, 100)
        trt = rng.normal(11, 1, 150)
        cc = np.repeat(np.arange(20), 5)
        tc = np.repeat(np.arange(30), 5)
        result = ClusterBootstrap(n_bootstrap=200, random_state=42).run(ctrl, trt, cc, tc)
        assert result.n_clusters == 50


# ── CI properties ────────────────────────────────────────────────────


class TestCI:
    """Confidence interval properties."""

    def test_ci_contains_ate(self):
        """CI contains the observed ATE."""
        ctrl, trt, cc, tc = _make_cluster_data(effect=0.5)
        result = ClusterBootstrap(n_bootstrap=1000, random_state=42).run(ctrl, trt, cc, tc)
        assert result.ci_lower <= result.ate <= result.ci_upper

    def test_ci_width_reasonable(self):
        """CI width is positive and finite."""
        ctrl, trt, cc, tc = _make_cluster_data()
        result = ClusterBootstrap(n_bootstrap=500, random_state=42).run(ctrl, trt, cc, tc)
        width = result.ci_upper - result.ci_lower
        assert width > 0
        assert np.isfinite(width)


# ── Reproducibility ──────────────────────────────────────────────────


class TestReproducibility:
    """Reproducibility with random_state."""

    def test_same_seed_same_result(self):
        """Same random_state produces identical results."""
        ctrl, trt, cc, tc = _make_cluster_data()
        r1 = ClusterBootstrap(n_bootstrap=500, random_state=123).run(ctrl, trt, cc, tc)
        r2 = ClusterBootstrap(n_bootstrap=500, random_state=123).run(ctrl, trt, cc, tc)
        np.testing.assert_allclose(r1.ate, r2.ate)
        np.testing.assert_allclose(r1.pvalue, r2.pvalue)


# ── Validation ───────────────────────────────────────────────────────


class TestValidation:
    """Input validation tests."""

    def test_alpha_out_of_range(self):
        """alpha outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            ClusterBootstrap(alpha=1.5)

    def test_n_bootstrap_too_small(self):
        """n_bootstrap < 100 raises ValueError."""
        with pytest.raises(ValueError, match="n_bootstrap"):
            ClusterBootstrap(n_bootstrap=10)

    def test_cluster_length_mismatch(self):
        """Cluster array length must match data."""
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="same length"):
            ClusterBootstrap(n_bootstrap=200, random_state=42).run(
                rng.normal(0, 1, 50),
                rng.normal(0, 1, 50),
                np.array([0] * 30),
                np.array([0] * 50),
            )

    def test_too_few_clusters(self):
        """< 2 clusters raises ValueError."""
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="at least 2 clusters"):
            ClusterBootstrap(n_bootstrap=200, random_state=42).run(
                rng.normal(0, 1, 10),
                rng.normal(0, 1, 10),
                np.array([0] * 10),
                np.array([0] * 5 + [1] * 5),
            )

    def test_clusters_not_array_like(self):
        """Non-array clusters raises TypeError."""
        rng = np.random.default_rng(42)
        with pytest.raises(TypeError, match="array-like"):
            ClusterBootstrap(n_bootstrap=200, random_state=42).run(
                rng.normal(0, 1, 10),
                rng.normal(0, 1, 10),
                42,
                np.array([0] * 10),
            )


# ── E2E scenario ─────────────────────────────────────────────────────


class TestE2E:
    """End-to-end scenario."""

    def test_user_level_ratio_metric(self):
        """Cluster bootstrap on user-level ratio metric (page views per session)."""
        rng = np.random.default_rng(99)
        n_users = 100
        ctrl_vals, trt_vals = [], []
        ctrl_ids, trt_ids = [], []

        for u in range(n_users):
            n_sessions = rng.integers(3, 20)
            ctrl_vals.extend(rng.poisson(5, n_sessions).tolist())
            ctrl_ids.extend([u] * n_sessions)

            n_sessions_t = rng.integers(3, 20)
            trt_vals.extend(rng.poisson(6, n_sessions_t).tolist())
            trt_ids.extend([u] * n_sessions_t)

        result = ClusterBootstrap(n_bootstrap=2000, random_state=42).run(
            np.array(ctrl_vals, dtype=float),
            np.array(trt_vals, dtype=float),
            np.array(ctrl_ids),
            np.array(trt_ids),
        )
        assert result.significant
        assert result.ate > 0
        assert result.n_clusters == 200
