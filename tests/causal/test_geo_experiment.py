"""Tests for GeoExperiment."""

from __future__ import annotations

import numpy as np
import pytest

from splita.causal.geo_experiment import GeoExperiment


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestGeoBasic:
    """Basic functionality tests."""

    def test_known_effect(self, rng):
        """Should detect a known positive incremental effect."""
        n_t, n_c, T_pre, T_post = 5, 8, 30, 10
        base = 100
        trt_pre = rng.normal(base, 5, (n_t, T_pre))
        ctrl_pre = rng.normal(base, 5, (n_c, T_pre))
        ctrl_post = rng.normal(base, 5, (n_c, T_post))
        trt_post = rng.normal(base + 10, 5, (n_t, T_post))

        r = GeoExperiment().fit(trt_pre, trt_post, ctrl_pre, ctrl_post)
        assert r.incremental_effect > 0
        assert r.n_treated_regions == n_t
        assert r.n_control_regions == n_c

    def test_no_effect(self, rng):
        """No effect should yield CI crossing zero."""
        n_t, n_c, T_pre, T_post = 4, 6, 20, 10
        trt_pre = rng.normal(100, 5, (n_t, T_pre))
        ctrl_pre = rng.normal(100, 5, (n_c, T_pre))
        trt_post = rng.normal(100, 5, (n_t, T_post))
        ctrl_post = rng.normal(100, 5, (n_c, T_post))

        r = GeoExperiment().fit(trt_pre, trt_post, ctrl_pre, ctrl_post)
        # CI should include 0
        assert r.ci_lower < 5 and r.ci_upper > -5

    def test_pre_rmse_low(self, rng):
        """Pre-RMSE should be low with similar pre-period data."""
        n_t, n_c, T_pre, T_post = 3, 5, 20, 10
        shared = rng.normal(100, 5, T_pre)
        trt_pre = np.tile(shared, (n_t, 1)) + rng.normal(0, 1, (n_t, T_pre))
        ctrl_pre = np.tile(shared, (n_c, 1)) + rng.normal(0, 1, (n_c, T_pre))
        trt_post = rng.normal(100, 5, (n_t, T_post))
        ctrl_post = rng.normal(100, 5, (n_c, T_post))

        r = GeoExperiment().fit(trt_pre, trt_post, ctrl_pre, ctrl_post)
        assert r.pre_rmse < 20

    def test_single_region(self, rng):
        """Should work with a single treated region (1-D input)."""
        T_pre, T_post = 20, 10
        trt_pre = rng.normal(100, 5, T_pre)
        trt_post = rng.normal(110, 5, T_post)
        ctrl_pre = rng.normal(100, 5, (3, T_pre))
        ctrl_post = rng.normal(100, 5, (3, T_post))

        r = GeoExperiment().fit(trt_pre, trt_post, ctrl_pre, ctrl_post)
        assert r.n_treated_regions == 1

    def test_ci_order(self, rng):
        """Lower CI should be less than upper CI."""
        n_t, n_c = 3, 5
        trt_pre = rng.normal(100, 5, (n_t, 20))
        trt_post = rng.normal(105, 5, (n_t, 10))
        ctrl_pre = rng.normal(100, 5, (n_c, 20))
        ctrl_post = rng.normal(100, 5, (n_c, 10))

        r = GeoExperiment().fit(trt_pre, trt_post, ctrl_pre, ctrl_post)
        assert r.ci_lower <= r.ci_upper

    def test_to_dict(self, rng):
        trt_pre = rng.normal(100, 5, (2, 10))
        trt_post = rng.normal(100, 5, (2, 5))
        ctrl_pre = rng.normal(100, 5, (3, 10))
        ctrl_post = rng.normal(100, 5, (3, 5))
        r = GeoExperiment().fit(trt_pre, trt_post, ctrl_pre, ctrl_post)
        d = r.to_dict()
        assert isinstance(d, dict)
        assert "incremental_effect" in d

    def test_repr(self, rng):
        trt_pre = rng.normal(100, 5, (2, 10))
        trt_post = rng.normal(100, 5, (2, 5))
        ctrl_pre = rng.normal(100, 5, (3, 10))
        ctrl_post = rng.normal(100, 5, (3, 5))
        r = GeoExperiment().fit(trt_pre, trt_post, ctrl_pre, ctrl_post)
        assert "GeoResult" in repr(r)


class TestGeoValidation:
    """Validation and error tests."""

    def test_alpha_out_of_range(self):
        with pytest.raises(ValueError, match="alpha"):
            GeoExperiment(alpha=1.5)

    def test_n_bootstrap_too_low(self):
        with pytest.raises(ValueError, match="n_bootstrap"):
            GeoExperiment(n_bootstrap=10)

    def test_mismatched_pre_periods(self, rng):
        with pytest.raises(ValueError, match="Pre-period length"):
            GeoExperiment().fit(
                rng.normal(0, 1, (2, 10)),
                rng.normal(0, 1, (2, 5)),
                rng.normal(0, 1, (3, 8)),  # different pre-period
                rng.normal(0, 1, (3, 5)),
            )

    def test_pre_period_too_short(self, rng):
        with pytest.raises(ValueError, match="at least 2"):
            GeoExperiment().fit(
                rng.normal(0, 1, (2, 1)),
                rng.normal(0, 1, (2, 5)),
                rng.normal(0, 1, (3, 1)),
                rng.normal(0, 1, (3, 5)),
            )
