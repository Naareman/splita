"""Tests for InExperimentVR."""

from __future__ import annotations

import numpy as np
import pytest

from splita.variance.inex import InExperimentVR


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


@pytest.fixture()
def data(rng):
    n = 200
    base = rng.normal(10, 2, n)
    ctrl = base[:100] + rng.normal(0, 1, 100)
    trt = base[100:] + 0.5 + rng.normal(0, 1, 100)
    cov_ctrl = base[:100] + rng.normal(0, 0.5, 100)
    cov_trt = base[100:] + rng.normal(0, 0.5, 100)
    return ctrl, trt, cov_ctrl, cov_trt


class TestFitTransform:
    def test_basic_fit_transform(self, data):
        ctrl, trt, cov_ctrl, cov_trt = data
        vr = InExperimentVR()
        ctrl_adj, trt_adj = vr.fit_transform(ctrl, trt, cov_ctrl, cov_trt)
        assert len(ctrl_adj) == len(ctrl)
        assert len(trt_adj) == len(trt)

    def test_stores_theta(self, data):
        ctrl, trt, cov_ctrl, cov_trt = data
        vr = InExperimentVR()
        vr.fit_transform(ctrl, trt, cov_ctrl, cov_trt)
        assert hasattr(vr, "theta_")
        assert isinstance(vr.theta_, float)

    def test_stores_variance_reduction(self, data):
        ctrl, trt, cov_ctrl, cov_trt = data
        vr = InExperimentVR()
        vr.fit_transform(ctrl, trt, cov_ctrl, cov_trt)
        assert 0.0 <= vr.variance_reduction_ <= 1.0

    def test_reduces_variance(self, data):
        ctrl, trt, cov_ctrl, cov_trt = data
        vr = InExperimentVR()
        ctrl_adj, trt_adj = vr.fit_transform(ctrl, trt, cov_ctrl, cov_trt)
        # Adjusted variance should be lower
        combined_orig = np.var(np.concatenate([ctrl, trt]))
        combined_adj = np.var(np.concatenate([ctrl_adj, trt_adj]))
        assert combined_adj < combined_orig

    def test_preserves_mean_difference(self, data):
        ctrl, trt, cov_ctrl, cov_trt = data
        vr = InExperimentVR()
        ctrl_adj, trt_adj = vr.fit_transform(ctrl, trt, cov_ctrl, cov_trt)
        orig_diff = np.mean(trt) - np.mean(ctrl)
        adj_diff = np.mean(trt_adj) - np.mean(ctrl_adj)
        assert abs(orig_diff - adj_diff) < 0.5  # approximately preserved


class TestValidation:
    def test_mismatched_lengths(self, rng):
        with pytest.raises(ValueError, match="same length"):
            vr = InExperimentVR()
            vr.fit(
                rng.normal(0, 1, 50),
                rng.normal(0, 1, 50),
                rng.normal(0, 1, 40),  # wrong length
                rng.normal(0, 1, 50),
            )

    def test_too_short_array(self, rng):
        with pytest.raises(ValueError, match="at least"):
            vr = InExperimentVR()
            vr.fit([1.0], [2.0, 3.0], [1.0], [2.0, 3.0])

    def test_zero_variance_covariate(self, rng):
        with pytest.raises(ValueError, match="zero variance"):
            vr = InExperimentVR()
            vr.fit(
                rng.normal(0, 1, 50),
                rng.normal(0, 1, 50),
                np.ones(50),
                np.ones(50),
            )

    def test_transform_before_fit(self, rng):
        with pytest.raises(RuntimeError, match="fitted"):
            vr = InExperimentVR()
            vr.transform(
                rng.normal(0, 1, 50),
                rng.normal(0, 1, 50),
                rng.normal(0, 1, 50),
                rng.normal(0, 1, 50),
            )

    def test_non_array_input(self):
        with pytest.raises(TypeError, match="array-like"):
            vr = InExperimentVR()
            vr.fit("not_array", [1.0, 2.0], [1.0, 2.0], [1.0, 2.0])


class TestFitThenTransform:
    def test_separate_fit_transform(self, data):
        ctrl, trt, cov_ctrl, cov_trt = data
        vr = InExperimentVR()
        vr.fit(ctrl, trt, cov_ctrl, cov_trt)
        ctrl_adj, trt_adj = vr.transform(ctrl, trt, cov_ctrl, cov_trt)
        assert len(ctrl_adj) == len(ctrl)

    def test_fit_returns_self(self, data):
        ctrl, trt, cov_ctrl, cov_trt = data
        vr = InExperimentVR()
        result = vr.fit(ctrl, trt, cov_ctrl, cov_trt)
        assert result is vr


class TestLowCorrelation:
    def test_low_correlation_warning(self, rng):
        ctrl = rng.normal(0, 1, 100)
        trt = rng.normal(0, 1, 100)
        cov_ctrl = rng.normal(0, 1, 100)  # uncorrelated
        cov_trt = rng.normal(0, 1, 100)
        vr = InExperimentVR()
        with pytest.warns(RuntimeWarning, match="Low correlation"):
            vr.fit(ctrl, trt, cov_ctrl, cov_trt)
