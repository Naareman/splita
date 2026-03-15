"""Tests for CarryoverDetector (carryover effect detection)."""

from __future__ import annotations

import numpy as np
import pytest

from splita.diagnostics.carryover import CarryoverDetector


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestCarryoverBasic:
    """Basic functionality tests."""

    def test_no_carryover(self, rng):
        """When control is stable, no carryover should be detected."""
        n = 200
        ctrl_pre = rng.normal(10, 1, n)
        ctrl_post = rng.normal(10, 1, n)
        trt_pre = rng.normal(10, 1, n)
        trt_post = rng.normal(12, 1, n)

        r = CarryoverDetector().detect(ctrl_pre, ctrl_post, trt_pre, trt_post)
        assert r.has_carryover is False
        assert r.control_change_pvalue > 0.05

    def test_carryover_detected(self, rng):
        """When control shifts post-experiment, carryover should be detected."""
        n = 200
        ctrl_pre = rng.normal(10, 1, n)
        ctrl_post = rng.normal(12, 1, n)  # control shifted!
        trt_pre = rng.normal(10, 1, n)
        trt_post = rng.normal(12, 1, n)

        r = CarryoverDetector().detect(ctrl_pre, ctrl_post, trt_pre, trt_post)
        assert r.has_carryover is True
        assert r.control_change_pvalue < 0.05

    def test_control_means_correct(self, rng):
        """Control pre/post means should be correctly reported."""
        n = 200
        ctrl_pre = rng.normal(10, 1, n)
        ctrl_post = rng.normal(15, 1, n)
        trt_pre = rng.normal(10, 1, n)
        trt_post = rng.normal(12, 1, n)

        r = CarryoverDetector().detect(ctrl_pre, ctrl_post, trt_pre, trt_post)
        assert abs(r.control_pre_mean - 10.0) < 0.5
        assert abs(r.control_post_mean - 15.0) < 0.5

    def test_message_no_carryover(self, rng):
        """Message should say 'No carryover' when stable."""
        n = 200
        ctrl_pre = rng.normal(10, 1, n)
        ctrl_post = rng.normal(10, 1, n)
        trt_pre = rng.normal(10, 1, n)
        trt_post = rng.normal(12, 1, n)

        r = CarryoverDetector().detect(ctrl_pre, ctrl_post, trt_pre, trt_post)
        assert "No carryover" in r.message

    def test_message_carryover(self, rng):
        """Message should say 'Carryover detected' when found."""
        n = 200
        ctrl_pre = rng.normal(10, 1, n)
        ctrl_post = rng.normal(13, 1, n)
        trt_pre = rng.normal(10, 1, n)
        trt_post = rng.normal(13, 1, n)

        r = CarryoverDetector().detect(ctrl_pre, ctrl_post, trt_pre, trt_post)
        assert "Carryover detected" in r.message

    def test_custom_alpha(self, rng):
        """More strict alpha should detect fewer carryover effects."""
        n = 200
        ctrl_pre = rng.normal(10, 1, n)
        ctrl_post = rng.normal(10.3, 1, n)  # small shift
        trt_pre = rng.normal(10, 1, n)
        trt_post = rng.normal(12, 1, n)

        r_loose = CarryoverDetector(alpha=0.10).detect(
            ctrl_pre, ctrl_post, trt_pre, trt_post
        )
        r_strict = CarryoverDetector(alpha=0.001).detect(
            ctrl_pre, ctrl_post, trt_pre, trt_post
        )

        # Same p-value, different thresholds
        assert abs(r_loose.control_change_pvalue - r_strict.control_change_pvalue) < 0.001

    def test_washout_periods_param(self):
        """washout_periods parameter should be accepted."""
        d = CarryoverDetector(washout_periods=5)
        assert d._washout_periods == 5

    def test_pvalue_range(self, rng):
        """p-value should be in [0, 1]."""
        n = 100
        ctrl_pre = rng.normal(10, 1, n)
        ctrl_post = rng.normal(10, 1, n)
        trt_pre = rng.normal(10, 1, n)
        trt_post = rng.normal(12, 1, n)

        r = CarryoverDetector().detect(ctrl_pre, ctrl_post, trt_pre, trt_post)
        assert 0.0 <= r.control_change_pvalue <= 1.0


class TestCarryoverValidation:
    """Input validation tests."""

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            CarryoverDetector(alpha=1.5)

    def test_invalid_washout(self):
        with pytest.raises((ValueError, TypeError)):
            CarryoverDetector(washout_periods=-1)

    def test_too_few_samples(self, rng):
        with pytest.raises(ValueError, match="at least"):
            CarryoverDetector().detect([1.0], [2.0], [1.0], [2.0])

    def test_non_array_input(self):
        with pytest.raises(TypeError, match="array-like"):
            CarryoverDetector().detect("not_array", [1, 2, 3], [1, 2, 3], [1, 2, 3])


class TestCarryoverResult:
    """Result object tests."""

    def test_to_dict(self, rng):
        n = 100
        r = CarryoverDetector().detect(
            rng.normal(10, 1, n),
            rng.normal(10, 1, n),
            rng.normal(10, 1, n),
            rng.normal(12, 1, n),
        )
        d = r.to_dict()
        assert isinstance(d, dict)
        assert "has_carryover" in d
        assert "message" in d

    def test_repr(self, rng):
        n = 100
        r = CarryoverDetector().detect(
            rng.normal(10, 1, n),
            rng.normal(10, 1, n),
            rng.normal(10, 1, n),
            rng.normal(12, 1, n),
        )
        assert "CarryoverResult" in repr(r)
