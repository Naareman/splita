"""Tests for splita.check — pre-analysis health report."""

from __future__ import annotations

import numpy as np
import pytest

from splita.check import check
from splita._types import CheckResult


# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def balanced_conversion(rng):
    ctrl = rng.binomial(1, 0.10, 5000)
    trt = rng.binomial(1, 0.12, 5000)
    return ctrl, trt


@pytest.fixture
def continuous_data(rng):
    ctrl = rng.normal(10, 2, 5000)
    trt = rng.normal(10.5, 2, 5000)
    return ctrl, trt


# ─── Basic Functionality ─────────────────────────────────────────────


class TestCheckBasic:
    def test_returns_check_result(self, balanced_conversion):
        ctrl, trt = balanced_conversion
        result = check(ctrl, trt)
        assert isinstance(result, CheckResult)

    def test_srm_passed_for_balanced_data(self, balanced_conversion):
        ctrl, trt = balanced_conversion
        result = check(ctrl, trt)
        assert result.srm_passed is True

    def test_has_checks_list(self, balanced_conversion):
        ctrl, trt = balanced_conversion
        result = check(ctrl, trt)
        assert isinstance(result.checks, list)
        assert len(result.checks) >= 4  # srm, outliers, data_quality, skewness, power

    def test_has_recommendations(self, balanced_conversion):
        ctrl, trt = balanced_conversion
        result = check(ctrl, trt)
        assert isinstance(result.recommendations, list)
        assert len(result.recommendations) >= 1

    def test_all_passed_healthy_data(self, balanced_conversion):
        ctrl, trt = balanced_conversion
        result = check(ctrl, trt)
        # At minimum srm and data_quality should pass
        assert result.srm_passed is True

    def test_continuous_data(self, continuous_data):
        ctrl, trt = continuous_data
        result = check(ctrl, trt)
        assert isinstance(result, CheckResult)
        assert result.srm_passed is True


# ─── Outlier Detection ───────────────────────────────────────────────


class TestCheckOutliers:
    def test_detects_outliers(self, rng):
        ctrl = rng.normal(10, 1, 1000)
        trt = rng.normal(10, 1, 1000)
        # Inject extreme outliers
        ctrl[:50] = 1000.0
        trt[:50] = -500.0
        result = check(ctrl, trt)
        assert result.has_outliers is True

    def test_no_outliers_clean_data(self, balanced_conversion):
        ctrl, trt = balanced_conversion
        result = check(ctrl, trt)
        # Binary data has no outliers by IQR
        assert result.has_outliers is False


# ─── Flicker Detection ──────────────────────────────────────────────


class TestCheckFlicker:
    def test_flicker_detection_clean(self, balanced_conversion):
        ctrl, trt = balanced_conversion
        n = len(ctrl) + len(trt)
        user_ids = np.arange(n)
        variants = np.array([0] * len(ctrl) + [1] * len(trt))
        result = check(ctrl, trt, user_ids=user_ids, variant_assignments=variants)
        assert result.flicker_rate == 0.0

    def test_flicker_detection_with_flickers(self, rng):
        ctrl = rng.binomial(1, 0.10, 100)
        trt = rng.binomial(1, 0.12, 100)
        # Users 0-4 appear in both variants
        user_ids = np.concatenate([
            np.arange(100),
            np.arange(5),  # first 5 users also in treatment
            np.arange(5, 100),
        ])
        variants = np.array([0] * 100 + [1] * 5 + [1] * 95)
        combined_ctrl = np.concatenate([ctrl, ctrl[:5]])
        combined_trt = np.concatenate([trt[:5], trt[5:]])
        # We need combined arrays for the check function
        result = check(ctrl, trt, user_ids=user_ids, variant_assignments=variants)
        assert result.flicker_rate is not None
        assert result.flicker_rate > 0

    def test_flicker_missing_variant_assignments(self, balanced_conversion):
        ctrl, trt = balanced_conversion
        user_ids = np.arange(len(ctrl) + len(trt))
        result = check(ctrl, trt, user_ids=user_ids)
        # Should flag missing variant_assignments
        flicker_checks = [c for c in result.checks if c["name"] == "flicker"]
        assert len(flicker_checks) == 1
        assert flicker_checks[0]["passed"] is False


# ─── Power Assessment ────────────────────────────────────────────────


class TestCheckPower:
    def test_high_power_large_effect(self, rng):
        ctrl = rng.binomial(1, 0.10, 10000)
        trt = rng.binomial(1, 0.20, 10000)
        result = check(ctrl, trt)
        assert result.is_powered is True

    def test_low_power_small_sample(self, rng):
        ctrl = rng.binomial(1, 0.10, 20)
        trt = rng.binomial(1, 0.11, 20)
        result = check(ctrl, trt)
        assert result.is_powered is False


# ─── Covariate Balance ──────────────────────────────────────────────


class TestCheckCovariate:
    def test_balanced_segments(self, rng):
        ctrl = rng.normal(10, 2, 1000)
        trt = rng.normal(10, 2, 1000)
        # Balanced segments
        segments = np.array(["A"] * 500 + ["B"] * 500 + ["A"] * 500 + ["B"] * 500)
        result = check(ctrl, trt, segments=segments)
        balance_checks = [c for c in result.checks if c["name"] == "covariate_balance"]
        assert len(balance_checks) == 1
        assert balance_checks[0]["passed"] is True

    def test_imbalanced_segments(self, rng):
        ctrl = rng.normal(10, 2, 1000)
        trt = rng.normal(10, 2, 1000)
        # Very imbalanced
        segments = np.array(["A"] * 900 + ["B"] * 100 + ["A"] * 100 + ["B"] * 900)
        result = check(ctrl, trt, segments=segments)
        balance_checks = [c for c in result.checks if c["name"] == "covariate_balance"]
        assert len(balance_checks) == 1
        assert balance_checks[0]["passed"] is False


# ─── Serialization ──────────────────────────────────────────────────


class TestCheckSerialization:
    def test_to_dict(self, balanced_conversion):
        ctrl, trt = balanced_conversion
        result = check(ctrl, trt)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "srm_passed" in d
        assert "checks" in d

    def test_repr(self, balanced_conversion):
        ctrl, trt = balanced_conversion
        result = check(ctrl, trt)
        text = repr(result)
        assert "CheckResult" in text
