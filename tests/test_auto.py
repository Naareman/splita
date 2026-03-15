"""Tests for splita.auto — zero-config complete analysis."""

from __future__ import annotations

import numpy as np
import pytest

from splita.auto import auto
from splita._types import AutoResult, ExperimentResult, SRMResult


# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def conversion_data(rng):
    ctrl = rng.binomial(1, 0.10, 5000)
    trt = rng.binomial(1, 0.12, 5000)
    return ctrl, trt


@pytest.fixture
def continuous_data(rng):
    ctrl = rng.normal(10, 2, 5000)
    trt = rng.normal(10.5, 2, 5000)
    return ctrl, trt


# ─── Basic Functionality ─────────────────────────────────────────────


class TestAutoBasic:
    def test_returns_auto_result(self, conversion_data):
        ctrl, trt = conversion_data
        result = auto(ctrl, trt)
        assert isinstance(result, AutoResult)

    def test_primary_result_is_experiment_result(self, conversion_data):
        ctrl, trt = conversion_data
        result = auto(ctrl, trt)
        assert isinstance(result.primary_result, ExperimentResult)

    def test_srm_result_is_srm_result(self, conversion_data):
        ctrl, trt = conversion_data
        result = auto(ctrl, trt)
        assert isinstance(result.srm_result, SRMResult)

    def test_pipeline_steps_populated(self, conversion_data):
        ctrl, trt = conversion_data
        result = auto(ctrl, trt)
        assert len(result.pipeline_steps) >= 4

    def test_detects_conversion_metric(self, conversion_data):
        ctrl, trt = conversion_data
        result = auto(ctrl, trt)
        assert result.primary_result.metric == "conversion"

    def test_detects_continuous_metric(self, continuous_data):
        ctrl, trt = continuous_data
        result = auto(ctrl, trt)
        assert result.primary_result.metric == "continuous"


# ─── Outlier Handling ────────────────────────────────────────────────


class TestAutoOutliers:
    def test_winsorizes_continuous_outliers(self, rng):
        ctrl = rng.normal(10, 2, 1000)
        trt = rng.normal(10, 2, 1000)
        ctrl[0] = 1000.0  # extreme outlier
        result = auto(ctrl, trt)
        # Should mention winsorization in pipeline
        winsorize_steps = [s for s in result.pipeline_steps if "insoriz" in s]
        assert len(winsorize_steps) >= 1

    def test_skips_outlier_handling_for_conversion(self, conversion_data):
        ctrl, trt = conversion_data
        result = auto(ctrl, trt)
        skip_steps = [s for s in result.pipeline_steps if "Skipped outlier" in s]
        assert len(skip_steps) == 1


# ─── CUPED Integration ──────────────────────────────────────────────


class TestAutoCUPED:
    def test_applies_cuped_with_pre_data(self, rng):
        pre = rng.normal(10, 2, 2000)
        ctrl = pre[:1000] + rng.normal(0, 1, 1000)
        trt = pre[1000:] + 0.5 + rng.normal(0, 1, 1000)
        result = auto(ctrl, trt, control_pre=pre[:1000], treatment_pre=pre[1000:])
        assert result.variance_reduction is not None
        assert result.variance_reduction > 0
        cuped_steps = [s for s in result.pipeline_steps if "CUPED" in s]
        assert len(cuped_steps) >= 1

    def test_no_cuped_without_pre_data(self, continuous_data):
        ctrl, trt = continuous_data
        result = auto(ctrl, trt)
        assert result.variance_reduction is None

    def test_cuped_recommendation_for_continuous(self, continuous_data):
        ctrl, trt = continuous_data
        result = auto(ctrl, trt)
        cuped_recs = [r for r in result.recommendations if "CUPED" in r]
        assert len(cuped_recs) >= 1


# ─── Multiple Metrics ───────────────────────────────────────────────


class TestAutoMultipleMetrics:
    def test_corrected_pvalues_with_metrics(self, rng):
        ctrl = rng.binomial(1, 0.10, 5000)
        trt = rng.binomial(1, 0.12, 5000)
        ctrl2 = rng.normal(10, 2, 5000)
        trt2 = rng.normal(10.1, 2, 5000)
        result = auto(ctrl, trt, metrics={"secondary": (ctrl2, trt2)})
        assert result.corrected_pvalues is not None
        assert len(result.corrected_pvalues) == 2

    def test_no_correction_single_metric(self, conversion_data):
        ctrl, trt = conversion_data
        result = auto(ctrl, trt)
        assert result.corrected_pvalues is None


# ─── SRM ─────────────────────────────────────────────────────────────


class TestAutoSRM:
    def test_srm_passes_balanced(self, conversion_data):
        ctrl, trt = conversion_data
        result = auto(ctrl, trt)
        assert result.srm_result.passed is True


# ─── Serialization ──────────────────────────────────────────────────


class TestAutoSerialization:
    def test_to_dict(self, conversion_data):
        ctrl, trt = conversion_data
        result = auto(ctrl, trt)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "primary_result" in d
        assert "pipeline_steps" in d

    def test_repr(self, conversion_data):
        ctrl, trt = conversion_data
        result = auto(ctrl, trt)
        text = repr(result)
        assert "AutoResult" in text
