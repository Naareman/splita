"""Tests for splita.recommend — experiment design guidance."""

from __future__ import annotations

import numpy as np
import pytest

from splita.recommend import recommend
from splita._types import RecommendationResult


# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def binary_data(rng):
    return rng.binomial(1, 0.10, 5000)


@pytest.fixture
def continuous_normal(rng):
    return rng.normal(10, 2, 5000)


@pytest.fixture
def continuous_skewed(rng):
    return rng.exponential(1, 5000)


@pytest.fixture
def small_sample(rng):
    return rng.normal(10, 2, 30)


# ─── Return Type ─────────────────────────────────────────────────────


class TestRecommendBasic:
    def test_returns_recommendation_result(self, binary_data):
        result = recommend(binary_data)
        assert isinstance(result, RecommendationResult)

    def test_reasoning_is_nonempty(self, binary_data):
        result = recommend(binary_data)
        assert len(result.reasoning) >= 1

    def test_code_example_is_nonempty(self, binary_data):
        result = recommend(binary_data)
        assert len(result.code_example) > 0

    def test_repr_works(self, binary_data):
        result = recommend(binary_data)
        text = repr(result)
        assert "RecommendationResult" in text

    def test_to_dict(self, binary_data):
        result = recommend(binary_data)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "recommended_test" in d
        assert "reasoning" in d


# ─── Test Selection ──────────────────────────────────────────────────


class TestTestSelection:
    def test_binary_recommends_ztest(self, binary_data):
        result = recommend(binary_data)
        assert "Z-test" in result.recommended_test

    def test_continuous_normal_recommends_ttest(self, continuous_normal):
        result = recommend(continuous_normal)
        assert "t-test" in result.recommended_test

    def test_skewed_recommends_mannwhitney(self, continuous_skewed):
        result = recommend(continuous_skewed)
        assert "Mann-Whitney" in result.recommended_test or "bootstrap" in result.recommended_test

    def test_small_sample_recommends_permutation(self, small_sample):
        result = recommend(small_sample)
        assert "Permutation" in result.recommended_test or "bootstrap" in result.recommended_test

    def test_ratio_metric_recommends_delta(self, continuous_normal):
        result = recommend(continuous_normal, metric="ratio")
        assert "Delta" in result.recommended_test or "delta" in result.recommended_test

    def test_clusters_recommends_cluster_experiment(self, continuous_normal):
        result = recommend(continuous_normal, has_clusters=True)
        assert "Cluster" in result.recommended_test

    def test_binary_with_treatment(self, rng):
        ctrl = rng.binomial(1, 0.10, 5000)
        trt = rng.binomial(1, 0.12, 5000)
        result = recommend(ctrl, trt)
        assert "Z-test" in result.recommended_test


# ─── Variance Reduction ─────────────────────────────────────────────


class TestVarianceReduction:
    def test_pre_data_recommends_cuped(self, continuous_normal):
        result = recommend(continuous_normal, has_pre_data=True)
        assert result.recommended_variance is not None
        assert "CUPED" in result.recommended_variance

    def test_continuous_recommends_outlier_handler(self, continuous_normal):
        result = recommend(continuous_normal)
        assert result.recommended_variance is not None
        assert "OutlierHandler" in result.recommended_variance

    def test_binary_no_outlier_handler(self, binary_data):
        result = recommend(binary_data)
        if result.recommended_variance is not None:
            assert "OutlierHandler" not in result.recommended_variance

    def test_pre_data_and_continuous_recommends_both(self, continuous_normal):
        result = recommend(continuous_normal, has_pre_data=True)
        assert result.recommended_variance is not None
        assert "OutlierHandler" in result.recommended_variance
        assert "CUPED" in result.recommended_variance


# ─── Multiple Testing ───────────────────────────────────────────────


class TestMultipleTesting:
    def test_multiple_metrics_recommends_bh(self, continuous_normal):
        result = recommend(continuous_normal, n_metrics=3)
        assert result.recommended_correction is not None
        assert "BH" in result.recommended_correction or "Benjamini" in result.recommended_correction

    def test_single_metric_no_correction(self, continuous_normal):
        result = recommend(continuous_normal, n_metrics=1)
        assert result.recommended_correction is None

    def test_multiple_metrics_warns(self, continuous_normal):
        result = recommend(continuous_normal, n_metrics=5)
        assert len(result.warnings) >= 1
        warning_text = " ".join(result.warnings)
        assert "false positive" in warning_text.lower() or "correction" in warning_text.lower()


# ─── Sequential Testing ─────────────────────────────────────────────


class TestSequentialTesting:
    def test_sequential_recommends_msprt(self, continuous_normal):
        result = recommend(continuous_normal, is_sequential=True)
        assert result.recommended_sequential is not None
        assert "mSPRT" in result.recommended_sequential

    def test_not_sequential_no_recommendation(self, continuous_normal):
        result = recommend(continuous_normal, is_sequential=False)
        assert result.recommended_sequential is None

    def test_sequential_warns_about_peeking(self, continuous_normal):
        result = recommend(continuous_normal, is_sequential=True)
        warning_text = " ".join(result.warnings)
        assert "sequential" in warning_text.lower() or "peek" in warning_text.lower()


# ─── Code Example ────────────────────────────────────────────────────


class TestCodeExample:
    def test_code_example_is_valid_python(self, continuous_normal):
        result = recommend(continuous_normal)
        # Should parse as valid Python (compile check)
        compile(result.code_example, "<string>", "exec")

    def test_code_example_includes_imports(self, continuous_normal):
        result = recommend(continuous_normal)
        assert "from splita" in result.code_example

    def test_code_example_with_cuped(self, continuous_normal):
        result = recommend(continuous_normal, has_pre_data=True)
        assert "CUPED" in result.code_example

    def test_code_example_with_clusters(self, continuous_normal):
        result = recommend(continuous_normal, has_clusters=True)
        assert "Cluster" in result.code_example

    def test_code_example_with_sequential(self, continuous_normal):
        result = recommend(continuous_normal, is_sequential=True)
        assert "mSPRT" in result.code_example


# ─── Warnings ────────────────────────────────────────────────────────


class TestWarnings:
    def test_small_sample_warns(self, small_sample):
        result = recommend(small_sample)
        assert len(result.warnings) >= 1
        warning_text = " ".join(result.warnings)
        assert "small" in warning_text.lower() or "sample" in warning_text.lower()

    def test_normal_data_no_sample_warning(self, continuous_normal):
        result = recommend(continuous_normal)
        warning_text = " ".join(result.warnings)
        assert "small" not in warning_text.lower()
