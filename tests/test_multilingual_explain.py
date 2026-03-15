"""Tests for multilingual explain() — Arabic, Spanish, Chinese support."""

from __future__ import annotations

import pytest

from splita._types import (
    BayesianResult,
    ExperimentResult,
    SampleSizeResult,
    SRMResult,
)
from splita.explain import explain


@pytest.fixture
def significant_result() -> ExperimentResult:
    return ExperimentResult(
        control_mean=0.10,
        treatment_mean=0.12,
        lift=0.02,
        relative_lift=0.20,
        pvalue=0.003,
        statistic=2.97,
        ci_lower=0.007,
        ci_upper=0.033,
        significant=True,
        alpha=0.05,
        method="ztest",
        metric="conversion",
        control_n=5000,
        treatment_n=5000,
        power=0.82,
        effect_size=0.15,
    )


@pytest.fixture
def nonsig_result() -> ExperimentResult:
    return ExperimentResult(
        control_mean=0.10,
        treatment_mean=0.101,
        lift=0.001,
        relative_lift=0.01,
        pvalue=0.42,
        statistic=0.81,
        ci_lower=-0.004,
        ci_upper=0.006,
        significant=False,
        alpha=0.05,
        method="ztest",
        metric="conversion",
        control_n=200,
        treatment_n=200,
        power=0.12,
        effect_size=0.01,
    )


@pytest.fixture
def srm_failed() -> SRMResult:
    return SRMResult(
        observed=[4500, 5500],
        expected_counts=[5000.0, 5000.0],
        chi2_statistic=100.0,
        pvalue=0.001,
        passed=False,
        alpha=0.01,
        deviations_pct=[-10.0, 10.0],
        worst_variant=1,
        message="SRM detected.",
    )


@pytest.fixture
def bayesian_result() -> BayesianResult:
    return BayesianResult(
        prob_b_beats_a=0.983,
        expected_loss_a=0.015,
        expected_loss_b=0.0003,
        lift=0.02,
        relative_lift=0.20,
        ci_lower=0.005,
        ci_upper=0.035,
        credible_level=0.95,
        control_mean=0.10,
        treatment_mean=0.12,
        prob_in_rope=None,
        rope=None,
        metric="conversion",
        control_n=5000,
        treatment_n=5000,
    )


@pytest.fixture
def sample_size_result() -> SampleSizeResult:
    return SampleSizeResult(
        n_per_variant=3843,
        n_total=7686,
        alpha=0.05,
        power=0.80,
        mde=0.02,
        relative_mde=0.20,
        baseline=0.10,
        metric="conversion",
        effect_size=0.064,
        days_needed=8,
    )


# ─── English (default) still works ──────────────────────────────────


class TestEnglishDefault:
    def test_default_is_english(self, significant_result):
        text = explain(significant_result)
        assert "significant" in text

    def test_explicit_en(self, significant_result):
        text = explain(significant_result, lang="en")
        assert "significant" in text


# ─── Arabic ──────────────────────────────────────────────────────────


class TestArabic:
    def test_experiment_arabic(self, significant_result):
        text = explain(significant_result, lang="ar")
        assert "دلالة إحصائية" in text

    def test_nonsig_arabic(self, nonsig_result):
        text = explain(nonsig_result, lang="ar")
        assert "لم يتم اكتشاف" in text

    def test_srm_arabic(self, srm_failed):
        text = explain(srm_failed, lang="ar")
        assert "تحذير" in text

    def test_bayesian_arabic(self, bayesian_result):
        text = explain(bayesian_result, lang="ar")
        assert "احتمال" in text

    def test_sample_size_arabic(self, sample_size_result):
        text = explain(sample_size_result, lang="ar")
        assert "مستخدم" in text


# ─── Spanish ─────────────────────────────────────────────────────────


class TestSpanish:
    def test_experiment_spanish(self, significant_result):
        text = explain(significant_result, lang="es")
        assert "estadísticamente significativo" in text

    def test_nonsig_spanish(self, nonsig_result):
        text = explain(nonsig_result, lang="es")
        assert "No se detectó" in text

    def test_srm_spanish(self, srm_failed):
        text = explain(srm_failed, lang="es")
        assert "ADVERTENCIA" in text

    def test_bayesian_spanish(self, bayesian_result):
        text = explain(bayesian_result, lang="es")
        assert "probabilidad" in text

    def test_sample_size_spanish(self, sample_size_result):
        text = explain(sample_size_result, lang="es")
        assert "usuarios" in text


# ─── Chinese ─────────────────────────────────────────────────────────


class TestChinese:
    def test_experiment_chinese(self, significant_result):
        text = explain(significant_result, lang="zh")
        assert "统计显著性" in text

    def test_nonsig_chinese(self, nonsig_result):
        text = explain(nonsig_result, lang="zh")
        assert "未检测到" in text

    def test_srm_chinese(self, srm_failed):
        text = explain(srm_failed, lang="zh")
        assert "警告" in text

    def test_bayesian_chinese(self, bayesian_result):
        text = explain(bayesian_result, lang="zh")
        assert "概率" in text

    def test_sample_size_chinese(self, sample_size_result):
        text = explain(sample_size_result, lang="zh")
        assert "用户" in text


# ─── Unsupported language ────────────────────────────────────────────


class TestUnsupportedLang:
    def test_invalid_lang_raises(self, significant_result):
        with pytest.raises(ValueError, match="must be one of"):
            explain(significant_result, lang="fr")

    def test_error_message_shows_supported(self, significant_result):
        with pytest.raises(ValueError, match="'en'"):
            explain(significant_result, lang="xx")
