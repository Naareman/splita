"""Tests for multilingual explain() — Arabic, Spanish, Chinese support."""

from __future__ import annotations

import pytest

from splita._types import (
    BanditResult,
    BayesianResult,
    ClusterResult,
    CorrectionResult,
    DiDResult,
    ExperimentResult,
    mSPRTState,
    SampleSizeResult,
    SRMResult,
    SurvivalResult,
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


# ─── Advanced types use English fallback for non-en languages ────────


class TestAdvancedTypesLangFallback:
    """Advanced result types should work with any lang parameter.

    They produce English output regardless of lang, since we haven't
    translated their templates yet. The key requirement is that they
    don't crash.
    """

    def test_bandit_all_langs(self):
        result = BanditResult(
            n_pulls_per_arm=[800, 200],
            prob_best=[0.95, 0.05],
            expected_loss=[0.0002, 0.015],
            current_best_arm=0,
            should_stop=True,
            total_reward=85.0,
            cumulative_regret=2.5,
            arm_means=[0.12, 0.08],
            arm_credible_intervals=[(0.10, 0.14), (0.06, 0.10)],
        )
        for lang in ("en", "ar", "es", "zh"):
            text = explain(result, lang=lang)
            assert "arm 0" in text
            assert "1000 total pulls" in text

    def test_correction_all_langs(self):
        result = CorrectionResult(
            pvalues=[0.01, 0.04, 0.20],
            adjusted_pvalues=[0.03, 0.06, 0.20],
            rejected=[True, False, False],
            alpha=0.05,
            method="Holm",
            n_rejected=1,
            n_tests=3,
            labels=["revenue", "clicks", "bounce"],
        )
        for lang in ("en", "ar", "es", "zh"):
            text = explain(result, lang=lang)
            assert "1 of 3" in text

    def test_msprt_all_langs(self):
        result = mSPRTState(
            n_control=500,
            n_treatment=500,
            mixture_lr=25.0,
            always_valid_pvalue=0.04,
            always_valid_ci_lower=0.005,
            always_valid_ci_upper=0.035,
            should_stop=True,
            current_effect_estimate=0.02,
        )
        for lang in ("en", "ar", "es", "zh"):
            text = explain(result, lang=lang)
            assert "1000 observations" in text

    def test_cluster_all_langs(self):
        result = ClusterResult(
            lift=0.05, pvalue=0.02,
            ci_lower=0.01, ci_upper=0.09,
            significant=True,
            n_clusters_control=20, n_clusters_treatment=20,
            icc=0.05,
        )
        for lang in ("en", "ar", "es", "zh"):
            text = explain(result, lang=lang)
            assert "Cluster-robust" in text

    def test_did_all_langs(self):
        result = DiDResult(
            att=0.05, se=0.02, pvalue=0.01,
            ci_lower=0.01, ci_upper=0.09,
            significant=True,
            pre_trend_diff=0.001,
            parallel_trends_pvalue=0.80,
        )
        for lang in ("en", "ar", "es", "zh"):
            text = explain(result, lang=lang)
            assert "ATT" in text

    def test_survival_all_langs(self):
        result = SurvivalResult(
            hazard_ratio=0.75, logrank_pvalue=0.01,
            significant=True,
            median_survival_ctrl=30.0, median_survival_trt=40.0,
            ci_lower=0.60, ci_upper=0.95,
            alpha=0.05, n_ctrl=200, n_trt=200,
            n_events_ctrl=50, n_events_trt=35,
        )
        for lang in ("en", "ar", "es", "zh"):
            text = explain(result, lang=lang)
            assert "hazard ratio" in text.lower()


# ─── Unsupported language ────────────────────────────────────────────


class TestUnsupportedLang:
    def test_invalid_lang_raises(self, significant_result):
        with pytest.raises(ValueError, match="must be one of"):
            explain(significant_result, lang="fr")

    def test_error_message_shows_supported(self, significant_result):
        with pytest.raises(ValueError, match="'en'"):
            explain(significant_result, lang="xx")
