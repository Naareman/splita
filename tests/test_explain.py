"""Tests for splita.explain — plain-English result interpretation."""

from __future__ import annotations

import json

import pytest

from splita._types import (
    AutoResult,
    BanditResult,
    BayesianResult,
    CheckResult,
    ClusterResult,
    ComparisonResult,
    CorrectionResult,
    DiagnosisResult,
    DiDResult,
    ExperimentResult,
    HTEResult,
    InteractionResult,
    MetaAnalysisResult,
    MonitorResult,
    mSPRTResult,
    mSPRTState,
    MultiObjectiveResult,
    NoveltyCurveResult,
    PowerSimulationResult,
    QuantileResult,
    RecommendationResult,
    SampleSizeResult,
    SimulationResult,
    SRMResult,
    StratifiedResult,
    SurvivalResult,
    SyntheticControlResult,
    TriggeredResult,
    WhatIfResult,
)
from splita.explain import explain


# ─── Fixtures ────────────────────────────────────────────────────────


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
def nonsignificant_result() -> ExperimentResult:
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
def underpowered_significant() -> ExperimentResult:
    return ExperimentResult(
        control_mean=0.10,
        treatment_mean=0.12,
        lift=0.02,
        relative_lift=0.20,
        pvalue=0.048,
        statistic=1.98,
        ci_lower=0.0001,
        ci_upper=0.04,
        significant=True,
        alpha=0.05,
        method="ztest",
        metric="conversion",
        control_n=500,
        treatment_n=500,
        power=0.55,
        effect_size=0.064,
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
        message="Sample ratio mismatch detected!",
    )


@pytest.fixture
def srm_passed() -> SRMResult:
    return SRMResult(
        observed=[4990, 5010],
        expected_counts=[5000.0, 5000.0],
        chi2_statistic=0.04,
        pvalue=0.84,
        passed=True,
        alpha=0.01,
        deviations_pct=[-0.2, 0.2],
        worst_variant=1,
        message="No sample ratio mismatch detected.",
    )


@pytest.fixture
def bayesian_strong() -> BayesianResult:
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
def bayesian_weak() -> BayesianResult:
    return BayesianResult(
        prob_b_beats_a=0.62,
        expected_loss_a=0.005,
        expected_loss_b=0.003,
        lift=0.005,
        relative_lift=0.05,
        ci_lower=-0.002,
        ci_upper=0.012,
        credible_level=0.95,
        control_mean=0.10,
        treatment_mean=0.105,
        prob_in_rope=0.35,
        rope=(-0.005, 0.005),
        metric="conversion",
        control_n=2000,
        treatment_n=2000,
    )


@pytest.fixture
def bayesian_control_wins() -> BayesianResult:
    return BayesianResult(
        prob_b_beats_a=0.03,
        expected_loss_a=0.0001,
        expected_loss_b=0.02,
        lift=-0.03,
        relative_lift=-0.25,
        ci_lower=-0.05,
        ci_upper=-0.01,
        credible_level=0.95,
        control_mean=0.12,
        treatment_mean=0.09,
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
        days_needed=None,
    )


@pytest.fixture
def sample_size_with_days() -> SampleSizeResult:
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


# ─── Test: ExperimentResult explain ─────────────────────────────────


class TestExplainExperiment:
    def test_significant_mentions_significant(
        self, significant_result: ExperimentResult
    ) -> None:
        text = explain(significant_result)
        assert "statistically significant" in text
        assert "p=0.00" in text

    def test_significant_mentions_effect_size(
        self, significant_result: ExperimentResult
    ) -> None:
        text = explain(significant_result)
        assert "Cohen's d" in text
        assert "negligible" in text  # 0.15 < 0.2 threshold

    def test_significant_mentions_power(
        self, significant_result: ExperimentResult
    ) -> None:
        text = explain(significant_result)
        assert "power" in text.lower()
        assert "adequate" in text

    def test_nonsignificant_mentions_no_difference(
        self, nonsignificant_result: ExperimentResult
    ) -> None:
        text = explain(nonsignificant_result)
        assert "No statistically significant" in text
        assert "p=0.42" in text

    def test_nonsignificant_suggests_cuped(
        self, nonsignificant_result: ExperimentResult
    ) -> None:
        text = explain(nonsignificant_result)
        assert "CUPED" in text

    def test_underpowered_warns(
        self, underpowered_significant: ExperimentResult
    ) -> None:
        text = explain(underpowered_significant)
        assert "underpowered" in text.lower()

    def test_decreased_effect(self) -> None:
        result = ExperimentResult(
            control_mean=0.12,
            treatment_mean=0.10,
            lift=-0.02,
            relative_lift=-0.167,
            pvalue=0.003,
            statistic=-2.97,
            ci_lower=-0.033,
            ci_upper=-0.007,
            significant=True,
            alpha=0.05,
            method="ztest",
            metric="conversion",
            control_n=5000,
            treatment_n=5000,
            power=0.82,
            effect_size=0.15,
        )
        text = explain(result)
        assert "decreased" in text


# ─── Test: SRMResult explain ────────────────────────────────────────


class TestExplainSRM:
    def test_srm_failed_warning(self, srm_failed: SRMResult) -> None:
        text = explain(srm_failed)
        assert "WARNING" in text
        assert "Sample Ratio Mismatch" in text
        assert "invalid" in text

    def test_srm_failed_suggestions(self, srm_failed: SRMResult) -> None:
        text = explain(srm_failed)
        assert "randomization" in text.lower()
        assert "bot" in text.lower()

    def test_srm_passed(self, srm_passed: SRMResult) -> None:
        text = explain(srm_passed)
        assert "No Sample Ratio Mismatch" in text
        assert "WARNING" not in text


# ─── Test: BayesianResult explain ───────────────────────────────────


class TestExplainBayesian:
    def test_strong_treatment(self, bayesian_strong: BayesianResult) -> None:
        text = explain(bayesian_strong)
        assert "98.30%" in text
        assert "ship" in text.lower()

    def test_weak_evidence(self, bayesian_weak: BayesianResult) -> None:
        text = explain(bayesian_weak)
        assert "62.00%" in text
        assert "not yet decisive" in text

    def test_rope_mentioned(self, bayesian_weak: BayesianResult) -> None:
        text = explain(bayesian_weak)
        assert "ROPE" in text
        assert "negligible" in text

    def test_control_wins(
        self, bayesian_control_wins: BayesianResult
    ) -> None:
        text = explain(bayesian_control_wins)
        assert "3.00%" in text
        assert "keep the control" in text.lower()


# ─── Test: SampleSizeResult explain ─────────────────────────────────


class TestExplainSampleSize:
    def test_basic_output(self, sample_size_result: SampleSizeResult) -> None:
        text = explain(sample_size_result)
        assert "3,843" in text
        assert "7,686" in text
        assert "80.00%" in text

    def test_with_days(
        self, sample_size_with_days: SampleSizeResult
    ) -> None:
        text = explain(sample_size_with_days)
        assert "8 days" in text


# ─── Test: BanditResult explain ─────────────────────────────────────


class TestExplainBandit:
    def test_bandit_output(self) -> None:
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
        text = explain(result)
        assert "1000 pulls" in text
        assert "arm 0" in text
        assert "95%" in text
        assert "0.0002" in text


# ─── Test: CorrectionResult explain ─────────────────────────────────


class TestExplainCorrection:
    def test_correction_with_labels(self) -> None:
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
        text = explain(result)
        assert "1 of 3" in text
        assert "Holm" in text
        assert "revenue" in text
        assert "clicks" in text

    def test_correction_without_labels(self) -> None:
        result = CorrectionResult(
            pvalues=[0.01, 0.04],
            adjusted_pvalues=[0.02, 0.08],
            rejected=[True, False],
            alpha=0.05,
            method="Bonferroni",
            n_rejected=1,
            n_tests=2,
            labels=None,
        )
        text = explain(result)
        assert "1 of 2" in text
        assert "test_0" in text


# ─── Test: mSPRT explain ────────────────────────────────────────────


class TestExplainMSPRT:
    def test_msprt_state_should_stop(self) -> None:
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
        text = explain(result)
        assert "1000 observations" in text
        assert "0.0400" in text
        assert "Should stop" in text
        assert "has detected" in text

    def test_msprt_state_should_not_stop(self) -> None:
        result = mSPRTState(
            n_control=100,
            n_treatment=100,
            mixture_lr=1.5,
            always_valid_pvalue=0.67,
            always_valid_ci_lower=-0.02,
            always_valid_ci_upper=0.04,
            should_stop=False,
            current_effect_estimate=0.01,
        )
        text = explain(result)
        assert "Should not stop" in text
        assert "has not detected" in text

    def test_msprt_result(self) -> None:
        result = mSPRTResult(
            n_control=2000,
            n_treatment=2000,
            mixture_lr=50.0,
            always_valid_pvalue=0.02,
            always_valid_ci_lower=0.005,
            always_valid_ci_upper=0.035,
            should_stop=True,
            current_effect_estimate=0.02,
            stopping_reason="boundary crossed",
            total_observations=4000,
            relative_speedup_vs_fixed_horizon=0.3,
        )
        text = explain(result)
        assert "4000 observations" in text
        assert "0.0200" in text
        assert "has detected" in text


# ─── Test: QuantileResult explain ────────────────────────────────────


class TestExplainQuantile:
    def test_quantile_with_significant(self) -> None:
        result = QuantileResult(
            quantiles=[0.25, 0.50, 0.75],
            control_quantiles=[5.0, 10.0, 15.0],
            treatment_quantiles=[5.5, 11.0, 16.0],
            differences=[0.5, 1.0, 1.0],
            ci_lower=[0.1, 0.3, 0.2],
            ci_upper=[0.9, 1.7, 1.8],
            pvalues=[0.02, 0.001, 0.01],
            significant=[True, True, True],
            alpha=0.05,
            control_n=500,
            treatment_n=500,
        )
        text = explain(result)
        assert "25%" in text
        assert "50%" in text
        assert "median difference" in text

    def test_quantile_no_significant(self) -> None:
        result = QuantileResult(
            quantiles=[0.50],
            control_quantiles=[10.0],
            treatment_quantiles=[10.1],
            differences=[0.1],
            ci_lower=[-0.5],
            ci_upper=[0.7],
            pvalues=[0.35],
            significant=[False],
            alpha=0.05,
            control_n=100,
            treatment_n=100,
        )
        text = explain(result)
        assert "No significant" in text


# ─── Test: ClusterResult explain ─────────────────────────────────────


class TestExplainCluster:
    def test_cluster_output(self) -> None:
        result = ClusterResult(
            lift=0.05,
            pvalue=0.02,
            ci_lower=0.01,
            ci_upper=0.09,
            significant=True,
            n_clusters_control=20,
            n_clusters_treatment=20,
            icc=0.05,
        )
        text = explain(result)
        assert "Cluster-robust" in text
        assert "lift" in text
        assert "design effect" in text


# ─── Test: StratifiedResult explain ──────────────────────────────────


class TestExplainStratified:
    def test_stratified_output(self) -> None:
        result = StratifiedResult(
            ate=0.03,
            se=0.01,
            pvalue=0.001,
            ci_lower=0.01,
            ci_upper=0.05,
            significant=True,
            n_strata=4,
            stratum_effects=[],
            alpha=0.05,
        )
        text = explain(result)
        assert "4 strata" in text
        assert "ATE" in text


# ─── Test: HTEResult explain ─────────────────────────────────────────


class TestExplainHTE:
    def test_hte_output(self) -> None:
        result = HTEResult(
            cate_estimates=[0.01, 0.02, 0.03],
            mean_cate=0.02,
            cate_std=0.008,
            top_features=None,
            method="t_learner",
        )
        text = explain(result)
        assert "CATE std" in text
        assert "0.008" in text
        assert "Mean CATE" in text


# ─── Test: TriggeredResult explain ───────────────────────────────────


class TestExplainTriggered:
    def test_triggered_output(self) -> None:
        itt = ExperimentResult(
            control_mean=0.10, treatment_mean=0.11,
            lift=0.01, relative_lift=0.10, pvalue=0.05,
            statistic=1.96, ci_lower=0.0, ci_upper=0.02,
            significant=True, alpha=0.05, method="ztest",
            metric="conversion", control_n=5000,
            treatment_n=5000, power=0.60, effect_size=0.05,
        )
        pp = ExperimentResult(
            control_mean=0.10, treatment_mean=0.13,
            lift=0.03, relative_lift=0.30, pvalue=0.001,
            statistic=3.2, ci_lower=0.01, ci_upper=0.05,
            significant=True, alpha=0.05, method="ztest",
            metric="conversion", control_n=2000,
            treatment_n=2000, power=0.85, effect_size=0.10,
        )
        result = TriggeredResult(
            itt_result=itt,
            per_protocol_result=pp,
            trigger_rate_control=0.40,
            trigger_rate_treatment=0.42,
        )
        text = explain(result)
        assert "ITT effect" in text
        assert "Per-protocol effect" in text
        assert "Trigger rate" in text


# ─── Test: InteractionResult explain ─────────────────────────────────


class TestExplainInteraction:
    def test_interaction_significant(self) -> None:
        result = InteractionResult(
            segment_results=[
                {"segment": "mobile", "lift": 0.05, "pvalue": 0.01},
                {"segment": "desktop", "lift": 0.01, "pvalue": 0.30},
            ],
            has_interaction=True,
            interaction_pvalue=0.02,
            strongest_segment="mobile",
        )
        text = explain(result)
        assert "does differ" in text
        assert "0.02" in text

    def test_interaction_not_significant(self) -> None:
        result = InteractionResult(
            segment_results=[
                {"segment": "mobile", "lift": 0.03, "pvalue": 0.05},
                {"segment": "desktop", "lift": 0.02, "pvalue": 0.08},
            ],
            has_interaction=False,
            interaction_pvalue=0.45,
            strongest_segment="mobile",
        )
        text = explain(result)
        assert "does not differ" in text


# ─── Test: MultiObjectiveResult explain ──────────────────────────────


class TestExplainMultiObjective:
    def test_multi_objective_output(self) -> None:
        m1 = ExperimentResult(
            control_mean=0.10, treatment_mean=0.12,
            lift=0.02, relative_lift=0.20, pvalue=0.003,
            statistic=2.97, ci_lower=0.007, ci_upper=0.033,
            significant=True, alpha=0.05, method="ztest",
            metric="revenue", control_n=5000,
            treatment_n=5000, power=0.82, effect_size=0.15,
        )
        result = MultiObjectiveResult(
            metric_results=[m1],
            pareto_dominant=True,
            tradeoffs=[],
            corrected_pvalues=[0.003],
            recommendation="adopt",
        )
        text = explain(result)
        assert "adopt" in text
        assert "1 of 1" in text


# ─── Test: DiDResult explain ─────────────────────────────────────────


class TestExplainDiD:
    def test_did_parallel_pass(self) -> None:
        result = DiDResult(
            att=0.05,
            se=0.02,
            pvalue=0.01,
            ci_lower=0.01,
            ci_upper=0.09,
            significant=True,
            pre_trend_diff=0.001,
            parallel_trends_pvalue=0.80,
        )
        text = explain(result)
        assert "ATT" in text
        assert "pass" in text

    def test_did_parallel_fail(self) -> None:
        result = DiDResult(
            att=0.05,
            se=0.02,
            pvalue=0.01,
            ci_lower=0.01,
            ci_upper=0.09,
            significant=True,
            pre_trend_diff=0.05,
            parallel_trends_pvalue=0.01,
        )
        text = explain(result)
        assert "fail" in text


# ─── Test: SyntheticControlResult explain ────────────────────────────


class TestExplainSyntheticControl:
    def test_synthetic_control_output(self) -> None:
        result = SyntheticControlResult(
            effect=2.5,
            weights=(0.4, 0.3, 0.3),
            pre_treatment_rmse=0.15,
            donor_contributions={0: 0.4, 1: 0.3, 2: 0.3},
            effect_series=[2.0, 2.5, 3.0],
        )
        text = explain(result)
        assert "2.50" in text
        assert "RMSE" in text


# ─── Test: MetaAnalysisResult explain ────────────────────────────────


class TestExplainMetaAnalysis:
    def test_meta_analysis_output(self) -> None:
        result = MetaAnalysisResult(
            combined_effect=0.05,
            combined_se=0.01,
            pvalue=0.001,
            ci_lower=0.03,
            ci_upper=0.07,
            heterogeneity_pvalue=0.30,
            i_squared=0.25,
            method="random",
            study_weights=[0.5, 0.3, 0.2],
            labels=["study_a", "study_b", "study_c"],
        )
        text = explain(result)
        assert "random" in text
        assert "0.05" in text
        assert "25%" in text


# ─── Test: AutoResult explain ────────────────────────────────────────


class TestExplainAuto:
    def test_auto_uses_reasoning(self, significant_result, srm_passed) -> None:
        result = AutoResult(
            primary_result=significant_result,
            srm_result=srm_passed,
            variance_reduction=None,
            corrected_pvalues=None,
            pipeline_steps=["step1"],
            recommendations=["ship it"],
            reasoning=["SRM passed", "Effect is significant", "Ship treatment"],
        )
        text = explain(result)
        assert "SRM passed" in text
        assert "Ship treatment" in text


# ─── Test: CheckResult explain ───────────────────────────────────────


class TestExplainCheck:
    def test_check_all_passed(self) -> None:
        result = CheckResult(
            srm_passed=True,
            flicker_rate=0.01,
            has_outliers=False,
            is_powered=True,
            all_passed=True,
            checks=[
                {"name": "srm", "passed": True},
                {"name": "power", "passed": True},
            ],
            recommendations=[],
        )
        text = explain(result)
        assert "passed" in text
        assert "2/2" in text

    def test_check_with_failures(self) -> None:
        result = CheckResult(
            srm_passed=False,
            flicker_rate=0.05,
            has_outliers=True,
            is_powered=False,
            all_passed=False,
            checks=[
                {"name": "srm", "passed": False},
                {"name": "power", "passed": False},
                {"name": "outliers", "passed": True},
            ],
            recommendations=["Investigate SRM", "Increase sample size"],
        )
        text = explain(result)
        assert "failed" in text
        assert "1/3" in text
        assert "Investigate SRM" in text


# ─── Test: MonitorResult explain ─────────────────────────────────────


class TestExplainMonitor:
    def test_monitor_output(self) -> None:
        result = MonitorResult(
            current_lift=0.015,
            current_pvalue=0.08,
            current_n=8000,
            srm_passed=True,
            guardrail_status=[],
            days_remaining=5,
            predicted_significant=True,
            recommendation="continue",
        )
        text = explain(result)
        assert "continue" in text
        assert "5 days remaining" in text

    def test_monitor_no_days(self) -> None:
        result = MonitorResult(
            current_lift=0.015,
            current_pvalue=0.08,
            current_n=8000,
            srm_passed=True,
            guardrail_status=[],
            days_remaining=None,
            predicted_significant=True,
            recommendation="continue",
        )
        text = explain(result)
        assert "days remaining unknown" in text


# ─── Test: WhatIfResult explain ──────────────────────────────────────


class TestExplainWhatIf:
    def test_whatif_output(self) -> None:
        result = WhatIfResult(
            original_n=5000,
            projected_n=10000,
            original_pvalue=0.08,
            projected_pvalue=0.02,
            original_significant=False,
            projected_significant=True,
            projected_power=0.92,
            message="Doubling sample size would make the result significant.",
        )
        text = explain(result)
        assert "Doubling sample size" in text


# ─── Test: PowerSimulationResult explain ─────────────────────────────


class TestExplainPowerSimulation:
    def test_power_sim_output(self) -> None:
        result = PowerSimulationResult(
            power=0.85,
            rejection_rate=0.85,
            n_simulations=1000,
            n_per_variant=500,
            alpha=0.05,
            mean_effect=0.02,
            mean_pvalue=0.03,
            ci_power_lower=0.82,
            ci_power_upper=0.88,
        )
        text = explain(result)
        assert "85%" in text
        assert "n=500" in text


# ─── Test: SimulationResult explain ──────────────────────────────────


class TestExplainSimulation:
    def test_simulation_output(self) -> None:
        result = SimulationResult(
            estimated_power=0.85,
            median_pvalue=0.02,
            median_lift=0.02,
            ci_width_median=0.03,
            significant_rate=0.85,
            false_negative_rate=0.15,
            recommendation="Sample size is adequate for the expected effect.",
        )
        text = explain(result)
        assert "Sample size is adequate" in text


# ─── Test: ComparisonResult explain ──────────────────────────────────


class TestExplainComparison:
    def test_comparison_significant(self) -> None:
        result = ComparisonResult(
            effect_a=0.02,
            effect_b=0.05,
            difference=0.03,
            se=0.01,
            pvalue=0.003,
            significant=True,
            ci_lower=0.01,
            ci_upper=0.05,
            direction="b_larger",
        )
        text = explain(result)
        assert "significantly different" in text
        assert "b_larger" in text

    def test_comparison_not_significant(self) -> None:
        result = ComparisonResult(
            effect_a=0.02,
            effect_b=0.025,
            difference=0.005,
            se=0.01,
            pvalue=0.62,
            significant=False,
            ci_lower=-0.015,
            ci_upper=0.025,
            direction="equivalent",
        )
        text = explain(result)
        assert "not significantly different" in text


# ─── Test: SurvivalResult explain ────────────────────────────────────


class TestExplainSurvival:
    def test_survival_output(self) -> None:
        result = SurvivalResult(
            hazard_ratio=0.75,
            logrank_pvalue=0.01,
            significant=True,
            median_survival_ctrl=30.0,
            median_survival_trt=40.0,
            ci_lower=0.60,
            ci_upper=0.95,
            alpha=0.05,
            n_ctrl=200,
            n_trt=200,
            n_events_ctrl=50,
            n_events_trt=35,
        )
        text = explain(result)
        assert "Hazard ratio" in text
        assert "0.75" in text
        assert "Log-rank" in text
        assert "ctrl=30.00" in text
        assert "trt=40.00" in text

    def test_survival_median_not_reached(self) -> None:
        result = SurvivalResult(
            hazard_ratio=0.90,
            logrank_pvalue=0.30,
            significant=False,
            median_survival_ctrl=None,
            median_survival_trt=None,
            ci_lower=0.70,
            ci_upper=1.15,
            alpha=0.05,
            n_ctrl=100,
            n_trt=100,
            n_events_ctrl=20,
            n_events_trt=18,
        )
        text = explain(result)
        assert "not reached" in text


# ─── Test: generic fallback ──────────────────────────────────────────


class TestExplainGenericFallback:
    def test_novelty_curve_uses_generic(self) -> None:
        """NoveltyCurveResult has no dedicated handler, should use generic."""
        result = NoveltyCurveResult(
            windows=[{"window_start": 0, "lift": 0.05, "pvalue": 0.01,
                       "ci_lower": 0.01, "ci_upper": 0.09}],
            has_novelty_effect=False,
            trend_direction="stable",
        )
        text = explain(result)
        assert "NoveltyCurveResult" in text
        assert "has_novelty_effect=False" in text
        assert "trend_direction=stable" in text

    def test_generic_does_not_raise(self) -> None:
        """Any dataclass should get a generic explanation, not a TypeError."""
        result = NoveltyCurveResult(
            windows=[],
            has_novelty_effect=False,
            trend_direction="stable",
        )
        # Should not raise
        text = explain(result)
        assert isinstance(text, str)
        assert len(text) > 0


# ─── Test: unsupported non-dataclass still raises ────────────────────


class TestExplainNonDataclass:
    def test_string_gets_generic(self) -> None:
        """Non-dataclass objects should still get a generic explanation."""
        text = explain("not a result")
        assert "str:" in text

    def test_int_gets_generic(self) -> None:
        text = explain(42)
        assert "int:" in text


# ─── Test: serialization roundtrip ──────────────────────────────────


class TestSerialization:
    def test_experiment_to_json(
        self, significant_result: ExperimentResult
    ) -> None:
        json_str = significant_result.to_json()
        parsed = json.loads(json_str)
        assert parsed["control_mean"] == 0.10
        assert parsed["significant"] is True

    def test_experiment_roundtrip(
        self, significant_result: ExperimentResult
    ) -> None:
        d = significant_result.to_dict()
        restored = ExperimentResult.from_dict(d)
        assert restored.control_mean == significant_result.control_mean
        assert restored.pvalue == significant_result.pvalue
        assert restored.significant == significant_result.significant
        assert restored.method == significant_result.method

    def test_srm_roundtrip(self, srm_failed: SRMResult) -> None:
        d = srm_failed.to_dict()
        restored = SRMResult.from_dict(d)
        assert restored.passed == srm_failed.passed
        assert restored.pvalue == srm_failed.pvalue
        assert restored.observed == srm_failed.observed

    def test_bayesian_roundtrip(
        self, bayesian_strong: BayesianResult
    ) -> None:
        json_str = bayesian_strong.to_json()
        parsed = json.loads(json_str)
        restored = BayesianResult.from_dict(parsed)
        assert restored.prob_b_beats_a == bayesian_strong.prob_b_beats_a
        assert restored.metric == bayesian_strong.metric

    def test_sample_size_roundtrip(
        self, sample_size_result: SampleSizeResult
    ) -> None:
        d = sample_size_result.to_dict()
        restored = SampleSizeResult.from_dict(d)
        assert restored.n_per_variant == sample_size_result.n_per_variant
        assert restored.baseline == sample_size_result.baseline

    def test_from_dict_ignores_extra_keys(self) -> None:
        d = dict(
            control_mean=0.10,
            treatment_mean=0.12,
            lift=0.02,
            relative_lift=0.2,
            pvalue=0.03,
            statistic=2.1,
            ci_lower=0.002,
            ci_upper=0.038,
            significant=True,
            alpha=0.05,
            method="ztest",
            metric="conversion",
            control_n=1000,
            treatment_n=1000,
            power=0.65,
            effect_size=0.06,
            extra_field="should be ignored",
        )
        result = ExperimentResult.from_dict(d)
        assert result.control_mean == 0.10
        assert not hasattr(result, "extra_field")

    def test_to_json_indent(
        self, significant_result: ExperimentResult
    ) -> None:
        json_str = significant_result.to_json(indent=4)
        # 4-space indentation should produce lines with 4 leading spaces
        assert "    " in json_str

    def test_to_json_parses_as_valid_json(
        self, significant_result: ExperimentResult
    ) -> None:
        json_str = significant_result.to_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert len(parsed) == 16  # 16 fields in ExperimentResult

    def test_bayesian_roundtrip_with_rope(
        self, bayesian_weak: BayesianResult
    ) -> None:
        json_str = bayesian_weak.to_json()
        parsed = json.loads(json_str)
        # rope is a tuple, serialized as a list in JSON
        restored = BayesianResult.from_dict(
            {**parsed, "rope": tuple(parsed["rope"]) if parsed["rope"] else None}
        )
        assert restored.prob_in_rope == bayesian_weak.prob_in_rope
        assert restored.rope == bayesian_weak.rope
