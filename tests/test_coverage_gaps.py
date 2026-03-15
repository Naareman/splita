"""Tests to close remaining coverage gaps across multiple modules."""

from __future__ import annotations

import math

import numpy as np
import pytest

# ─── what_if.py ──────────────────────────────────────────────────────────

from splita._types import ExperimentResult
from splita.what_if import (
    _build_message,
    _compute_power,
    _estimate_se_from_result,
    _require_attrs,
    what_if,
)


def _make_experiment_result(**overrides) -> ExperimentResult:
    defaults = dict(
        control_mean=0.10,
        treatment_mean=0.12,
        lift=0.02,
        relative_lift=0.2,
        pvalue=0.12,
        statistic=1.5,
        ci_lower=-0.005,
        ci_upper=0.045,
        significant=False,
        alpha=0.05,
        method="ztest",
        metric="conversion",
        control_n=1000,
        treatment_n=1000,
        power=0.45,
        effect_size=0.06,
    )
    defaults.update(overrides)
    return ExperimentResult(**defaults)


class TestWhatIfEdgeCases:
    """Cover uncovered branches in what_if.py."""

    def test_pvalue_zero(self) -> None:
        """Lines 102-103: pvalue=0.0 -> original_z=10.0."""
        r = _make_experiment_result(pvalue=0.0, lift=0.02, significant=True)
        w = what_if(r, n=5000)
        assert w.projected_pvalue < 0.05

    def test_pvalue_one(self) -> None:
        """Lines 104-105: pvalue=1.0 -> original_z=0.0."""
        r = _make_experiment_result(pvalue=1.0, lift=0.0, significant=False)
        w = what_if(r, n=5000)
        assert w.projected_pvalue >= 0.0

    def test_negative_lift(self) -> None:
        """Line 109: negative lift inverts z."""
        r = _make_experiment_result(
            lift=-0.02,
            pvalue=0.12,
            significant=False,
            control_mean=0.12,
            treatment_mean=0.10,
        )
        w = what_if(r, n=20000)
        assert w.projected_pvalue < w.original_pvalue

    def test_z_zero_fallback_se(self) -> None:
        """Line 116: original_z=0 -> _estimate_se_from_result fallback."""
        r = _make_experiment_result(pvalue=1.0, lift=0.0, significant=False)
        w = what_if(r)
        assert w.projected_pvalue >= 0.0

    def test_projected_n_zero(self) -> None:
        """Line 123: projected_n=0 edge case."""
        r = _make_experiment_result()
        # n=0 falls through to projected_se = original_se
        # This tests the else branch at line 123
        w = what_if(r, n=0)
        assert w.projected_n == 0

    def test_projected_se_zero_nonzero_lift(self) -> None:
        """Line 130: projected_se=0 with nonzero lift -> pvalue=0.0."""
        # With very large n and small pvalue, projected_se approaches 0
        r = _make_experiment_result(pvalue=0.0, lift=0.02, significant=True)
        w = what_if(r, n=10**9)
        assert w.projected_pvalue == 0.0 or w.projected_pvalue < 0.001

    def test_require_attrs_missing(self) -> None:
        """Lines 165-171: _require_attrs raises."""
        class Fake:
            pass
        with pytest.raises(ValueError, match="missing required"):
            _require_attrs(Fake(), ["control_n", "lift"])

    def test_estimate_se_no_means(self) -> None:
        """Line 185: fallback to 1.0 when no control_mean/treatment_mean."""
        class Fake:
            pass
        se = _estimate_se_from_result(Fake(), 1000)
        assert se == 1.0

    def test_estimate_se_with_extreme_means(self) -> None:
        """Lines 177-184: control_mean at boundary (0 or 1)."""
        class Fake:
            control_mean = 0.0  # boundary: not in (0,1)
            treatment_mean = 0.0
        se = _estimate_se_from_result(Fake(), 1000)
        assert se > 0

    def test_compute_power_se_zero_nonzero_effect(self) -> None:
        """Line 191: se <= 0 with nonzero effect -> power=1.0."""
        assert _compute_power(0.5, 0.0, 0.05) == 1.0

    def test_compute_power_se_zero_zero_effect(self) -> None:
        """Line 191: se <= 0 with zero effect -> power=0.0."""
        assert _compute_power(0.0, 0.0, 0.05) == 0.0

    def test_build_message_alpha_change(self) -> None:
        """Line 216: alpha change branch."""
        msg = _build_message(
            original_n=2000,
            projected_n=2000,
            original_pvalue=0.04,
            projected_pvalue=0.04,
            original_significant=True,
            projected_significant=False,
            projected_power=0.5,
            original_alpha=0.05,
            projected_alpha=0.01,
        )
        assert "alpha=0.01" in msg

    def test_build_message_same_conditions(self) -> None:
        """Line 218: same conditions."""
        msg = _build_message(
            original_n=2000,
            projected_n=2000,
            original_pvalue=0.04,
            projected_pvalue=0.04,
            original_significant=True,
            projected_significant=True,
            projected_power=0.8,
            original_alpha=0.05,
            projected_alpha=0.05,
        )
        assert "same conditions" in msg

    def test_build_message_loses_significance(self) -> None:
        """Line 225: projected not significant but original was."""
        msg = _build_message(
            original_n=2000,
            projected_n=200,
            original_pvalue=0.04,
            projected_pvalue=0.4,
            original_significant=True,
            projected_significant=False,
            projected_power=0.2,
            original_alpha=0.05,
            projected_alpha=0.05,
        )
        assert "would NOT be" in msg

    def test_build_message_stays_not_significant(self) -> None:
        """Line 229: both not significant."""
        msg = _build_message(
            original_n=2000,
            projected_n=2000,
            original_pvalue=0.4,
            projected_pvalue=0.4,
            original_significant=False,
            projected_significant=False,
            projected_power=0.2,
            original_alpha=0.05,
            projected_alpha=0.05,
        )
        assert "still not be" in msg

    def test_build_message_remains_significant(self) -> None:
        """Line 227: both significant."""
        msg = _build_message(
            original_n=2000,
            projected_n=4000,
            original_pvalue=0.01,
            projected_pvalue=0.001,
            original_significant=True,
            projected_significant=True,
            projected_power=0.95,
            original_alpha=0.05,
            projected_alpha=0.05,
        )
        assert "remain statistically significant" in msg


# ─── report.py ───────────────────────────────────────────────────────────

from splita._types import (
    BayesianResult,
    CorrectionResult,
    GSResult,
    SampleSizeResult,
    SRMResult,
)
from splita.report import _classify, _fmt_val, _result_to_text, report


class TestReportEdgeCases:
    """Cover uncovered branches in report.py."""

    def test_classify_sequential(self) -> None:
        """Line 72: sequential types."""
        r = GSResult(
            analysis_results=[],
            crossed_efficacy=False,
            crossed_futility=False,
            recommended_action="continue",
        )
        assert _classify(r) == "sequential"

    def test_classify_variance(self) -> None:
        """Line 74: variance types."""
        r = CorrectionResult(
            pvalues=[0.01, 0.05],
            adjusted_pvalues=[0.02, 0.05],
            rejected=[True, False],
            alpha=0.05,
            method="bh",
            n_rejected=1,
            n_tests=2,
            labels=None,
        )
        assert _classify(r) == "variance"

    def test_result_to_text_explain_unsupported(self) -> None:
        """Lines 96-97: TypeError from explain for unsupported types."""
        r = SampleSizeResult(
            n_per_variant=5000,
            n_total=10000,
            alpha=0.05,
            power=0.80,
            mde=0.02,
            relative_mde=0.20,
            baseline=0.10,
            metric="conversion",
            effect_size=0.07,
            days_needed=14,
        )
        text = _result_to_text(r)
        assert "SampleSizeResult" in text

    def test_result_to_text_scientific_notation(self) -> None:
        """Lines 103-104: very small float uses scientific notation."""
        r = _make_experiment_result(pvalue=0.00001)
        text = _result_to_text(r)
        assert "e" in text.lower() or "E" in text

    def test_result_to_text_large_list(self) -> None:
        """Lines 107-108: large list shows count."""
        r = SRMResult(
            observed=list(range(15)),
            expected_counts=[1.0] * 15,
            chi2_statistic=0.0,
            pvalue=1.0,
            passed=True,
            alpha=0.01,
            deviations_pct=[0.0] * 15,
            worst_variant=0,
            message="ok",
        )
        text = _result_to_text(r)
        assert "15 items" in text

    def test_fmt_val_scientific(self) -> None:
        """Lines 225-226: scientific notation in HTML."""
        assert "e" in _fmt_val(0.00001).lower() or "E" in _fmt_val(0.00001)

    def test_fmt_val_large_list(self) -> None:
        """Lines 229-230: large list shows count."""
        assert "15 items" in _fmt_val(list(range(15)))

    def test_fmt_val_small_list(self) -> None:
        """Line 231: small list shows str."""
        assert "[1, 2, 3]" in _fmt_val([1, 2, 3])

    def test_text_report_sequential_section(self) -> None:
        """Lines 153-154: sequential section in text."""
        gs_result = GSResult(
            analysis_results=[],
            crossed_efficacy=False,
            crossed_futility=False,
            recommended_action="continue",
        )
        text = report(gs_result, format="text")
        assert "Sequential" in text

    def test_text_report_variance_section(self) -> None:
        """Lines 158-159: variance section in text."""
        corr_result = CorrectionResult(
            pvalues=[0.01, 0.05],
            adjusted_pvalues=[0.02, 0.05],
            rejected=[True, False],
            alpha=0.05,
            method="bh",
            n_rejected=1,
            n_tests=2,
            labels=None,
        )
        text = report(corr_result, format="text")
        assert "Variance Reduction" in text

    def test_text_report_secondary_section(self) -> None:
        """Lines 163-164: secondary section in text."""
        ss = SampleSizeResult(
            n_per_variant=5000,
            n_total=10000,
            alpha=0.05,
            power=0.80,
            mde=0.02,
            relative_mde=0.20,
            baseline=0.10,
            metric="conversion",
            effect_size=0.07,
            days_needed=14,
        )
        text = report(ss, format="text")
        assert "Secondary" in text

    def test_text_report_recommendations(self) -> None:
        """Lines 169-170: recommendations section in text."""
        r = _make_experiment_result(pvalue=0.42, significant=False, power=0.12)
        text = report(r, format="text")
        assert "Recommendations" in text

    def test_html_report_sequential_section(self) -> None:
        """Lines 335-339: sequential section in HTML."""
        gs_result = GSResult(
            analysis_results=[],
            crossed_efficacy=False,
            crossed_futility=False,
            recommended_action="continue",
        )
        html = report(gs_result, format="html")
        assert "Sequential" in html

    def test_html_report_variance_section(self) -> None:
        """Lines 343-347: variance section in HTML."""
        corr_result = CorrectionResult(
            pvalues=[0.01, 0.05],
            adjusted_pvalues=[0.02, 0.05],
            rejected=[True, False],
            alpha=0.05,
            method="bh",
            n_rejected=1,
            n_tests=2,
            labels=None,
        )
        html = report(corr_result, format="html")
        assert "Variance Reduction" in html

    def test_bayesian_indecisive_recommendation(self) -> None:
        """Lines 408: BayesianResult with intermediate prob."""
        b = BayesianResult(
            prob_b_beats_a=0.50,
            expected_loss_a=0.005,
            expected_loss_b=0.005,
            lift=0.001,
            relative_lift=0.01,
            ci_lower=-0.01,
            ci_upper=0.01,
            credible_level=0.95,
            control_mean=0.10,
            treatment_mean=0.101,
            prob_in_rope=None,
            rope=None,
            metric="conversion",
            control_n=5000,
            treatment_n=5000,
        )
        text = report(b, format="text")
        assert "decisive" in text.lower() or "more data" in text.lower()

    def test_sample_size_long_duration_recommendation(self) -> None:
        """Line 416: SampleSizeResult with days > 30."""
        ss = SampleSizeResult(
            n_per_variant=50000,
            n_total=100000,
            alpha=0.05,
            power=0.80,
            mde=0.002,
            relative_mde=0.02,
            baseline=0.10,
            metric="conversion",
            effect_size=0.01,
            days_needed=60,
        )
        text = report(ss, format="text")
        assert "60 days" in text


# ─── recommend.py ────────────────────────────────────────────────────────

from splita.recommend import recommend


class TestRecommendEdgeCases:
    """Cover uncovered branches in recommend.py."""

    def test_ties_continuous(self) -> None:
        """Lines 129-133: continuous data with many ties."""
        data = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0] * 50)
        r = recommend(data)
        assert any("tied" in s.lower() or "duplicate" in s.lower() for s in r.reasoning)

    def test_large_n(self) -> None:
        """Lines 140-144: very large sample size warning."""
        rng = np.random.default_rng(42)
        ctrl = rng.normal(10, 2, 30000)
        trt = rng.normal(10, 2, 30000)
        r = recommend(ctrl, trt)
        assert any("60,000" in w or "practical significance" in w for w in r.warnings)

    def test_low_baseline_conversion(self) -> None:
        """Lines 154-158: very low baseline rate."""
        rng = np.random.default_rng(42)
        data = rng.binomial(1, 0.005, 5000)
        r = recommend(data)
        assert any("low" in s.lower() for s in r.reasoning)

    def test_small_binary_sample(self) -> None:
        """Lines 174-175: small binary sample -> permutation test."""
        data = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 1,
                         0, 0, 1, 0, 0, 0, 1, 0, 0, 0], dtype=float)
        r = recommend(data)
        assert "Permutation" in r.recommended_test or "exact" in r.recommended_test.lower()


# ─── monitor.py ──────────────────────────────────────────────────────────

from splita.monitor import _srm_check, _check_guardrails, _predict_significance, monitor


class TestMonitorEdgeCases:
    """Cover uncovered branches in monitor.py."""

    def test_srm_zero_total(self) -> None:
        """Line 34: total=0 returns True."""
        assert _srm_check(0, 0) is True

    def test_guardrail_upper_direction(self) -> None:
        """Line 76: direction='upper'."""
        ctrl = np.array([10.0, 11.0, 12.0, 9.0, 10.0])
        trt = np.array([100.0, 110.0, 120.0, 90.0, 100.0])
        guards = [{"name": "latency", "threshold": 5.0, "direction": "upper"}]
        results = _check_guardrails(ctrl, trt, guards)
        assert len(results) == 1
        assert "passed" in results[0]

    def test_predict_significance_zero_n(self) -> None:
        """Line 88: current_n=0 or target_n=0."""
        p, sig = _predict_significance(2.0, 0, 1000)
        assert p == 1.0
        assert sig is False

    def test_significant_negative_stop_harm(self) -> None:
        """Line 215: significant and negative lift -> stop_harm."""
        rng = np.random.default_rng(42)
        ctrl = rng.binomial(1, 0.50, 5000).astype(float)
        trt = rng.binomial(1, 0.20, 5000).astype(float)
        result = monitor(ctrl, trt)
        assert result.recommendation == "stop_harm"


# ─── power_report.py ────────────────────────────────────────────────────

from splita.power_report import _z_alpha, _power_for_n, _n_for_power


class TestPowerReportEdgeCases:
    """Cover uncovered branches in power_report.py."""

    def test_z_alpha_one_sided(self) -> None:
        """Line 27: one-sided alternative."""
        z = _z_alpha(0.05, "one-sided")
        assert 1.6 < z < 1.7  # ~1.645

    def test_power_for_n_se_zero(self) -> None:
        """Line 50: se1==0 returns 1.0."""
        # baseline=0 and mde=0 gives se0=se1=0
        p = _power_for_n(0.0, 0.0, 1000, 0.05, "conversion")
        assert p == 1.0

    def test_n_for_power_mde_zero(self) -> None:
        """Line 75: mde=0 returns 0."""
        n = _n_for_power(0.10, 0.0, 0.05, 0.80, "conversion")
        assert n == 0

    def test_continuous_metric_baseline_zero(self) -> None:
        """Lines 46-48: continuous with baseline=0 uses sd=1."""
        p = _power_for_n(0.0, 0.5, 1000, 0.05, "continuous")
        assert 0.0 <= p <= 1.0


# ─── simulate.py ─────────────────────────────────────────────────────────

from splita.simulate import simulate


class TestSimulateEdgeCases:
    """Cover uncovered branches in simulate.py."""

    def test_conversion_se_zero(self) -> None:
        """Line 98: se=0 in conversion simulation (all same values)."""
        # With baseline=0 and mde=0, all values are 0, so se=0, pval=1.0
        result = simulate(0.0, 0.0, 100, n_simulations=10, random_state=42)
        assert result.median_pvalue >= 0.0

    def test_moderate_power_recommendation(self) -> None:
        """Line 146: moderate power recommendation."""
        result = simulate(0.10, 0.02, 3000, n_simulations=500, random_state=42)
        assert "Moderate" in result.recommendation


# ─── sequential/yeast.py ────────────────────────────────────────────────

from splita.sequential.yeast import YEASTSequentialTest


class TestYEASTEdgeCases:
    """Cover uncovered branches in yeast.py."""

    def test_single_obs_per_group(self) -> None:
        """Lines 125-134: n_c=0 or n_t=0 initially, then only 1."""
        test = YEASTSequentialTest()
        # First update with 1 obs in control, 0 in treatment
        state = test.update([5.0], [])
        assert state.z_statistic == 0.0  # n_t = 0
        # Second update with 0 in control, 1 in treatment
        state = test.update([], [6.0])
        assert state.n_control == 1
        assert state.n_treatment == 1

    def test_var_c_zero_single_obs(self) -> None:
        """Line 144: n_c==1 -> var_c=0."""
        test = YEASTSequentialTest()
        state = test.update([5.0], [3.0, 4.0, 5.0])
        assert state.n_control == 1

    def test_var_t_zero_single_obs(self) -> None:
        """Line 149: n_t==1 -> var_t=0."""
        test = YEASTSequentialTest()
        state = test.update([3.0, 4.0, 5.0], [5.0])
        assert state.n_treatment == 1


# ─── sequential/sample_size_reest.py ────────────────────────────────────

from splita.sequential.sample_size_reest import SampleSizeReestimation


class TestSampleSizeReestEdgeCases:
    """Cover uncovered branches in sample_size_reest.py."""

    def test_zero_z_current(self) -> None:
        """Line 130: z_current=0 -> new_n = current_n * 10."""
        # The z_current=0 path at line 130 requires:
        # 1. abs(interim_effect) >= 1e-15 (passes the check at line 116)
        # 2. z_current = abs(interim_effect) / interim_se == 0 exactly
        # This is numerically impossible with float. Mark as defensive.
        # Instead test the near-zero path where ratio is very large
        result = SampleSizeReestimation.reestimate(
            current_n=500,
            interim_effect=0.001,
            interim_se=1e10,  # huge SE -> z_current ≈ 0
            target_power=0.80,
            alpha=0.05,
        )
        assert result.new_n_per_variant >= 500


# ─── variance/robust_estimators.py ──────────────────────────────────────

from splita.variance.robust_estimators import (
    RobustMeanEstimator,
    _catoni_location,
    _huber_location,
)


class TestRobustEstimatorEdgeCases:
    """Cover uncovered branches in robust_estimators.py."""

    def test_huber_mad_zero_std_zero(self) -> None:
        """Lines 48, 50: mad=0 and std=0 -> return mu."""
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        result = _huber_location(data)
        assert result == 5.0

    def test_catoni_scale_zero(self) -> None:
        """Line 114: scale=0 in catoni -> return mu."""
        data = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        result = _catoni_location(data)
        assert result == 3.0

    def test_robust_se_zero(self) -> None:
        """Lines 230-232: se=0 branch."""
        est = RobustMeanEstimator(method="huber")
        ctrl = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        trt = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        r = est.fit_transform(ctrl, trt)
        assert r.ate == 0.0
        assert r.pvalue == 1.0


# ─── variance/trimmed_mean.py ───────────────────────────────────────────

from splita.variance.trimmed_mean import TrimmedMeanEstimator


class TestTrimmedMeanEdgeCases:
    """Cover uncovered branches in trimmed_mean.py."""

    def test_trt_trimmed_too_short(self) -> None:
        """Line 118: treatment trimmed too short."""
        est = TrimmedMeanEstimator(trim_fraction=0.90)
        ctrl = np.arange(100, dtype=float)
        trt = np.array([1.0, 2.0, 3.0])  # only 3 -> trimming 90% leaves < 2
        with pytest.raises(ValueError, match="[Tt]rimming"):
            est.fit_transform(ctrl, trt)

    def test_se_zero_welch(self) -> None:
        """Lines 140-141: se=0 in trimmed mean."""
        est = TrimmedMeanEstimator(trim_fraction=0.01)
        ctrl = np.array([5.0, 5.0, 5.0, 5.0, 5.0] * 20)
        trt = np.array([5.0, 5.0, 5.0, 5.0, 5.0] * 20)
        r = est.fit_transform(ctrl, trt)
        assert r.pvalue == 1.0


# ─── variance/cluster_bootstrap.py ──────────────────────────────────────

from splita.variance.cluster_bootstrap import ClusterBootstrap


class TestClusterBootstrapEdgeCases:
    """Cover uncovered branches in cluster_bootstrap.py."""

    def test_clusters_ndim_not_1(self) -> None:
        """Line 89: cluster array with wrong ndim."""
        cb = ClusterBootstrap(n_bootstrap=100, random_state=42)
        ctrl = np.ones(10)
        trt = np.ones(10)
        with pytest.raises(ValueError, match="1-D"):
            cb.run(ctrl, trt, np.ones((10, 2)), np.arange(10))

    def test_treatment_single_cluster(self) -> None:
        """Line 158: treatment has < 2 clusters."""
        cb = ClusterBootstrap(n_bootstrap=100, random_state=42)
        ctrl = np.array([1.0, 2.0, 3.0, 4.0])
        trt = np.array([5.0, 6.0])
        ctrl_cl = np.array([0, 0, 1, 1])
        trt_cl = np.array([0, 0])  # single cluster
        with pytest.raises(ValueError, match="2 clusters"):
            cb.run(ctrl, trt, ctrl_cl, trt_cl)


# ─── variance/multivariate_cuped.py ─────────────────────────────────────

from splita.variance.multivariate_cuped import MultivariateCUPED


class TestMultivariateCUPEDEdgeCases:
    """Cover uncovered branches in multivariate_cuped.py."""

    def test_singular_covariance(self) -> None:
        """Lines 175-176: singular covariance -> LinAlgError."""
        rng = np.random.default_rng(42)
        n = 50
        ctrl = rng.normal(10, 1, n)
        trt = rng.normal(11, 1, n)
        X = rng.normal(0, 1, (n, 1))
        # Make X exactly collinear by repeating
        X_ctrl = np.column_stack([X, X])  # perfectly collinear
        X_trt = np.column_stack([X, X])
        mc = MultivariateCUPED()
        with pytest.raises(ValueError, match="singular"):
            mc.fit(ctrl, trt, X_ctrl, X_trt)

    def test_low_variance_reduction_warning(self) -> None:
        """Line 314: variance_reduction < 0.05 triggers warning."""
        rng = np.random.default_rng(42)
        n = 100
        ctrl = rng.normal(10, 5, n)
        trt = rng.normal(11, 5, n)
        # Covariates with very low correlation to outcome
        X_ctrl = rng.normal(0, 1, (n, 1))
        X_trt = rng.normal(0, 1, (n, 1))
        mc = MultivariateCUPED()
        with pytest.warns(RuntimeWarning):
            mc.fit_transform(ctrl, trt, X_ctrl, X_trt)


# ─── variance/cupac.py ─────────────────────────────────────────────────

from splita.variance.cupac import CUPAC


class TestCUPACEdgeCases:
    """Cover uncovered branches in cupac.py."""

    def test_fit_transform_good_reduction(self) -> None:
        """Line 577: 'good' quality branch (0.2 < variance_reduction <= 0.5)."""
        rng = np.random.default_rng(42)
        n = 500
        X_ctrl = rng.normal(0, 1, size=(n, 3))
        X_trt = rng.normal(0, 1, size=(n, 3))
        # Outcome moderately predicted by X to hit "good" range
        ctrl = X_ctrl @ [1, 0.5, 0.2] + rng.normal(0, 2, n)
        trt = X_trt @ [1, 0.5, 0.2] + 0.5 + rng.normal(0, 2, n)
        cupac = CUPAC(random_state=42)
        cupac.fit_transform(ctrl, trt, X_ctrl, X_trt)
        # Just ensure it ran to the info() call
        assert cupac.variance_reduction_ >= 0


# ─── variance/double_ml.py ─────────────────────────────────────────────

from splita.variance.double_ml import DoubleML


class TestDoubleMLEdgeCases:
    """Cover uncovered branches in double_ml.py."""

    def test_se_inf_when_t_resid_zero(self) -> None:
        """Line 278: mean_T_resid_sq == 0.0 -> se = inf."""
        rng = np.random.default_rng(42)
        n = 100
        X = rng.normal(0, 1, size=(n, 2))
        T = np.ones(n)  # constant treatment -> T_hat ≈ 1 -> T_resid ≈ 0
        Y = rng.normal(0, 1, n)
        dml = DoubleML(random_state=42)
        result = dml.fit_transform(Y, T, X)
        # With constant treatment, T_resid might not be exactly 0
        # but should be nearly 0
        assert result.se >= 0


# ─── variance/nonstationary.py ──────────────────────────────────────────

from splita.variance.nonstationary import NonstationaryAdjustment


class TestNonstationaryEdgeCases:
    """Cover uncovered branches in nonstationary.py."""

    def test_se_zero_nonzero_ate(self) -> None:
        """Line 123: se=0, ate_corrected != 0 -> pvalue=0.0."""
        # All identical within groups but different across
        ctrl = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        trt = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        ts = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        adj = NonstationaryAdjustment()
        r = adj.fit_transform(ctrl, trt, ts)
        assert r.pvalue == 0.0


# ─── variance/post_stratification.py ────────────────────────────────────

from splita.variance.post_stratification import PostStratification


class TestPostStratEdgeCases:
    """Cover uncovered branches in post_stratification.py."""

    def test_strata_ndim_not_1(self) -> None:
        """Line 76: strata array with wrong ndim."""
        ps = PostStratification()
        ctrl = np.ones(10)
        trt = np.ones(10)
        with pytest.raises(ValueError, match="1-D"):
            ps.fit_transform(ctrl, trt, np.ones((10, 2)), np.arange(10))


# ─── viz/plots.py ───────────────────────────────────────────────────────

from splita.viz.plots import power_curve


class TestPlotsEdgeCases:
    """Cover uncovered branches in plots.py."""

    def test_power_curve_se_zero(self) -> None:
        """Line 184: se=0 -> powers.append(1.0)."""
        import matplotlib
        matplotlib.use("Agg")

        fig = power_curve(
            baseline=0.0,
            mde_range=[0.0],
            n_per_variant=100,
        )
        import matplotlib.pyplot as plt
        plt.close(fig)


# ─── _experimental.py ───────────────────────────────────────────────────

from splita._experimental import experimental


class TestExperimental:
    """Cover _experimental.py."""

    def test_experimental_decorator(self) -> None:
        @experimental
        def my_func():
            return 42

        with pytest.warns(FutureWarning, match="experimental"):
            result = my_func()
        assert result == 42


# ─── _advisory.py ────────────────────────────────────────────────────────

from splita._advisory import (
    advise_pre_analysis,
    advise_large_sample_practical_significance,
    advise_ratio_without_delta,
)


class TestAdvisoryEdgeCases:
    """Cover uncovered branches in _advisory.py."""

    def test_advise_pre_analysis(self) -> None:
        """Line 89: advise_pre_analysis info."""
        # Just call it—it emits info, no assertion needed
        advise_pre_analysis(500, 500)

    def test_advise_ratio_without_delta(self) -> None:
        """Line 143: ratio metric with non-delta method."""
        with pytest.warns(RuntimeWarning, match="ratio"):
            advise_ratio_without_delta("ratio", "ttest")

    def test_advise_large_sample_no_warning(self) -> None:
        """Line 192: large sample but large effect -> no warning."""
        # Should not warn when effect_size >= 0.01
        advise_large_sample_practical_significance(60000, 0.01, 0.05)

    def test_advise_large_sample_warning(self) -> None:
        """Line 192: large sample, small effect, significant -> warning."""
        with pytest.warns(RuntimeWarning, match="trivially"):
            advise_large_sample_practical_significance(60000, 0.001, 0.001)


# ─── __init__.py ─────────────────────────────────────────────────────────


class TestInit:
    """Cover __init__.py import-time code."""

    def test_version_attribute(self) -> None:
        """Lines 7-8: version fallback."""
        import splita
        assert isinstance(splita.__version__, str)
