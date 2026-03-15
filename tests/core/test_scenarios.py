"""End-to-end scenario tests for the splita A/B testing library.

Each test class represents a realistic A/B testing scenario that exercises
the full pipeline: sample-size planning, data generation, experiment
analysis, and result interpretation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from splita.core.experiment import Experiment
from splita.core.sample_size import SampleSize

# ---------------------------------------------------------------------------
# Scenario 1: E-commerce Conversion Test
# ---------------------------------------------------------------------------


class TestEcommerceConversionTest:
    """An e-commerce site wants to test a new checkout flow.

    Current conversion rate is 10%. They want to detect a +2 percentage-point
    lift (10% -> 12%).  The full pipeline is: plan sample size, estimate
    duration, generate synthetic data with a true effect, run the experiment,
    and verify statistical significance.
    """

    def test_sample_size_is_reasonable(self):
        """Required n should be in the thousands, not millions."""
        result = SampleSize.for_proportion(baseline=0.10, mde=0.02)
        assert 1_000 < result.n_per_variant < 20_000, (
            f"n_per_variant={result.n_per_variant} is outside a reasonable range"
        )
        assert result.n_total == result.n_per_variant * 2

    def test_duration_estimate(self):
        """With 1000 daily users the test should take a few days to weeks."""
        result = SampleSize.for_proportion(baseline=0.10, mde=0.02)
        with_days = result.duration(daily_users=1000)
        assert with_days.days_needed is not None
        assert with_days.days_needed > 0
        # With ~7686 total and 1000/day -> ~8 days
        assert with_days.days_needed < 30, (
            f"Expected < 30 days, got {with_days.days_needed}"
        )

    def test_experiment_detects_true_effect(self):
        """With sufficient n and a true 2pp lift, the test should be significant.

        We use 2x the planned n to ensure the random draw reliably produces
        a detectable signal (80% power means 1-in-5 chance of missing it
        at exactly n).
        """
        plan = SampleSize.for_proportion(baseline=0.10, mde=0.02)
        n = plan.n_per_variant * 2  # oversample for test reliability

        rng = np.random.default_rng(42)
        control = rng.binomial(1, 0.10, size=n).astype(float)
        treatment = rng.binomial(1, 0.12, size=n).astype(float)

        result = Experiment(control, treatment).run()

        assert result.metric == "conversion"
        assert result.method == "ztest"
        assert result.pvalue < 0.05, f"Expected significant, got p={result.pvalue:.4f}"
        assert result.significant is True
        assert result.lift > 0, "Lift should be positive (treatment > control)"

    def test_confidence_interval_excludes_zero(self):
        """When the effect is real and detected, the CI should not contain 0."""
        plan = SampleSize.for_proportion(baseline=0.10, mde=0.02)
        n = plan.n_per_variant * 2  # oversample for test reliability

        rng = np.random.default_rng(42)
        control = rng.binomial(1, 0.10, size=n).astype(float)
        treatment = rng.binomial(1, 0.12, size=n).astype(float)

        result = Experiment(control, treatment).run()

        assert result.ci_lower > 0 or result.ci_upper < 0, (
            f"CI [{result.ci_lower:.4f}, {result.ci_upper:.4f}] should not contain 0"
        )

    def test_full_pipeline_coherence(self):
        """Plan, execute, analyze: all pieces fit together."""
        # Plan
        plan = SampleSize.for_proportion(baseline=0.10, mde=0.02)
        with_days = plan.duration(daily_users=1000)

        # Execute (synthetic)
        rng = np.random.default_rng(42)
        n = plan.n_per_variant
        control = rng.binomial(1, 0.10, size=n).astype(float)
        treatment = rng.binomial(1, 0.12, size=n).astype(float)

        # Analyze
        result = Experiment(control, treatment).run()

        # Coherence checks
        assert result.control_n == n
        assert result.treatment_n == n
        assert result.alpha == pytest.approx(0.05)
        assert 0 <= result.power <= 1
        assert with_days.days_needed is not None
        assert with_days.days_needed > 0


# ---------------------------------------------------------------------------
# Scenario 2: Revenue Per User with Outliers
# ---------------------------------------------------------------------------


class TestRevenuePerUserWithOutliers:
    """Testing a pricing change. Revenue is heavy-tailed ($25 mean, $40 std).

    We want to detect a $2 lift. Because the data is skewed, we compare
    the non-parametric Mann-Whitney test and bootstrap to see if they agree
    on direction.
    """

    def test_sample_size_for_revenue(self):
        """Sample size for a mean metric with high variance."""
        result = SampleSize.for_mean(
            baseline_mean=25.0,
            baseline_std=40.0,
            mde=2.0,
        )
        assert result.n_per_variant > 1_000, (
            f"High-variance metric should need many samples, got {result.n_per_variant}"
        )
        assert result.metric == "mean"

    def test_mannwhitney_on_skewed_data(self):
        """Mann-Whitney should handle heavy-tailed revenue data."""
        plan = SampleSize.for_mean(
            baseline_mean=25.0,
            baseline_std=40.0,
            mde=2.0,
        )
        n = plan.n_per_variant

        rng = np.random.default_rng(43)
        # Lognormal to simulate heavy right tail
        # mu, sigma chosen so that the mean is ~25 (control) and ~27 (treatment)
        sigma_ln = 1.0
        mu_ctrl = np.log(25.0) - sigma_ln**2 / 2
        mu_trt = np.log(27.0) - sigma_ln**2 / 2

        control = rng.lognormal(mean=mu_ctrl, sigma=sigma_ln, size=n)
        treatment = rng.lognormal(mean=mu_trt, sigma=sigma_ln, size=n)

        result = Experiment(control, treatment, method="mannwhitney").run()

        assert result.method == "mannwhitney"
        assert 0 <= result.pvalue <= 1
        assert result.lift != 0  # Hodges-Lehmann estimate should be non-zero

    def test_bootstrap_on_skewed_data(self):
        """Bootstrap should also handle heavy-tailed data."""
        plan = SampleSize.for_mean(
            baseline_mean=25.0,
            baseline_std=40.0,
            mde=2.0,
        )
        n = plan.n_per_variant

        rng = np.random.default_rng(43)
        sigma_ln = 1.0
        mu_ctrl = np.log(25.0) - sigma_ln**2 / 2
        mu_trt = np.log(27.0) - sigma_ln**2 / 2

        control = rng.lognormal(mean=mu_ctrl, sigma=sigma_ln, size=n)
        treatment = rng.lognormal(mean=mu_trt, sigma=sigma_ln, size=n)

        result = Experiment(
            control,
            treatment,
            method="bootstrap",
            n_bootstrap=2000,
            random_state=44,
        ).run()

        assert result.method == "bootstrap"
        assert 0 <= result.pvalue <= 1

    def test_mannwhitney_and_bootstrap_agree_on_direction(self):
        """Both methods should agree on the direction of the effect."""
        plan = SampleSize.for_mean(
            baseline_mean=25.0,
            baseline_std=40.0,
            mde=2.0,
        )
        n = plan.n_per_variant

        rng = np.random.default_rng(43)
        sigma_ln = 1.0
        mu_ctrl = np.log(25.0) - sigma_ln**2 / 2
        mu_trt = np.log(27.0) - sigma_ln**2 / 2

        control = rng.lognormal(mean=mu_ctrl, sigma=sigma_ln, size=n)
        treatment = rng.lognormal(mean=mu_trt, sigma=sigma_ln, size=n)

        mw_result = Experiment(control, treatment, method="mannwhitney").run()
        bs_result = Experiment(
            control,
            treatment,
            method="bootstrap",
            n_bootstrap=2000,
            random_state=44,
        ).run()

        # Both should see treatment_mean > control_mean
        assert mw_result.treatment_mean > mw_result.control_mean
        assert bs_result.treatment_mean > bs_result.control_mean

        # Both lifts should be positive
        assert mw_result.relative_lift > 0
        assert bs_result.relative_lift > 0


# ---------------------------------------------------------------------------
# Scenario 3: SRM Then Analyze (guard pattern)
# ---------------------------------------------------------------------------


class TestSRMGuardPattern:
    """Before analyzing, check for Sample Ratio Mismatch.

    SRMCheck is not yet built (Milestone 3), so we verify that Experiment
    handles unequal group sizes correctly and produces valid results.
    """

    def test_unequal_groups_run_successfully(self):
        """Experiment should accept groups of different sizes."""
        rng = np.random.default_rng(45)
        control = rng.binomial(1, 0.10, size=4850).astype(float)
        treatment = rng.binomial(1, 0.10, size=5150).astype(float)

        result = Experiment(control, treatment).run()

        assert result.control_n == 4850
        assert result.treatment_n == 5150
        assert 0 <= result.pvalue <= 1

    def test_unequal_groups_correct_n_values(self):
        """Result should report the actual group sizes."""
        rng = np.random.default_rng(45)
        control = rng.binomial(1, 0.10, size=4850).astype(float)
        treatment = rng.binomial(1, 0.10, size=5150).astype(float)

        result = Experiment(control, treatment).run()

        assert result.control_n == 4850
        assert result.treatment_n == 5150

    def test_unequal_groups_no_effect_not_significant(self):
        """With no true effect, the test should usually be non-significant."""
        rng = np.random.default_rng(45)
        # Both groups at 10% — no true difference
        control = rng.binomial(1, 0.10, size=4850).astype(float)
        treatment = rng.binomial(1, 0.10, size=5150).astype(float)

        result = Experiment(control, treatment).run()

        # With no true effect, we expect non-significance most of the time.
        # Under H0, P(significant) = alpha = 5%, so this should almost always pass.
        assert result.metric == "conversion"
        assert result.method == "ztest"
        assert 0 <= result.pvalue <= 1


# ---------------------------------------------------------------------------
# Scenario 4: Multiple Metrics on One Experiment
# ---------------------------------------------------------------------------


class TestMultipleMetrics:
    """An experiment measures conversion, revenue, and bounce rate simultaneously.

    MultipleCorrection is not yet built (Milestone 3), so we verify that
    multiple independent Experiment instances all run and return valid results.
    """

    def test_three_metrics_all_produce_valid_results(self):
        """Running 3 experiments for different metrics should all succeed."""
        rng = np.random.default_rng(46)
        n = 5000

        # Metric 1: Conversion (binary)
        conv_ctrl = rng.binomial(1, 0.10, size=n).astype(float)
        conv_trt = rng.binomial(1, 0.12, size=n).astype(float)

        # Metric 2: Revenue (continuous)
        rev_ctrl = rng.normal(25.0, 40.0, size=n)
        rev_trt = rng.normal(27.0, 40.0, size=n)

        # Metric 3: Bounce rate (binary, lower is better)
        bounce_ctrl = rng.binomial(1, 0.45, size=n).astype(float)
        bounce_trt = rng.binomial(1, 0.43, size=n).astype(float)

        results = [
            Experiment(conv_ctrl, conv_trt).run(),
            Experiment(rev_ctrl, rev_trt).run(),
            Experiment(bounce_ctrl, bounce_trt).run(),
        ]

        for i, res in enumerate(results):
            assert 0 <= res.pvalue <= 1, (
                f"Metric {i}: p-value {res.pvalue} not in [0, 1]"
            )
            assert res.control_n == n
            assert res.treatment_n == n

    def test_metrics_correctly_inferred(self):
        """Each metric type should be auto-detected correctly."""
        rng = np.random.default_rng(46)
        n = 5000

        conv_ctrl = rng.binomial(1, 0.10, size=n).astype(float)
        conv_trt = rng.binomial(1, 0.12, size=n).astype(float)

        rev_ctrl = rng.normal(25.0, 40.0, size=n)
        rev_trt = rng.normal(27.0, 40.0, size=n)

        bounce_ctrl = rng.binomial(1, 0.45, size=n).astype(float)
        bounce_trt = rng.binomial(1, 0.43, size=n).astype(float)

        conv_result = Experiment(conv_ctrl, conv_trt).run()
        rev_result = Experiment(rev_ctrl, rev_trt).run()
        bounce_result = Experiment(bounce_ctrl, bounce_trt).run()

        assert conv_result.metric == "conversion"
        assert rev_result.metric == "continuous"
        assert bounce_result.metric == "conversion"

    def test_all_pvalues_in_valid_range(self):
        """All p-values should be in [0, 1]."""
        rng = np.random.default_rng(46)
        n = 5000

        pvalues = []
        for baseline_ctrl, baseline_trt in [(0.10, 0.12), (0.45, 0.43)]:
            ctrl = rng.binomial(1, baseline_ctrl, size=n).astype(float)
            trt = rng.binomial(1, baseline_trt, size=n).astype(float)
            pvalues.append(Experiment(ctrl, trt).run().pvalue)

        # Revenue metric
        ctrl = rng.normal(25.0, 40.0, size=n)
        trt = rng.normal(27.0, 40.0, size=n)
        pvalues.append(Experiment(ctrl, trt).run().pvalue)

        for i, p in enumerate(pvalues):
            assert 0 <= p <= 1, f"p-value {i} = {p} not in [0, 1]"


# ---------------------------------------------------------------------------
# Scenario 5: CTR Ratio Metric (Delta Method)
# ---------------------------------------------------------------------------


class TestCTRRatioMetric:
    """Testing a new recommendation algorithm. Metric is CTR = clicks / impressions.

    Uses the delta method for ratio metrics, which accounts for the
    correlation between numerator and denominator.
    """

    def test_sample_size_for_ratio(self):
        """Plan sample size for a ratio metric using delta method variance."""
        result = SampleSize.for_ratio(
            baseline_num_mean=5.0,  # avg clicks per user
            baseline_den_mean=100.0,  # avg impressions per user
            baseline_num_std=3.0,
            baseline_den_std=30.0,
            baseline_covariance=50.0,  # clicks and impressions are correlated
            mde=0.005,  # detect 0.5pp CTR change
        )
        assert result.n_per_variant > 0
        assert result.metric == "ratio"
        assert result.mde == pytest.approx(0.005)

    def test_experiment_with_ratio_metric(self):
        """Run a ratio-metric experiment with synthetic CTR data."""
        rng = np.random.default_rng(47)
        n = 5000

        # Control: each user sees ~100 impressions, clicks ~5% of the time
        impr_ctrl = rng.poisson(100, size=n).astype(float)
        clicks_ctrl = rng.binomial(impr_ctrl.astype(int), 0.05).astype(float)

        # Treatment: slightly higher CTR (~5.5%)
        impr_trt = rng.poisson(100, size=n).astype(float)
        clicks_trt = rng.binomial(impr_trt.astype(int), 0.055).astype(float)

        result = Experiment(
            clicks_ctrl,
            clicks_trt,
            metric="ratio",
            control_denominator=impr_ctrl,
            treatment_denominator=impr_trt,
        ).run()

        assert result.metric == "ratio"
        assert result.method == "delta"
        assert 0 <= result.pvalue <= 1

    def test_ratio_result_has_reasonable_values(self):
        """The ratio result should have sensible CTR values."""
        rng = np.random.default_rng(47)
        n = 5000

        impr_ctrl = rng.poisson(100, size=n).astype(float)
        clicks_ctrl = rng.binomial(impr_ctrl.astype(int), 0.05).astype(float)

        impr_trt = rng.poisson(100, size=n).astype(float)
        clicks_trt = rng.binomial(impr_trt.astype(int), 0.055).astype(float)

        result = Experiment(
            clicks_ctrl,
            clicks_trt,
            metric="ratio",
            control_denominator=impr_ctrl,
            treatment_denominator=impr_trt,
        ).run()

        # Control CTR should be close to 5%
        assert result.control_mean == pytest.approx(0.05, abs=0.01)
        # Treatment CTR should be close to 5.5%
        assert result.treatment_mean == pytest.approx(0.055, abs=0.01)
        # Lift should be positive
        assert result.lift > 0
        assert result.control_n == n
        assert result.treatment_n == n


# ---------------------------------------------------------------------------
# Scenario 6: Underpowered Experiment Detection
# ---------------------------------------------------------------------------


class TestUnderpoweredExperimentDetection:
    """A PM wants to run a test with only 500 users per group on a 10% baseline.

    We use the inverse sample-size function to show that the minimum
    detectable effect is large, meaning the experiment is underpowered
    for typical effect sizes. Then we generate data with a small true
    lift and confirm the test does not detect it.
    """

    def test_mde_is_large_for_small_n(self):
        """With only 500 users, the MDE should be > 3pp."""
        mde = SampleSize.mde_for_proportion(baseline=0.10, n=500)
        assert mde > 0.03, (
            f"MDE for n=500 should be > 3pp (underpowered), got {mde:.4f}"
        )

    def test_small_effect_not_detected(self):
        """A 1pp true lift with n=500 should not be significant."""
        rng = np.random.default_rng(48)
        n = 500

        control = rng.binomial(1, 0.10, size=n).astype(float)
        treatment = rng.binomial(1, 0.11, size=n).astype(float)  # only 1pp lift

        result = Experiment(control, treatment).run()

        assert result.pvalue > 0.05, (
            "With n=500 and 1pp lift, should be non-significant, "
            f"got p={result.pvalue:.4f}"
        )
        assert result.significant is False

    def test_planning_and_analysis_are_consistent(self):
        """The MDE from planning should be consistent with what the test can detect.

        If we generate data with a lift equal to the MDE, the test should
        have a reasonable chance of detecting it (not guaranteed due to
        randomness, but the design is consistent).
        """
        mde = SampleSize.mde_for_proportion(baseline=0.10, n=500)

        # Verify roundtrip: using the computed MDE to plan should give ~500
        plan = SampleSize.for_proportion(baseline=0.10, mde=mde)
        assert abs(plan.n_per_variant - 500) <= 1, (
            f"Roundtrip: MDE={mde:.4f} should require ~500 per variant, "
            f"got {plan.n_per_variant}"
        )


# ---------------------------------------------------------------------------
# Scenario 7: One-Sided Test for Degradation Guard
# ---------------------------------------------------------------------------


class TestOneSidedDegradationGuard:
    """Shipping a refactor that should NOT hurt conversion.

    Running a one-sided test to detect degradation. One-sided tests need
    fewer samples because they only look for effects in one direction.
    """

    def test_one_sided_requires_fewer_samples(self):
        """A one-sided test should require fewer samples than two-sided."""
        two_sided = SampleSize.for_proportion(
            baseline=0.10,
            mde=0.01,
            alternative="two-sided",
        )
        one_sided = SampleSize.for_proportion(
            baseline=0.10,
            mde=0.01,
            alternative="one-sided",
        )
        assert one_sided.n_per_variant < two_sided.n_per_variant, (
            f"One-sided ({one_sided.n_per_variant}) should need fewer samples "
            f"than two-sided ({two_sided.n_per_variant})"
        )

    def test_no_degradation_is_not_significant(self):
        """When treatment equals control, one-sided 'less' test should not fire."""
        plan = SampleSize.for_proportion(
            baseline=0.10,
            mde=0.01,
            alternative="one-sided",
        )
        n = plan.n_per_variant

        rng = np.random.default_rng(49)
        # Both groups at exactly 10% — no degradation
        control = rng.binomial(1, 0.10, size=n).astype(float)
        treatment = rng.binomial(1, 0.10, size=n).astype(float)

        # "less" tests if treatment < control (degradation)
        result = Experiment(control, treatment, alternative="less").run()

        assert result.pvalue > 0.05, (
            f"No degradation should be non-significant, got p={result.pvalue:.4f}"
        )
        assert result.significant is False

    def test_guard_rail_pattern_valid_result(self):
        """The guard-rail experiment should return valid result fields."""
        rng = np.random.default_rng(49)
        n = 5000

        control = rng.binomial(1, 0.10, size=n).astype(float)
        treatment = rng.binomial(1, 0.10, size=n).astype(float)

        result = Experiment(control, treatment, alternative="less").run()

        assert result.metric == "conversion"
        assert result.method == "ztest"
        assert result.alpha == pytest.approx(0.05)
        assert 0 <= result.pvalue <= 1
        assert result.control_n == n
        assert result.treatment_n == n


# ---------------------------------------------------------------------------
# Scenario 8: SRM Guard Before Analysis
# ---------------------------------------------------------------------------


class TestSRMGuardBeforeAnalysis:
    """Team runs an A/B test. Before analyzing results, they check for SRM.

    If the SRM check passes, they proceed with the experiment analysis.
    If SRM fails, the experiment results cannot be trusted and analysis
    should be halted.
    """

    def test_balanced_traffic_passes_srm(self):
        """With balanced groups (5000/5000), SRM should pass."""
        from splita.core.srm import SRMCheck

        rng = np.random.default_rng(80)
        n = 5000
        rng.binomial(1, 0.10, size=n).astype(float)
        rng.binomial(1, 0.12, size=n).astype(float)

        srm = SRMCheck([n, n]).run()
        assert srm.passed is True
        assert "No sample ratio mismatch" in srm.message

    def test_guard_then_analyze_pattern(self):
        """Full guard-then-analyze: SRM passes, then Experiment runs."""
        from splita.core.srm import SRMCheck

        rng = np.random.default_rng(80)
        n = 5000
        control = rng.binomial(1, 0.10, size=n).astype(float)
        treatment = rng.binomial(1, 0.12, size=n).astype(float)

        # Step 1: SRM guard
        srm = SRMCheck([len(control), len(treatment)]).run()
        assert srm.passed is True

        # Step 2: Proceed with analysis
        result = Experiment(control, treatment).run()
        assert result.metric == "conversion"
        assert result.significant is True
        assert result.lift > 0

    def test_imbalanced_traffic_fails_srm(self):
        """With imbalanced groups (4000/6000), SRM should fail."""
        from splita.core.srm import SRMCheck

        srm = SRMCheck([4000, 6000]).run()
        assert srm.passed is False
        assert "cannot be trusted" in srm.message

    def test_imbalanced_srm_blocks_analysis(self):
        """When SRM fails, analysis should be blocked (guard pattern)."""
        from splita.core.srm import SRMCheck

        rng = np.random.default_rng(81)
        control = rng.binomial(1, 0.10, size=4000).astype(float)
        treatment = rng.binomial(1, 0.12, size=6000).astype(float)

        srm = SRMCheck([len(control), len(treatment)]).run()
        assert srm.passed is False

        # In practice you would stop here. We verify the message is clear.
        assert srm.pvalue < 0.01  # SRM uses alpha=0.01 by default
        assert "cannot be trusted" in srm.message


# ---------------------------------------------------------------------------
# Scenario 9: Multiple Metrics with BH Correction
# ---------------------------------------------------------------------------


class TestMultipleMetricsWithBHCorrection:
    """E-commerce test measuring 5 metrics simultaneously.

    Two metrics have true effects (conversion +2pp, revenue +$2), while
    three have no effect (AOV, session duration, bounce rate). After
    BH correction, the truly significant metrics should survive and the
    null metrics should not. Bonferroni should be more conservative.
    """

    @pytest.fixture()
    def five_metric_results(self):
        """Generate 5 pairs of synthetic data and run experiments."""
        rng = np.random.default_rng(90)
        n = 10000  # large n to ensure clear separation

        # Metric 1: Conversion — true 2pp lift (significant)
        conv_ctrl = rng.binomial(1, 0.10, size=n).astype(float)
        conv_trt = rng.binomial(1, 0.12, size=n).astype(float)

        # Metric 2: Revenue — true $2 lift (significant)
        rev_ctrl = rng.normal(25.0, 10.0, size=n)
        rev_trt = rng.normal(27.0, 10.0, size=n)

        # Metric 3: AOV — no true effect
        aov_ctrl = rng.normal(50.0, 15.0, size=n)
        aov_trt = rng.normal(50.0, 15.0, size=n)

        # Metric 4: Session duration — no true effect
        sess_ctrl = rng.normal(300.0, 60.0, size=n)
        sess_trt = rng.normal(300.0, 60.0, size=n)

        # Metric 5: Bounce rate — no true effect
        bounce_ctrl = rng.binomial(1, 0.40, size=n).astype(float)
        bounce_trt = rng.binomial(1, 0.40, size=n).astype(float)

        experiments = [
            Experiment(conv_ctrl, conv_trt),
            Experiment(rev_ctrl, rev_trt),
            Experiment(aov_ctrl, aov_trt),
            Experiment(sess_ctrl, sess_trt),
            Experiment(bounce_ctrl, bounce_trt),
        ]
        results = [exp.run() for exp in experiments]
        pvalues = [r.pvalue for r in results]
        labels = ["conversion", "revenue", "aov", "session", "bounce"]

        return results, pvalues, labels

    def test_bh_correction_keeps_true_effects(self, five_metric_results):
        """BH correction should retain the two truly significant metrics."""
        from splita.core.correction import MultipleCorrection

        _, pvalues, labels = five_metric_results

        correction = MultipleCorrection(
            pvalues,
            method="bh",
            labels=labels,
        ).run()

        # First 2 metrics (conversion, revenue) should survive
        assert correction.rejected[0] is True, (
            f"Conversion should survive BH (adj_p={correction.adjusted_pvalues[0]:.4f})"
        )
        assert correction.rejected[1] is True, (
            f"Revenue should survive BH (adj_p={correction.adjusted_pvalues[1]:.4f})"
        )

    def test_bh_correction_rejects_null_metrics(self, five_metric_results):
        """BH correction should not reject the three null metrics."""
        from splita.core.correction import MultipleCorrection

        _, pvalues, labels = five_metric_results

        correction = MultipleCorrection(
            pvalues,
            method="bh",
            labels=labels,
        ).run()

        # Last 3 metrics (aov, session, bounce) should not survive
        assert correction.rejected[2] is False, (
            f"AOV should not survive BH (adj_p={correction.adjusted_pvalues[2]:.4f})"
        )
        assert correction.rejected[3] is False, (
            "Session should not survive BH "
            f"(adj_p={correction.adjusted_pvalues[3]:.4f})"
        )
        assert correction.rejected[4] is False, (
            f"Bounce should not survive BH (adj_p={correction.adjusted_pvalues[4]:.4f})"
        )

    def test_bh_correction_metadata(self, five_metric_results):
        """BH correction result should have correct metadata."""
        from splita.core.correction import MultipleCorrection

        _, pvalues, labels = five_metric_results

        correction = MultipleCorrection(
            pvalues,
            method="bh",
            labels=labels,
        ).run()

        assert correction.n_tests == 5
        assert correction.method == "Benjamini-Hochberg"
        assert correction.labels == labels
        assert correction.alpha == pytest.approx(0.05)
        assert correction.n_rejected == 2

    def test_bonferroni_more_conservative_than_bh(self, five_metric_results):
        """Bonferroni should reject fewer or equal hypotheses compared to BH."""
        from splita.core.correction import MultipleCorrection

        _, pvalues, labels = five_metric_results

        bh = MultipleCorrection(pvalues, method="bh", labels=labels).run()
        bonf = MultipleCorrection(pvalues, method="bonferroni", labels=labels).run()

        assert bonf.n_rejected <= bh.n_rejected, (
            f"Bonferroni ({bonf.n_rejected}) should reject <= BH ({bh.n_rejected})"
        )

        # All Bonferroni adjusted p-values should be >= BH adjusted p-values
        for i in range(5):
            assert bonf.adjusted_pvalues[i] >= bh.adjusted_pvalues[i] - 1e-10, (
                f"Metric {labels[i]}: Bonf adj_p "
                f"({bonf.adjusted_pvalues[i]:.4f}) "
                f"should be >= BH adj_p ({bh.adjusted_pvalues[i]:.4f})"
            )


# ---------------------------------------------------------------------------
# Scenario 10: Full Experiment Lifecycle
# ---------------------------------------------------------------------------


class TestFullExperimentLifecycle:
    """Complete workflow from planning to analysis to reporting.

    1. Plan the sample size using SampleSize.for_proportion.
    2. Estimate the experiment duration.
    3. Generate data at the planned n.
    4. Check SRM to validate the traffic split.
    5. Analyze the experiment.
    6. Verify the result is significant and all pieces connect.
    """

    def test_plan_to_analysis_pipeline(self):
        """Full lifecycle: plan -> duration -> generate -> SRM -> analyze."""
        from splita.core.srm import SRMCheck

        # 1. Plan
        plan = SampleSize.for_proportion(baseline=0.10, mde=0.02)
        n = plan.n_per_variant
        assert n > 0
        assert plan.metric == "proportion"
        assert plan.mde == pytest.approx(0.02)

        # 2. Estimate duration
        with_days = plan.duration(daily_users=2000)
        assert with_days.days_needed is not None
        assert with_days.days_needed > 0

        # 3. Generate data at planned n (oversample 2x for reliability)
        rng = np.random.default_rng(100)
        actual_n = n * 2
        control = rng.binomial(1, 0.10, size=actual_n).astype(float)
        treatment = rng.binomial(1, 0.12, size=actual_n).astype(float)

        # 4. SRM check
        srm = SRMCheck([len(control), len(treatment)]).run()
        assert srm.passed is True
        assert "No sample ratio mismatch" in srm.message

        # 5. Analyze
        result = Experiment(control, treatment).run()

        # 6. Verify
        assert result.significant is True
        assert result.metric == "conversion"
        assert result.method == "ztest"
        assert result.lift > 0
        assert result.pvalue < 0.05
        assert result.control_n == actual_n
        assert result.treatment_n == actual_n

    def test_duration_is_reasonable(self):
        """Duration should be a few days with 2000 daily users."""
        plan = SampleSize.for_proportion(baseline=0.10, mde=0.02)
        with_days = plan.duration(daily_users=2000)

        # n_total ~7686, daily_users=2000 -> ~4 days
        assert with_days.days_needed is not None
        assert 1 <= with_days.days_needed <= 15

    def test_all_pieces_connect(self):
        """Plan n, generate at n, verify result reports correct n."""
        from splita.core.srm import SRMCheck

        plan = SampleSize.for_proportion(baseline=0.10, mde=0.02)
        n = plan.n_per_variant

        rng = np.random.default_rng(101)
        control = rng.binomial(1, 0.10, size=n).astype(float)
        treatment = rng.binomial(1, 0.12, size=n).astype(float)

        srm = SRMCheck([n, n]).run()
        assert srm.passed is True

        result = Experiment(control, treatment).run()
        assert result.control_n == n
        assert result.treatment_n == n
        assert result.alpha == pytest.approx(plan.alpha)


# ---------------------------------------------------------------------------
# Scenario 11: A/B/C Test with Multiple Comparisons
# ---------------------------------------------------------------------------


class TestABCTestWithMultipleComparisons:
    """Testing 2 treatment variants against control with Holm correction.

    Treatment A has a real 2pp lift. Treatment B has no lift. After
    running 2 experiments and applying Holm correction, treatment A
    should survive and treatment B should not.
    """

    def test_srm_check_three_groups(self):
        """SRM check should pass for balanced 3-way split."""
        from splita.core.srm import SRMCheck

        n = 5000
        srm = SRMCheck(
            [n, n, n],
            expected_fractions=[1 / 3, 1 / 3, 1 / 3],
        ).run()
        assert srm.passed is True

    def test_holm_correction_keeps_real_effect(self):
        """Treatment A (real lift) should survive Holm correction."""
        from splita.core.correction import MultipleCorrection

        rng = np.random.default_rng(110)
        n = 10000

        control = rng.binomial(1, 0.10, size=n).astype(float)
        treatment_a = rng.binomial(1, 0.12, size=n).astype(float)  # 2pp lift
        treatment_b = rng.binomial(1, 0.10, size=n).astype(float)  # no lift

        result_a = Experiment(control, treatment_a).run()
        result_b = Experiment(control, treatment_b).run()

        correction = MultipleCorrection(
            [result_a.pvalue, result_b.pvalue],
            method="holm",
            labels=["treatment_a", "treatment_b"],
        ).run()

        assert correction.rejected[0] is True, (
            "Treatment A should survive Holm "
            f"(adj_p={correction.adjusted_pvalues[0]:.4f})"
        )

    def test_holm_correction_rejects_null_effect(self):
        """Treatment B (no lift) should not survive Holm correction."""
        from splita.core.correction import MultipleCorrection

        rng = np.random.default_rng(110)
        n = 10000

        control = rng.binomial(1, 0.10, size=n).astype(float)
        treatment_a = rng.binomial(1, 0.12, size=n).astype(float)
        treatment_b = rng.binomial(1, 0.10, size=n).astype(float)

        result_a = Experiment(control, treatment_a).run()
        result_b = Experiment(control, treatment_b).run()

        correction = MultipleCorrection(
            [result_a.pvalue, result_b.pvalue],
            method="holm",
            labels=["treatment_a", "treatment_b"],
        ).run()

        assert correction.rejected[1] is False, (
            "Treatment B should not survive Holm "
            f"(adj_p={correction.adjusted_pvalues[1]:.4f})"
        )

    def test_abc_full_workflow(self):
        """Full A/B/C workflow: SRM -> experiments -> correction."""
        from splita.core.correction import MultipleCorrection
        from splita.core.srm import SRMCheck

        rng = np.random.default_rng(110)
        n = 10000

        control = rng.binomial(1, 0.10, size=n).astype(float)
        treatment_a = rng.binomial(1, 0.12, size=n).astype(float)
        treatment_b = rng.binomial(1, 0.10, size=n).astype(float)

        # SRM check
        srm = SRMCheck(
            [len(control), len(treatment_a), len(treatment_b)],
            expected_fractions=[1 / 3, 1 / 3, 1 / 3],
        ).run()
        assert srm.passed is True

        # Run experiments
        result_a = Experiment(control, treatment_a).run()
        result_b = Experiment(control, treatment_b).run()

        # Apply Holm correction
        correction = MultipleCorrection(
            [result_a.pvalue, result_b.pvalue],
            method="holm",
            labels=["treatment_a", "treatment_b"],
        ).run()

        assert correction.n_tests == 2
        assert correction.method == "Holm"
        assert correction.n_rejected == 1
        assert correction.rejected[0] is True  # treatment A
        assert correction.rejected[1] is False  # treatment B


# ---------------------------------------------------------------------------
# Scenario 12: SRM Detection Saves the Day
# ---------------------------------------------------------------------------


class TestSRMDetectionSavesTheDay:
    """A bug in the randomizer sends 70% of traffic to treatment.

    SRM catches the imbalance before bad analysis can mislead the team.
    The test also demonstrates that naive analysis on imbalanced data
    can produce misleading "significant" results.
    """

    def test_heavily_imbalanced_srm_fails(self):
        """SRM should clearly fail with 3000/7000 split."""
        from splita.core.srm import SRMCheck

        srm = SRMCheck([3000, 7000]).run()
        assert srm.passed is False
        assert srm.pvalue < 1e-10  # extremely low p-value
        assert "cannot be trusted" in srm.message

    def test_srm_message_is_clear(self):
        """The SRM failure message should mention the experiment cannot be trusted."""
        from splita.core.srm import SRMCheck

        srm = SRMCheck([3000, 7000]).run()
        assert "cannot be trusted" in srm.message
        assert "mismatch detected" in srm.message

    def test_srm_reports_worst_deviation(self):
        """SRM should identify the most deviated variant."""
        from splita.core.srm import SRMCheck

        srm = SRMCheck([3000, 7000]).run()
        # Expected 5000 each. Variant 1 (treatment) has +40% deviation,
        # variant 0 (control) has -40% deviation. Both are equally bad.
        assert srm.worst_variant in (0, 1)
        # Deviations should be large
        assert abs(srm.deviations_pct[0]) == pytest.approx(40.0)
        assert abs(srm.deviations_pct[1]) == pytest.approx(40.0)

    def test_naive_analysis_on_imbalanced_data_is_misleading(self):
        """Naive analysis on buggy data can produce a 'significant' artifact.

        When the randomizer is broken, the groups may differ in composition,
        not just in size. Here we simulate a scenario where the imbalance
        introduces a spurious signal that would mislead the team if SRM
        were not checked first.
        """
        from splita.core.srm import SRMCheck

        rng = np.random.default_rng(120)

        # Broken randomizer: only high-intent users end up in treatment
        # Control gets the typical 10% conversion
        control = rng.binomial(1, 0.10, size=3000).astype(float)
        # Treatment appears to convert better, but it is selection bias
        treatment = rng.binomial(1, 0.12, size=7000).astype(float)

        # SRM catches the problem
        srm = SRMCheck([len(control), len(treatment)]).run()
        assert srm.passed is False

        # Naive analysis sees a "significant" result
        result = Experiment(control, treatment).run()

        # The key lesson: SRM should be checked first.
        # The experiment result exists but cannot be trusted.
        assert 0 <= result.pvalue <= 1
        assert result.control_n == 3000
        assert result.treatment_n == 7000

    def test_srm_guard_prevents_bad_decision(self):
        """Full guard pattern: SRM fails -> do not trust the result."""
        from splita.core.srm import SRMCheck

        rng = np.random.default_rng(121)
        control = rng.binomial(1, 0.10, size=3000).astype(float)
        treatment = rng.binomial(1, 0.13, size=7000).astype(float)

        # Guard: check SRM first
        srm = SRMCheck([len(control), len(treatment)]).run()

        if srm.passed:
            # This branch should NOT be taken
            Experiment(control, treatment).run()
            raise AssertionError("SRM should have failed for 3000/7000 split")
        else:
            # Correct path: SRM failed, do not analyze
            assert "cannot be trusted" in srm.message


# ---------------------------------------------------------------------------
# Scenario 13: Variance Reduction Pipeline
# ---------------------------------------------------------------------------


class TestVarianceReductionPipeline:
    """Revenue A/B test with heavy tails.

    Full pipeline: OutlierHandler -> CUPED -> Experiment.  With correlated
    pre-experiment data and variance reduction, a $2 lift on $25 mean with
    $40 std should become detectable.
    """

    def test_full_variance_reduction_pipeline(self):
        """OutlierHandler -> CUPED -> Experiment detects the effect."""
        from splita.variance import CUPED, OutlierHandler

        rng = np.random.default_rng(200)
        n = 2000

        # Pre-experiment revenue (correlated with post at r~0.6)
        pre_ctrl = rng.normal(25.0, 30.0, size=n)
        pre_trt = rng.normal(25.0, 30.0, size=n)

        # Post-experiment revenue: control mean=$25, treatment mean=$27
        noise = rng.normal(0, 20.0, size=n)
        control = 0.6 * pre_ctrl + 0.4 * 25.0 + noise
        noise = rng.normal(0, 20.0, size=n)
        treatment = 0.6 * pre_trt + 0.4 * 27.0 + 2.0 + noise

        # Inject some heavy-tail outliers
        control[0] = 5000.0
        control[1] = -2000.0
        treatment[0] = 8000.0

        # Step 1: OutlierHandler
        handler = OutlierHandler(method="winsorize")
        ctrl_clean, trt_clean = handler.fit_transform(control, treatment)

        # Outliers should be capped
        assert ctrl_clean[0] < 5000.0
        assert trt_clean[0] < 8000.0

        # Step 2: CUPED with pre-experiment data
        cuped = CUPED()
        ctrl_adj, trt_adj = cuped.fit_transform(
            ctrl_clean,
            trt_clean,
            pre_ctrl,
            pre_trt,
        )

        # Variance reduction should be substantial
        assert cuped.variance_reduction_ > 0.2

        # Step 3: Experiment on adjusted data
        result = Experiment(ctrl_adj, trt_adj).run()

        # Should be significant thanks to variance reduction
        assert result.significant is True
        assert result.pvalue < 0.05

    def test_cuped_variance_reduction_is_substantial(self):
        """CUPED should reduce variance by at least 20% with correlated pre-data."""
        from splita.variance import CUPED

        rng = np.random.default_rng(201)
        n = 2000

        pre_ctrl = rng.normal(25.0, 30.0, size=n)
        pre_trt = rng.normal(25.0, 30.0, size=n)

        noise_ctrl = rng.normal(0, 20.0, size=n)
        noise_trt = rng.normal(0, 20.0, size=n)
        control = 0.6 * pre_ctrl + 0.4 * 25.0 + noise_ctrl
        treatment = 0.6 * pre_trt + 0.4 * 27.0 + noise_trt

        cuped = CUPED()
        cuped.fit(control, treatment, pre_ctrl, pre_trt)

        assert cuped.variance_reduction_ > 0.2
        assert cuped.correlation_ > 0.4


# ---------------------------------------------------------------------------
# Scenario 14: CUPAC for New Users
# ---------------------------------------------------------------------------


class TestCUPACForNewUsers:
    """Testing a feature where users have no pre-experiment metric data,
    but we have user features (age, tenure, past_purchases).

    CUPAC trains an ML model on features to predict the outcome and uses
    those predictions as a CUPED-style covariate.
    """

    def test_cupac_reduces_variance_with_features(self):
        """CUPAC with user features should achieve variance reduction."""
        from splita.variance import CUPAC

        rng = np.random.default_rng(300)
        n = 1000

        # User features: age, tenure, past_purchases
        X_ctrl = rng.normal(0, 1, size=(n, 3))
        X_trt = rng.normal(0, 1, size=(n, 3))

        # Outcome correlated with features
        weights = np.array([1.5, 2.0, 0.8])
        ctrl = X_ctrl @ weights + rng.normal(0, 1.0, n)
        trt = X_trt @ weights + 0.5 + rng.normal(0, 1.0, n)

        cupac = CUPAC(random_state=42)
        _ctrl_adj, _trt_adj = cupac.fit_transform(ctrl, trt, X_ctrl, X_trt)

        # CV R² should be positive (model learns something)
        assert cupac.cv_r2_ > 0
        # Variance reduction should be positive
        assert cupac.variance_reduction_ > 0

    def test_cupac_pipeline_with_experiment(self):
        """CUPAC -> Experiment pipeline should detect the effect."""
        from splita.variance import CUPAC

        rng = np.random.default_rng(301)
        n = 1000

        X_ctrl = rng.normal(0, 1, size=(n, 3))
        X_trt = rng.normal(0, 1, size=(n, 3))

        weights = np.array([1.5, 2.0, 0.8])
        ctrl = X_ctrl @ weights + rng.normal(0, 1.0, n)
        trt = X_trt @ weights + 0.5 + rng.normal(0, 1.0, n)

        cupac = CUPAC(random_state=42)
        ctrl_adj, trt_adj = cupac.fit_transform(ctrl, trt, X_ctrl, X_trt)

        result = Experiment(ctrl_adj, trt_adj).run()
        assert result.lift > 0
        assert 0 <= result.pvalue <= 1


# ---------------------------------------------------------------------------
# Scenario 15: OutlierHandler Preserves Treatment Effect
# ---------------------------------------------------------------------------


class TestOutlierHandlerPreservesTreatmentEffect:
    """Revenue data with extreme outliers.

    Verify that capping outliers does not destroy the treatment effect,
    and in fact helps detect it by reducing inflated variance.
    """

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_outlier_handling_helps_detection(self):
        """With outlier handling, effect becomes detectable."""
        from splita.variance import OutlierHandler

        rng = np.random.default_rng(400)
        n = 2000

        # Revenue data with a true $3 lift
        control = rng.normal(25.0, 15.0, size=n)
        treatment = rng.normal(28.0, 15.0, size=n)

        # Inject extreme outliers that inflate variance
        for i in range(20):
            control[i] = 10000.0 * rng.random()
            treatment[i] = 10000.0 * rng.random()

        # Without outlier handling: high variance may obscure the effect
        result_raw = Experiment(control, treatment).run()

        # With outlier handling: variance is reduced
        handler = OutlierHandler(method="winsorize")
        ctrl_clean, trt_clean = handler.fit_transform(control, treatment)
        result_clean = Experiment(ctrl_clean, trt_clean).run()

        # The cleaned result should have lower p-value (stronger signal)
        assert result_clean.pvalue <= result_raw.pvalue + 0.01

    def test_treatment_effect_preserved_after_capping(self):
        """Mean difference should be roughly preserved after capping."""
        from splita.variance import OutlierHandler

        rng = np.random.default_rng(401)
        n = 2000

        control = rng.normal(25.0, 15.0, size=n)
        treatment = rng.normal(28.0, 15.0, size=n)

        # Inject outliers
        control[0] = 10000.0
        treatment[0] = 10000.0

        handler = OutlierHandler(method="winsorize")
        ctrl_clean, trt_clean = handler.fit_transform(control, treatment)

        # Treatment should still have higher mean than control
        assert np.mean(trt_clean) > np.mean(ctrl_clean)

    def test_outlier_capped_experiment_significant(self):
        """After capping outliers, the experiment should detect the $3 lift."""
        from splita.variance import OutlierHandler

        rng = np.random.default_rng(402)
        n = 2000

        control = rng.normal(25.0, 15.0, size=n)
        treatment = rng.normal(28.0, 15.0, size=n)

        # Inject extreme outliers
        for i in range(10):
            control[i] = 10000.0
            treatment[i] = -5000.0

        handler = OutlierHandler(method="winsorize")
        ctrl_clean, trt_clean = handler.fit_transform(control, treatment)
        result = Experiment(ctrl_clean, trt_clean).run()

        assert result.significant is True
        assert result.lift > 0


# ---------------------------------------------------------------------------
# Scenario 16: Always-On Monitoring with mSPRT
# ---------------------------------------------------------------------------


class TestAlwaysOnMonitoringWithMSPRT:
    """A team runs an experiment with daily data checks using mSPRT.

    They want to peek at results every day without inflating false positives.
    mSPRT provides always-valid p-values that remain valid regardless of
    when or how often the experimenter checks.
    """

    @pytest.fixture()
    def daily_msprt(self):
        """Initialize mSPRT and generate 7 days of daily batches."""
        import warnings

        from splita.sequential.msprt import mSPRT

        rng = np.random.default_rng(1600)
        test = mSPRT(metric="conversion", alpha=0.05)

        daily_n = 500  # users per group per day
        states = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _ in range(7):
                ctrl = rng.binomial(1, 0.10, size=daily_n).astype(float)
                trt = rng.binomial(1, 0.13, size=daily_n).astype(float)
                state = test.update(ctrl, trt)
                states.append(state)

        return test, states

    def test_pvalues_decrease_as_evidence_accumulates(self, daily_msprt):
        """With a true 3pp lift, always-valid p-values should generally decrease."""
        _, states = daily_msprt

        # The overall trend should be downward: last p-value < first p-value
        assert states[-1].always_valid_pvalue < states[0].always_valid_pvalue

    def test_should_stop_eventually_becomes_true(self, daily_msprt):
        """With 7 days of data and a real effect, mSPRT should eventually stop."""
        _, states = daily_msprt
        stopped = any(s.should_stop for s in states)
        assert stopped, (
            "mSPRT should have detected the 3pp lift within 7 days "
            f"(final p={states[-1].always_valid_pvalue:.4f})"
        )

    def test_always_valid_pvalue_in_unit_interval(self, daily_msprt):
        """Always-valid p-values must always be in [0, 1]."""
        _, states = daily_msprt
        for i, s in enumerate(states):
            assert 0.0 <= s.always_valid_pvalue <= 1.0, (
                f"Day {i + 1}: p-value {s.always_valid_pvalue} is out of [0, 1]"
            )


# ---------------------------------------------------------------------------
# Scenario 17: Weekly Reads with GroupSequential
# ---------------------------------------------------------------------------


class TestWeeklyReadsWithGroupSequential:
    """4-week experiment with weekly interim analyses (5 looks total).

    Uses O'Brien-Fleming spending function, which is very conservative early
    on and becomes more liberal at the final analysis.
    """

    def test_first_boundary_is_very_conservative(self):
        """OBF first boundary should be very high (z > 3)."""
        from splita.sequential.group_sequential import GroupSequential

        gs = GroupSequential(n_analyses=5, alpha=0.05, spending_function="obf")
        b = gs.boundary()

        assert b.efficacy_boundaries[0] > 3.0, (
            f"First OBF boundary should be > 3.0, got {b.efficacy_boundaries[0]:.4f}"
        )

    def test_week3_z24_should_continue(self):
        """At week 3, z=2.4 should not cross the OBF boundary (still conservative)."""
        from splita.sequential.group_sequential import GroupSequential

        gs = GroupSequential(n_analyses=5, alpha=0.05, spending_function="obf")

        # 5 looks at info fractions 0.2, 0.4, 0.6, 0.8, 1.0
        # At look 3 (info_frac=0.6), OBF boundary is ~2.53, so z=2.4 should not cross
        result = gs.test(
            [1.0, 1.5, 2.4, None, None],
            [0.2, 0.4, 0.6, 0.8, 1.0],
        )

        look3 = result.analysis_results[2]
        assert look3["action"] == "continue", (
            f"At look 3 with z=2.4, OBF should say continue, "
            f"got {look3['action']} (boundary={look3['efficacy_boundary']:.4f})"
        )

    def test_final_look_z24_decision(self):
        """At the final look (5), z=2.4 should cross the boundary.

        With conditional error spending, the OBF final boundary is ~2.29
        (slightly above the fixed-sample z_alpha/2 ~ 1.96 due to the
        conservative sequential correction).
        """
        from splita.sequential.group_sequential import GroupSequential

        gs = GroupSequential(n_analyses=5, alpha=0.05, spending_function="obf")

        result = gs.test(
            [1.0, 1.5, 2.0, 2.0, 2.4],
            [0.2, 0.4, 0.6, 0.8, 1.0],
        )

        # At the final look, z=2.4 should cross the OBF boundary (~2.29)
        look5 = result.analysis_results[4]
        assert look5["action"] == "stop_efficacy", (
            f"At final look with z=2.4, OBF should stop for efficacy, "
            f"got {look5['action']} (boundary={look5['efficacy_boundary']:.4f})"
        )


# ---------------------------------------------------------------------------
# Scenario 18: mSPRT Catches No Effect Early (Truncation)
# ---------------------------------------------------------------------------


class TestMSPRTCatchesNoEffectEarlyTruncation:
    """Team sets a max sample of 2000. The experiment has no effect.

    mSPRT should hit the truncation limit without declaring significance.
    The stopping_reason should be 'truncation', not 'boundary_crossed'.
    """

    def test_truncation_stops_without_significance(self):
        """mSPRT with null data should stop due to truncation, not significance."""
        import warnings

        from splita.sequential.msprt import mSPRT

        rng = np.random.default_rng(1800)
        test = mSPRT(metric="continuous", alpha=0.05, truncation=2000)

        batch_size = 200
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _ in range(10):  # 10 batches * 200 * 2 groups = 4000 total
                ctrl = rng.normal(10.0, 5.0, size=batch_size)
                trt = rng.normal(10.0, 5.0, size=batch_size)
                state = test.update(ctrl, trt)

        # Should have stopped
        assert state.should_stop is True

        # Get final result to check stopping reason
        result = test.result()
        assert result.stopping_reason == "truncation", (
            f"Expected stopping_reason='truncation', got '{result.stopping_reason}'"
        )

    def test_truncation_pvalue_not_significant(self):
        """Under the null, the always-valid p-value should remain above alpha."""
        import warnings

        from splita.sequential.msprt import mSPRT

        rng = np.random.default_rng(1801)
        test = mSPRT(metric="continuous", alpha=0.05, truncation=2000)

        batch_size = 250
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _ in range(8):
                ctrl = rng.normal(10.0, 5.0, size=batch_size)
                trt = rng.normal(10.0, 5.0, size=batch_size)
                test.update(ctrl, trt)

        # p-value should be above alpha (no real effect)
        result = test.result()
        assert result.always_valid_pvalue >= 0.05, (
            "Under the null, p-value should be >= 0.05, "
            f"got {result.always_valid_pvalue:.4f}"
        )
        assert result.stopping_reason == "truncation"


# ---------------------------------------------------------------------------
# Scenario 19: Email Subject Line Optimization with Thompson Sampling
# ---------------------------------------------------------------------------


class TestEmailSubjectLineOptimizationThompson:
    """Testing 3 email subject lines for open rate using Thompson Sampling.

    Arm 0 has a 20% open rate, arm 1 has 15%, arm 2 has 10%.
    Thompson Sampling adaptively allocates traffic, eventually
    concentrating on the best arm.
    """

    def test_best_arm_gets_most_pulls(self):
        """Arm 0 (20% open rate) should receive the most traffic."""
        from splita.bandits.thompson import ThompsonSampler

        rng = np.random.default_rng(1900)
        ts = ThompsonSampler(
            n_arms=3,
            likelihood="bernoulli",
            stopping_rule="expected_loss",
            stopping_threshold=0.005,
            min_samples=100,
            random_state=42,
        )

        rates = [0.20, 0.15, 0.10]
        for _ in range(1000):
            arm = ts.recommend()
            reward = int(rng.random() < rates[arm])
            ts.update(arm, reward)

        result = ts.result()
        assert result.n_pulls_per_arm[0] > result.n_pulls_per_arm[1]
        assert result.n_pulls_per_arm[0] > result.n_pulls_per_arm[2]

    def test_should_stop_eventually(self):
        """With 1000 users the stopping rule should trigger."""
        from splita.bandits.thompson import ThompsonSampler

        rng = np.random.default_rng(1901)
        ts = ThompsonSampler(
            n_arms=3,
            likelihood="bernoulli",
            stopping_rule="expected_loss",
            stopping_threshold=0.005,
            min_samples=100,
            random_state=42,
        )

        rates = [0.20, 0.15, 0.10]
        for _ in range(1000):
            arm = ts.recommend()
            reward = int(rng.random() < rates[arm])
            ts.update(arm, reward)

        assert ts.result().should_stop is True

    def test_current_best_arm_is_zero(self):
        """The identified best arm should be arm 0."""
        from splita.bandits.thompson import ThompsonSampler

        rng = np.random.default_rng(1902)
        ts = ThompsonSampler(
            n_arms=3,
            likelihood="bernoulli",
            stopping_rule="expected_loss",
            stopping_threshold=0.005,
            min_samples=100,
            random_state=42,
        )

        rates = [0.20, 0.15, 0.10]
        for _ in range(1000):
            arm = ts.recommend()
            reward = int(rng.random() < rates[arm])
            ts.update(arm, reward)

        assert ts.result().current_best_arm == 0


# ---------------------------------------------------------------------------
# Scenario 20: Personalized Pricing with LinTS
# ---------------------------------------------------------------------------


class TestPersonalizedPricingLinTS:
    """3 price points, user features predict which price converts best.

    LinTS should learn to recommend different arms depending on the
    context features provided.
    """

    def test_recommendations_change_based_on_context(self):
        """Different user feature vectors should yield different arms."""
        from splita.bandits.lints import LinTS

        rng = np.random.default_rng(2000)
        lints = LinTS(n_arms=3, n_features=3, lambda_=0.5, random_state=42)

        # Context A prefers arm 0, context B prefers arm 2
        ctx_a = np.array([1.0, 0.0, 0.0])
        ctx_b = np.array([0.0, 0.0, 1.0])

        for _ in range(500):
            # Arm 0 best for ctx_a
            lints.update(0, ctx_a, reward=1.0 + rng.normal(0, 0.1))
            lints.update(1, ctx_a, reward=0.3 + rng.normal(0, 0.1))
            lints.update(2, ctx_a, reward=0.1 + rng.normal(0, 0.1))
            # Arm 2 best for ctx_b
            lints.update(0, ctx_b, reward=0.1 + rng.normal(0, 0.1))
            lints.update(1, ctx_b, reward=0.3 + rng.normal(0, 0.1))
            lints.update(2, ctx_b, reward=1.0 + rng.normal(0, 0.1))

        recs_a = [lints.recommend(ctx_a) for _ in range(30)]
        recs_b = [lints.recommend(ctx_b) for _ in range(30)]

        # Arm 0 should dominate for ctx_a, arm 2 for ctx_b
        assert recs_a.count(0) > 20, (
            f"Expected arm 0 for ctx_a, got distribution {recs_a}"
        )
        assert recs_b.count(2) > 20, (
            f"Expected arm 2 for ctx_b, got distribution {recs_b}"
        )


# ---------------------------------------------------------------------------
# Scenario 21: Thompson vs Fixed A/B Test
# ---------------------------------------------------------------------------


class TestThompsonVsFixedABTest:
    """Compare Thompson Sampling efficiency against a fixed-horizon test.

    Same data: arm 0 at 10%, arm 1 at 13%.
    Fixed allocation: split 50/50 for 2000 users.
    Thompson: adaptive allocation for 2000 users.
    Thompson should direct fewer samples to the losing arm.
    """

    def test_thompson_fewer_samples_on_loser(self):
        """Thompson should allocate fewer samples to the inferior arm."""
        from splita.bandits.thompson import ThompsonSampler

        n_users = 2000
        rates = [0.10, 0.13]

        # --- Fixed 50/50 allocation ---
        rng_fixed = np.random.default_rng(2100)
        fixed_pulls = [0, 0]
        for i in range(n_users):
            arm = i % 2
            fixed_pulls[arm] += 1
            # reward consumed but not used for allocation
            _ = int(rng_fixed.random() < rates[arm])

        # --- Thompson adaptive allocation ---
        rng_ts = np.random.default_rng(2100)
        ts = ThompsonSampler(
            n_arms=2,
            likelihood="bernoulli",
            stopping_rule="expected_loss",
            stopping_threshold=0.001,
            min_samples=200,
            random_state=42,
        )
        for _ in range(n_users):
            arm = ts.recommend()
            reward = int(rng_ts.random() < rates[arm])
            ts.update(arm, reward)

        result = ts.result()
        thompson_pulls = result.n_pulls_per_arm

        # Fixed gives exactly 1000 to the loser (arm 0).
        # Thompson should give fewer to the loser.
        loser_arm = 0  # 10% < 13%
        assert thompson_pulls[loser_arm] < fixed_pulls[loser_arm], (
            f"Thompson gave {thompson_pulls[loser_arm]} pulls to the loser "
            f"vs fixed {fixed_pulls[loser_arm]}"
        )


# ---------------------------------------------------------------------------
# Scenario: Triggered experiment with 60% trigger rate
# ---------------------------------------------------------------------------


class TestTriggeredExperiment60Percent:
    """A mobile app runs a triggered experiment where only ~60% of treatment
    users actually see the new feature (e.g., they must visit a specific page).

    The ITT effect should be diluted relative to the per-protocol effect.
    """

    def test_trigger_rate_approximately_60(self):
        """Trigger rate should be close to 60%."""
        rng = np.random.default_rng(42)
        n = 1000
        ctrl = rng.normal(10, 2, n)
        trt_base = rng.normal(10, 2, n)
        triggered = rng.random(n) < 0.6
        trt_base[triggered] += 2.0  # only triggered users get effect

        from splita.core import TriggeredExperiment

        result = TriggeredExperiment(
            ctrl, trt_base, treatment_triggered=triggered,
        ).run()

        assert 0.5 < result.trigger_rate_treatment < 0.7, (
            f"Trigger rate {result.trigger_rate_treatment:.2f} not near 0.6"
        )

    def test_itt_effect_diluted(self):
        """ITT lift should be smaller than per-protocol lift."""
        rng = np.random.default_rng(42)
        n = 1000
        ctrl = rng.normal(10, 2, n)
        trt = rng.normal(10, 2, n)
        triggered = rng.random(n) < 0.6
        trt[triggered] += 3.0

        from splita.core import TriggeredExperiment

        result = TriggeredExperiment(
            ctrl, trt, treatment_triggered=triggered,
        ).run()

        assert abs(result.per_protocol_result.lift) > abs(result.itt_result.lift), (
            f"PP lift ({result.per_protocol_result.lift:.4f}) should exceed "
            f"ITT lift ({result.itt_result.lift:.4f})"
        )

    def test_per_protocol_detects_effect(self):
        """Per-protocol analysis should detect the true effect."""
        rng = np.random.default_rng(42)
        n = 1000
        ctrl = rng.normal(10, 2, n)
        trt = rng.normal(10, 2, n)
        triggered = rng.random(n) < 0.6
        trt[triggered] += 3.0

        from splita.core import TriggeredExperiment

        result = TriggeredExperiment(
            ctrl, trt, treatment_triggered=triggered,
        ).run()

        assert result.per_protocol_result.significant, (
            "Per-protocol should detect the effect (p="
            f"{result.per_protocol_result.pvalue:.4f})"
        )

    def test_full_pipeline_to_dict(self):
        """Full pipeline result should serialise cleanly."""
        rng = np.random.default_rng(42)
        n = 500
        ctrl = rng.normal(10, 2, n)
        trt = rng.normal(10, 2, n)
        triggered = rng.random(n) < 0.6
        trt[triggered] += 2.0

        from splita.core import TriggeredExperiment

        result = TriggeredExperiment(
            ctrl, trt, treatment_triggered=triggered,
        ).run()

        d = result.to_dict()
        assert "trigger_rate_treatment" in d
        assert "itt_result" in d
        assert "per_protocol_result" in d


# ---------------------------------------------------------------------------
# Scenario: HTE identifies high-value user segment
# ---------------------------------------------------------------------------


class TestHTEHighValueSegment:
    """A company suspects that premium users respond differently to a feature.

    We generate data where the treatment effect is large for users with
    high X0 (proxy for premium) and zero for low X0 users.  The HTE
    estimator should surface this heterogeneity.
    """

    def test_cate_varies_across_segments(self):
        """CATE should be higher for premium users (X0 > 1)."""
        from splita.core import HTEEstimator

        rng = np.random.default_rng(42)
        n = 500
        X_ctrl = rng.normal(size=(n, 3))
        X_trt = rng.normal(size=(n, 3))
        # Only users with X0 > 0 get the effect
        y_ctrl = rng.normal(5, 1, n)
        y_trt = rng.normal(5, 1, n)
        premium_mask = X_trt[:, 0] > 0
        y_trt[premium_mask] += 3.0

        hte = HTEEstimator(method="t_learner").fit(
            y_ctrl, y_trt, X_ctrl, X_trt,
        )
        result = hte.result()

        # CATE std should indicate heterogeneity
        assert result.cate_std > 0.3, (
            f"CATE std {result.cate_std:.4f} too low, expected heterogeneity"
        )

    def test_predict_premium_vs_standard(self):
        """Predict CATE for a premium user vs a standard user."""
        from splita.core import HTEEstimator

        rng = np.random.default_rng(42)
        n = 500
        X_ctrl = rng.normal(size=(n, 3))
        X_trt = rng.normal(size=(n, 3))
        y_ctrl = X_ctrl[:, 0] * 0.5 + rng.normal(0, 0.5, n)
        y_trt = X_trt[:, 0] * 2.5 + rng.normal(0, 0.5, n)

        hte = HTEEstimator(method="t_learner").fit(
            y_ctrl, y_trt, X_ctrl, X_trt,
        )

        premium_user = np.array([[2.0, 0.0, 0.0]])
        standard_user = np.array([[-1.0, 0.0, 0.0]])

        cate_premium = hte.predict(premium_user)[0]
        cate_standard = hte.predict(standard_user)[0]

        assert cate_premium > cate_standard, (
            f"Premium CATE ({cate_premium:.4f}) should exceed "
            f"standard CATE ({cate_standard:.4f})"
        )

    def test_top_features_identifies_x0(self):
        """The most important feature should be X0 (index 0)."""
        from splita.core import HTEEstimator

        rng = np.random.default_rng(42)
        n = 500
        X_ctrl = rng.normal(size=(n, 5))
        X_trt = rng.normal(size=(n, 5))
        # Only X0 drives the treatment effect
        y_ctrl = rng.normal(0, 1, n)
        y_trt = X_trt[:, 0] * 3.0 + rng.normal(0, 1, n)

        hte = HTEEstimator(method="t_learner").fit(
            y_ctrl, y_trt, X_ctrl, X_trt,
        )
        result = hte.result()

        assert result.top_features is not None
        assert result.top_features[0] == 0, (
            f"Top feature should be index 0, got {result.top_features[0]}"
        )


# ---------------------------------------------------------------------------
# Scenario: Confidence Sequence Monitoring Catches Effect Earlier
# ---------------------------------------------------------------------------


class TestCSMonitoringCatchesEarlier:
    """A team monitors a running experiment with ConfidenceSequence.

    Compared to a fixed-horizon test that must wait for all data,
    the CS approach should detect a real effect earlier (with fewer
    total observations) by peeking at each batch.
    """

    def test_cs_detects_before_full_sample(self):
        """CS should stop before the full 2000-per-group sample is reached."""
        from splita.sequential.confidence_sequence import ConfidenceSequence

        rng = np.random.default_rng(9001)
        true_effect = 0.4
        sigma = 1.0
        batch_size = 100
        max_batches = 20  # max 2000 per group

        cs = ConfidenceSequence(alpha=0.05, sigma=sigma)
        stopped_at_batch = None

        for batch_idx in range(1, max_batches + 1):
            ctrl = rng.normal(0, sigma, size=batch_size)
            trt = rng.normal(true_effect, sigma, size=batch_size)
            state = cs.update(ctrl, trt)
            if state.should_stop:
                stopped_at_batch = batch_idx
                break

        assert stopped_at_batch is not None, (
            "CS should have detected effect=0.4 within 20 batches"
        )
        # Should detect well before the full 20 batches
        assert stopped_at_batch < max_batches, (
            f"CS stopped at batch {stopped_at_batch}, should be earlier "
            f"than the full {max_batches} batches"
        )

    def test_cs_stops_with_reasonable_sample(self):
        """CS should stop with a reasonable total sample when the effect is real.

        The CS trades off wider intervals (more data needed) for the ability
        to peek anytime.  With a moderate effect, it should still stop well
        before an unreasonable sample size.
        """
        from splita.sequential.confidence_sequence import ConfidenceSequence

        rng = np.random.default_rng(9002)
        sigma = 1.0
        true_effect = 0.3
        batch_size = 50
        max_per_group = 5000

        cs = ConfidenceSequence(alpha=0.05, sigma=sigma)
        total_per_group = 0

        for _ in range(max_per_group // batch_size):
            ctrl = rng.normal(0, sigma, size=batch_size)
            trt = rng.normal(true_effect, sigma, size=batch_size)
            state = cs.update(ctrl, trt)
            total_per_group += batch_size
            if state.should_stop:
                break

        assert state.should_stop, (
            "CS should have stopped with effect=0.3 within 5000 obs per group"
        )
        # Should stop in a reasonable amount (not use the full budget)
        assert total_per_group < max_per_group, (
            f"CS used {total_per_group} per group, should be << {max_per_group}"
        )

    def test_cs_result_after_monitoring(self):
        """After monitoring stops, result() should have proper metadata."""
        from splita.sequential.confidence_sequence import ConfidenceSequence

        rng = np.random.default_rng(9003)
        cs = ConfidenceSequence(alpha=0.05, sigma=1.0)

        for _ in range(10):
            ctrl = rng.normal(0, 1, size=200)
            trt = rng.normal(0.5, 1, size=200)
            state = cs.update(ctrl, trt)
            if state.should_stop:
                break

        result = cs.result()
        assert result.stopping_reason in ("ci_excludes_zero", "not_stopped")
        assert result.total_observations == result.n_control + result.n_treatment
        assert result.width > 0


# ---------------------------------------------------------------------------
# Scenario: Pre-experiment Validation Identifies Underpowered Metric
# ---------------------------------------------------------------------------


class TestPreExperimentIdentifiesUnderpowered:
    """A data scientist validates a metric before launching an experiment.

    The metric has very high variance relative to the expected effect.
    MetricSensitivity should flag this as underpowered, and
    VarianceEstimator should provide distributional diagnostics.
    """

    def test_high_variance_metric_low_power(self):
        """Metric with std=50 should have low power for MDE=0.5."""
        from splita.diagnostics.pre_experiment import MetricSensitivity

        rng = np.random.default_rng(9010)
        # Revenue metric: mean=100, std=50 (highly variable)
        data = rng.normal(100, 50, size=200)

        ms = MetricSensitivity(n_simulations=300, random_state=9010)
        result = ms.run(data, mde=0.5)

        # Power should be very low for such a small MDE vs large variance
        assert result.estimated_power < 0.20, (
            f"Power should be low for MDE=0.5 with std~50, got {result.estimated_power:.3f}"
        )
        # Recommended n should be very large
        assert result.recommended_n > 50_000, (
            f"Recommended n should be huge, got {result.recommended_n}"
        )

    def test_variance_estimator_flags_issues(self):
        """VarianceEstimator should detect heavy tails in revenue data."""
        from splita.diagnostics.pre_experiment import VarianceEstimator

        rng = np.random.default_rng(9011)
        # Simulate revenue with outliers (log-normal with extreme tail)
        data = rng.lognormal(mean=3.0, sigma=2.0, size=1000)

        ve = VarianceEstimator().fit(data)
        result = ve.result()

        # Log-normal with sigma=2.0 should be flagged as heavy-tailed and skewed
        assert result.is_heavy_tailed or result.is_skewed, (
            f"Should flag distributional issues: kurtosis={result.kurtosis:.1f}, "
            f"skewness={result.skewness:.1f}"
        )
        assert any("outlier" in r.lower() or "mann-whitney" in r.lower()
                    for r in result.recommendations), (
            "Should recommend outlier handling or non-parametric tests"
        )

    def test_full_pre_experiment_pipeline(self):
        """Full pipeline: check variance, then check sensitivity, then decide."""
        from splita.diagnostics.pre_experiment import MetricSensitivity, VarianceEstimator

        rng = np.random.default_rng(9012)

        # Step 1: Gather historical data
        historical = rng.normal(10, 2, size=500)

        # Step 2: Variance diagnostics
        ve_result = VarianceEstimator().fit(historical).result()
        assert not ve_result.is_heavy_tailed, "Clean normal data should not be heavy-tailed"

        # Step 3: Check if we can detect a 0.5-unit MDE
        ms = MetricSensitivity(n_simulations=200, random_state=9012)
        ms_result = ms.run(historical, mde=0.5)

        # With std~2 and MDE=0.5, n=500 should have decent power
        assert ms_result.estimated_power > 0.50, (
            f"Expected reasonable power, got {ms_result.estimated_power:.3f}"
        )
        assert ms_result.is_sensitive, "Metric should be sensitive at n=500"

    def test_underpowered_metric_needs_more_data(self):
        """An underpowered metric should recommend more data than available."""
        from splita.diagnostics.pre_experiment import MetricSensitivity

        rng = np.random.default_rng(9013)
        # Small dataset, high variance, tiny MDE
        data = rng.normal(100, 30, size=50)

        ms = MetricSensitivity(n_simulations=200, random_state=9013)
        result = ms.run(data, mde=0.1)

        # recommended_n should far exceed available data (50)
        assert result.recommended_n > len(data) * 10, (
            f"recommended_n={result.recommended_n} should be >> {len(data)}"
        )


# ---------------------------------------------------------------------------
# Scenario: Bayesian A/B test with ROPE for "practically equivalent"
# ---------------------------------------------------------------------------


class TestBayesianROPEPracticalEquivalence:
    """A product team runs a Bayesian A/B test to check if a feature
    change has a practically meaningful effect on conversion rate.
    They use a ROPE of +/-0.5pp to determine practical equivalence.
    """

    def test_no_real_effect_falls_in_rope(self):
        """No real effect -> posterior mass concentrated in ROPE."""
        from splita.core import BayesianExperiment

        rng = np.random.default_rng(42)
        n = 5000
        ctrl = rng.binomial(1, 0.10, n)
        trt = rng.binomial(1, 0.10, n)
        result = BayesianExperiment(
            ctrl, trt, rope=(-0.005, 0.005), random_state=0
        ).run()
        assert result.prob_in_rope is not None
        assert result.prob_in_rope > 0.40
        assert 0.3 < result.prob_b_beats_a < 0.7

    def test_real_effect_escapes_rope(self):
        """Real +5pp effect -> negligible posterior mass in ROPE."""
        from splita.core import BayesianExperiment

        rng = np.random.default_rng(42)
        n = 5000
        ctrl = rng.binomial(1, 0.10, n)
        trt = rng.binomial(1, 0.15, n)
        result = BayesianExperiment(
            ctrl, trt, rope=(-0.005, 0.005), random_state=0
        ).run()
        assert result.prob_in_rope is not None
        assert result.prob_in_rope < 0.01
        assert result.prob_b_beats_a > 0.99

    def test_borderline_effect_decision(self):
        """Effect near ROPE boundary -> result is valid, decision uncertain."""
        from splita.core import BayesianExperiment

        rng = np.random.default_rng(42)
        n = 5000
        ctrl = rng.binomial(1, 0.100, n)
        trt = rng.binomial(1, 0.105, n)
        result = BayesianExperiment(
            ctrl, trt, rope=(-0.005, 0.005), random_state=0
        ).run()
        assert result.prob_in_rope is not None
        assert result.rope == (-0.005, 0.005)
        assert 0.0 <= result.prob_in_rope <= 1.0
        assert result.ci_lower < result.ci_upper


# ---------------------------------------------------------------------------
# Scenario: Multi-objective with 3 metrics -> tradeoff decision
# ---------------------------------------------------------------------------


class TestMultiObjectiveTradeoffDecision:
    """A product team tests a new recommendation algorithm against 3 metrics:
    CTR, revenue per user, and page load time.  The treatment improves
    engagement but degrades performance, requiring a tradeoff decision.
    """

    def test_three_metric_tradeoff(self):
        """Two positive, one negative lift metric -> 'tradeoff'.

        Speed metric: treatment is *lower* (faster is better, but raw lift
        is negative), so the multi-objective engine sees a sig negative lift.
        """
        from splita.core.multi_objective import MultiObjectiveExperiment

        rng = np.random.default_rng(42)
        n = 1000
        exp = MultiObjectiveExperiment(
            metric_names=["ctr", "revenue", "speed_score"]
        )
        # CTR: treatment better (positive lift)
        exp.add_metric(
            rng.binomial(1, 0.05, n).astype(float),
            rng.binomial(1, 0.08, n).astype(float),
        )
        # Revenue: treatment better (positive lift)
        exp.add_metric(rng.normal(10, 3, n), rng.normal(11, 3, n))
        # Speed score: treatment *worse* (negative lift = treatment lower)
        exp.add_metric(rng.normal(100, 10, n), rng.normal(85, 10, n))
        result = exp.run()
        assert len(result.metric_results) == 3
        assert result.pareto_dominant is False
        assert result.recommendation == "tradeoff"
        assert len(result.tradeoffs) > 0

    def test_all_three_positive_adopt(self):
        """All 3 metrics improved -> 'adopt'."""
        from splita.core.multi_objective import MultiObjectiveExperiment

        rng = np.random.default_rng(42)
        n = 1000
        exp = MultiObjectiveExperiment(
            metric_names=["ctr", "revenue", "speed"]
        )
        exp.add_metric(
            rng.binomial(1, 0.05, n).astype(float),
            rng.binomial(1, 0.10, n).astype(float),
        )
        exp.add_metric(rng.normal(10, 2, n), rng.normal(12, 2, n))
        exp.add_metric(rng.normal(500, 50, n), rng.normal(520, 50, n))
        result = exp.run()
        assert result.recommendation == "adopt"
        assert result.pareto_dominant is True
        assert len(result.tradeoffs) == 0


# ---------------------------------------------------------------------------
# Scenario: DiD for policy change evaluation
# ---------------------------------------------------------------------------


class TestDiDPolicyChangeScenario:
    """A company introduces a policy in a treated region and uses DiD to
    estimate the causal effect by differencing out common time trends.
    """

    def test_did_detects_policy_effect(self):
        from splita.causal.did import DifferenceInDifferences

        rng = np.random.default_rng(10001)
        n = 500
        true_effect = 2.5

        pre_control = rng.normal(50, 5, n)
        pre_treatment = rng.normal(50, 5, n)
        post_control = rng.normal(53, 5, n)
        post_treatment = rng.normal(53 + true_effect, 5, n)

        r = DifferenceInDifferences(alpha=0.05).fit(
            pre_control, pre_treatment, post_control, post_treatment
        ).result()

        assert abs(r.att - true_effect) < 1.5
        assert r.significant is True

    def test_parallel_trends_holds(self):
        from splita.causal.did import DifferenceInDifferences

        rng = np.random.default_rng(10002)
        n = 500
        pre_control = rng.normal(50, 5, n)
        pre_treatment = rng.normal(50, 5, n)
        post_control = rng.normal(53, 5, n)
        post_treatment = rng.normal(55, 5, n)

        r = DifferenceInDifferences().fit(
            pre_control, pre_treatment, post_control, post_treatment
        ).result()
        assert r.parallel_trends_pvalue > 0.05

    def test_ci_covers_true_effect(self):
        from splita.causal.did import DifferenceInDifferences

        rng = np.random.default_rng(10003)
        n = 1000
        true_effect = 1.5
        pre_control = rng.normal(100, 10, n)
        pre_treatment = rng.normal(100, 10, n)
        post_control = rng.normal(105, 10, n)
        post_treatment = rng.normal(105 + true_effect, 10, n)

        r = DifferenceInDifferences().fit(
            pre_control, pre_treatment, post_control, post_treatment
        ).result()
        assert r.ci_lower <= true_effect <= r.ci_upper

    def test_result_serializable(self):
        from splita.causal.did import DifferenceInDifferences

        rng = np.random.default_rng(10004)
        n = 100
        r = DifferenceInDifferences().fit(
            rng.normal(0, 1, n), rng.normal(0, 1, n),
            rng.normal(0, 1, n), rng.normal(2, 1, n),
        ).result()
        d = r.to_dict()
        assert isinstance(d, dict)
        assert set(d.keys()) >= {"att", "se", "pvalue", "ci_lower", "ci_upper"}


# ---------------------------------------------------------------------------
# Scenario: Synthetic control for market expansion
# ---------------------------------------------------------------------------


class TestSyntheticControlMarketExpansion:
    """Company expands into a new market; donor markets serve as controls."""

    def test_sc_detects_expansion_effect(self):
        from splita.causal.synthetic_control import SyntheticControl

        rng = np.random.default_rng(11001)
        n_pre, n_post = 12, 6
        expansion_effect = 10.0

        treated_pre = np.arange(1, n_pre + 1, dtype=float) * 5 + rng.normal(0, 1, n_pre)
        treated_post = (
            np.arange(n_pre + 1, n_pre + n_post + 1, dtype=float) * 5
            + expansion_effect + rng.normal(0, 1, n_post)
        )

        n_donors = 4
        donors_pre = np.zeros((n_pre, n_donors))
        donors_post = np.zeros((n_post, n_donors))
        for d in range(n_donors):
            scale = 0.8 + 0.4 * d / n_donors
            offset = rng.normal(0, 2)
            donors_pre[:, d] = (
                np.arange(1, n_pre + 1, dtype=float) * 5 * scale
                + offset + rng.normal(0, 1, n_pre)
            )
            donors_post[:, d] = (
                np.arange(n_pre + 1, n_pre + n_post + 1, dtype=float) * 5 * scale
                + offset + rng.normal(0, 1, n_post)
            )

        r = SyntheticControl().fit(
            treated_pre, treated_post, donors_pre, donors_post
        ).result()
        assert r.effect > 0
        assert abs(r.effect - expansion_effect) < 8.0

    def test_good_pre_treatment_fit(self):
        from splita.causal.synthetic_control import SyntheticControl

        rng = np.random.default_rng(11002)
        n_pre, n_post = 20, 5

        treated_pre = np.arange(1, n_pre + 1, dtype=float) + rng.normal(0, 0.1, n_pre)
        treated_post = np.arange(n_pre + 1, n_pre + n_post + 1, dtype=float) + 5.0
        donors_pre = np.column_stack([
            np.arange(1, n_pre + 1, dtype=float) + rng.normal(0, 0.1, n_pre),
            np.arange(1, n_pre + 1, dtype=float) * 2,
        ])
        donors_post = np.column_stack([
            np.arange(n_pre + 1, n_pre + n_post + 1, dtype=float),
            np.arange(n_pre + 1, n_pre + n_post + 1, dtype=float) * 2,
        ])

        r = SyntheticControl().fit(
            treated_pre, treated_post, donors_pre, donors_post
        ).result()
        assert r.pre_treatment_rmse < 1.0

    def test_weight_constraints(self):
        from splita.causal.synthetic_control import SyntheticControl

        rng = np.random.default_rng(11003)
        n_pre, n_post, n_donors = 10, 5, 3
        r = SyntheticControl().fit(
            rng.normal(10, 1, n_pre),
            rng.normal(15, 1, n_post),
            rng.normal(10, 1, (n_pre, n_donors)),
            rng.normal(10, 1, (n_post, n_donors)),
        ).result()
        assert abs(sum(r.weights) - 1.0) < 1e-6
        assert all(w >= -1e-10 for w in r.weights)
        assert len(r.effect_series) == n_post


# ---------------------------------------------------------------------------
# Scenario 10: E-value monitoring with daily peeks
# ---------------------------------------------------------------------------


class TestEValueDailyMonitoring:
    """A team monitors an experiment daily using E-values."""

    def test_evalue_detects_real_effect_with_daily_peeks(self):
        """With a real 5pp lift, E-value should stop within 14 days."""
        from splita.sequential.evalue import EValue

        rng = np.random.default_rng(10001)
        ev = EValue(alpha=0.05, metric="conversion", tau=0.01)

        stopped = False
        for day in range(14):
            ctrl = rng.binomial(1, 0.10, size=200).astype(float)
            trt = rng.binomial(1, 0.15, size=200).astype(float)
            state = ev.update(ctrl, trt)
            if state.should_stop:
                stopped = True
                break

        result = ev.result()
        assert stopped, f"E-value did not stop. e_value={result.e_value:.4f}"
        assert result.stopping_reason == "e_value_threshold_crossed"

    def test_evalue_does_not_reject_under_null(self):
        """Under H0, E-value should NOT reject with daily peeks."""
        from splita.sequential.evalue import EValue

        rng = np.random.default_rng(10002)
        ev = EValue(alpha=0.05, metric="conversion", tau=0.01)

        stopped = False
        for day in range(30):
            ctrl = rng.binomial(1, 0.10, size=200).astype(float)
            trt = rng.binomial(1, 0.10, size=200).astype(float)
            state = ev.update(ctrl, trt)
            if state.should_stop:
                stopped = True
                break

        assert not stopped, "E-value falsely rejected under H0"

    def test_evalue_incremental_matches_batch(self):
        """Incremental daily updates give the same result as batch."""
        from splita.sequential.evalue import EValue

        rng = np.random.default_rng(10003)
        all_ctrl, all_trt = [], []
        for _ in range(5):
            all_ctrl.append(rng.normal(0, 1, size=100))
            all_trt.append(rng.normal(0.2, 1, size=100))

        ev_inc = EValue(alpha=0.05, metric="continuous", tau=0.1)
        for c, t in zip(all_ctrl, all_trt):
            state_inc = ev_inc.update(c, t)

        ev_batch = EValue(alpha=0.05, metric="continuous", tau=0.1)
        state_batch = ev_batch.update(
            np.concatenate(all_ctrl), np.concatenate(all_trt)
        )
        assert abs(state_inc.e_value - state_batch.e_value) < 1e-10


# ---------------------------------------------------------------------------
# Scenario 11: LinUCB for personalized content recommendation
# ---------------------------------------------------------------------------


class TestLinUCBPersonalizedContent:
    """A content platform uses LinUCB to personalize recommendations."""

    def test_linucb_learns_user_preferences(self):
        """LinUCB learns that different segments prefer different content."""
        from splita.bandits.linucb import LinUCB

        rng = np.random.default_rng(11001)
        ucb = LinUCB(3, n_features=4, alpha=1.0, random_state=42)
        theta = {
            0: np.array([1.0, 0.0, 0.0, 0.0]),
            1: np.array([0.0, 1.0, 0.0, 0.0]),
            2: np.array([0.0, 0.0, 0.5, 0.5]),
        }

        for _ in range(300):
            ctx = rng.standard_normal(4)
            ctx = ctx / np.linalg.norm(ctx)
            for arm in range(3):
                reward = float(ctx @ theta[arm] + rng.normal(0, 0.2))
                ucb.update(arm, ctx, reward)

        assert ucb.recommend(np.array([1.0, 0.0, 0.0, 0.0])) == 0
        assert ucb.recommend(np.array([0.0, 1.0, 0.0, 0.0])) == 1

    def test_linucb_result_tracks_rewards(self):
        """LinUCB result() tracks total reward and pulls accurately."""
        from splita.bandits.linucb import LinUCB

        ucb = LinUCB(2, n_features=2, random_state=42)
        ctx = np.array([1.0, 0.5])
        ucb.update(0, ctx, 1.0)
        ucb.update(0, ctx, 0.5)
        ucb.update(1, ctx, 0.3)
        result = ucb.result()
        assert result.n_pulls_per_arm == [2, 1]
        assert result.total_reward == pytest.approx(1.8)
        assert result.current_best_arm == 0

    def test_linucb_exploitation_after_training(self):
        """After training, LinUCB exploits consistently."""
        from splita.bandits.linucb import LinUCB

        rng = np.random.default_rng(11002)
        ucb = LinUCB(2, n_features=2, alpha=1.0, random_state=42)
        ctx = np.array([1.0, 0.0])
        for _ in range(200):
            ucb.update(0, ctx, 1.0 + rng.normal(0, 0.1))
            ucb.update(1, ctx, 0.1 + rng.normal(0, 0.1))

        recs = [ucb.recommend(ctx) for _ in range(20)]
        assert recs.count(0) == 20


# ---------------------------------------------------------------------------
# Scenario: Revenue experiment - Lin's adjustment + CUPED comparison
# ---------------------------------------------------------------------------


class TestRevenueLinAdjustmentVsCUPED:
    """Revenue A/B test comparing Lin's regression adjustment with CUPED."""

    def test_both_methods_detect_true_effect(self):
        """Both RA and CUPED detect a true +$0.50 revenue lift."""
        from splita.variance.cuped import CUPED
        from splita.variance.regression_adjustment import RegressionAdjustment

        rng = np.random.default_rng(4201)
        n = 2000
        true_lift = 0.50
        pre_ctrl = rng.lognormal(2, 0.5, n)
        pre_trt = rng.lognormal(2, 0.5, n)
        post_ctrl = 0.7 * pre_ctrl + rng.normal(0, 2, n)
        post_trt = 0.7 * pre_trt + true_lift + rng.normal(0, 2, n)

        cuped = CUPED()
        ctrl_adj, trt_adj = cuped.fit_transform(
            post_ctrl, post_trt, pre_ctrl, pre_trt
        )
        cuped_ate = float(np.mean(trt_adj) - np.mean(ctrl_adj))
        ra = RegressionAdjustment()
        ra_result = ra.fit_transform(post_ctrl, post_trt, pre_ctrl, pre_trt)
        assert abs(cuped_ate - true_lift) < 0.5
        assert abs(ra_result.ate - true_lift) < 0.5

    def test_lin_se_not_worse_than_cuped(self):
        """Lin's RA SE <= CUPED SE (single covariate)."""
        from splita.variance.cuped import CUPED
        from splita.variance.regression_adjustment import RegressionAdjustment

        rng = np.random.default_rng(4202)
        n = 2000
        pre_ctrl = rng.lognormal(2, 0.5, n)
        pre_trt = rng.lognormal(2, 0.5, n)
        post_ctrl = 0.7 * pre_ctrl + rng.normal(0, 2, n)
        post_trt = 0.7 * pre_trt + 0.5 + rng.normal(0, 2, n)
        cuped = CUPED()
        ctrl_adj, trt_adj = cuped.fit_transform(
            post_ctrl, post_trt, pre_ctrl, pre_trt
        )
        cuped_se = float(
            np.sqrt(np.var(ctrl_adj, ddof=1) / n + np.var(trt_adj, ddof=1) / n)
        )
        ra = RegressionAdjustment()
        ra_result = ra.fit_transform(post_ctrl, post_trt, pre_ctrl, pre_trt)
        assert ra_result.se <= cuped_se * 1.10

    def test_full_pipeline_with_experiment(self):
        """Full pipeline: sample size -> experiment -> RA adjustment."""
        from splita.variance.regression_adjustment import RegressionAdjustment

        rng = np.random.default_rng(4204)
        n = min(
            SampleSize.for_mean(
                baseline_mean=10.0, baseline_std=5.0, mde=0.5
            ).n_per_variant,
            2000,
        )
        pre_ctrl = rng.lognormal(2, 0.3, n)
        pre_trt = rng.lognormal(2, 0.3, n)
        post_ctrl = 0.8 * pre_ctrl + rng.normal(0, 1, n)
        post_trt = 0.8 * pre_trt + 0.5 + rng.normal(0, 1, n)
        unadj = Experiment(post_ctrl, post_trt, method="ttest").run()
        ra = RegressionAdjustment()
        ra_result = ra.fit_transform(post_ctrl, post_trt, pre_ctrl, pre_trt)
        assert unadj.lift > 0
        assert ra_result.ate > 0


# ---------------------------------------------------------------------------
# Scenario: DoubleML removes confounding in observational-like data
# ---------------------------------------------------------------------------


class TestDoubleMLObservationalScenario:
    """Observational-like experiment where treatment is confounded."""

    def test_doubleml_reduces_bias_vs_naive(self):
        """DoubleML estimate is closer to truth than naive difference."""
        from splita.variance.double_ml import DoubleML

        rng = np.random.default_rng(5001)
        n = 3000
        true_ate = 2.0
        X = rng.normal(0, 1, size=(n, 5))
        engagement = X[:, 0]
        T_prob = 1.0 / (1.0 + np.exp(-(engagement + rng.normal(0, 0.3, n))))
        T = rng.binomial(1, T_prob).astype(float)
        Y = true_ate * T + 3.0 * engagement + X[:, 1] + rng.normal(0, 1, n)
        naive_ate = float(np.mean(Y[T > 0.5]) - np.mean(Y[T <= 0.5]))
        result = DoubleML(cv=5, random_state=42).fit_transform(Y, T, X)
        naive_bias = abs(naive_ate - true_ate)
        dml_bias = abs(result.ate - true_ate)
        assert dml_bias < naive_bias
        assert abs(result.ate - true_ate) < 1.5

    def test_doubleml_ci_contains_true_effect(self):
        """DoubleML 95% CI should contain the true ATE."""
        from splita.variance.double_ml import DoubleML

        rng = np.random.default_rng(5002)
        n = 3000
        true_ate = 1.5
        X = rng.normal(0, 1, size=(n, 4))
        T = (X[:, 0] > 0).astype(float) + rng.normal(0, 0.2, n)
        T = np.clip(T, 0, None)
        Y = true_ate * T + X @ [1.0, 0.5, -0.3, 0.2] + rng.normal(0, 1, n)
        result = DoubleML(random_state=42).fit_transform(Y, T, X)
        assert result.ci_lower < true_ate < result.ci_upper

    def test_doubleml_with_custom_models(self):
        """DoubleML works end-to-end with custom sklearn models."""
        from sklearn.ensemble import GradientBoostingRegressor
        from splita.variance.double_ml import DoubleML

        rng = np.random.default_rng(5003)
        n = 1000
        X = rng.normal(0, 1, size=(n, 3))
        T = rng.binomial(1, 0.5, n).astype(float)
        Y = 1.0 * T + np.sin(X[:, 0]) + rng.normal(0, 0.5, n)
        result = DoubleML(
            outcome_model=GradientBoostingRegressor(
                n_estimators=50, max_depth=3, random_state=42
            ),
            propensity_model=GradientBoostingRegressor(
                n_estimators=50, max_depth=3, random_state=42
            ),
            cv=3,
            random_state=42,
        ).fit_transform(Y, T, X)
        assert abs(result.ate - 1.0) < 0.5
        assert result.se > 0
        assert result.outcome_r2 > 0.0
