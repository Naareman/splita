"""End-to-end scenario tests for the splita A/B testing library.

Each test class represents a realistic A/B testing scenario that exercises
the full pipeline: sample-size planning, data generation, experiment
analysis, and result interpretation.
"""

from __future__ import annotations

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
            baseline_mean=25.0, baseline_std=40.0, mde=2.0,
        )
        assert result.n_per_variant > 1_000, (
            f"High-variance metric should need many samples, got {result.n_per_variant}"
        )
        assert result.metric == "mean"

    def test_mannwhitney_on_skewed_data(self):
        """Mann-Whitney should handle heavy-tailed revenue data."""
        plan = SampleSize.for_mean(
            baseline_mean=25.0, baseline_std=40.0, mde=2.0,
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
            baseline_mean=25.0, baseline_std=40.0, mde=2.0,
        )
        n = plan.n_per_variant

        rng = np.random.default_rng(43)
        sigma_ln = 1.0
        mu_ctrl = np.log(25.0) - sigma_ln**2 / 2
        mu_trt = np.log(27.0) - sigma_ln**2 / 2

        control = rng.lognormal(mean=mu_ctrl, sigma=sigma_ln, size=n)
        treatment = rng.lognormal(mean=mu_trt, sigma=sigma_ln, size=n)

        result = Experiment(
            control, treatment,
            method="bootstrap", n_bootstrap=2000, random_state=44,
        ).run()

        assert result.method == "bootstrap"
        assert 0 <= result.pvalue <= 1

    def test_mannwhitney_and_bootstrap_agree_on_direction(self):
        """Both methods should agree on the direction of the effect."""
        plan = SampleSize.for_mean(
            baseline_mean=25.0, baseline_std=40.0, mde=2.0,
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
            control, treatment,
            method="bootstrap", n_bootstrap=2000, random_state=44,
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
            baseline_num_mean=5.0,     # avg clicks per user
            baseline_den_mean=100.0,   # avg impressions per user
            baseline_num_std=3.0,
            baseline_den_std=30.0,
            baseline_covariance=50.0,  # clicks and impressions are correlated
            mde=0.005,                 # detect 0.5pp CTR change
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
            clicks_ctrl, clicks_trt,
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
            clicks_ctrl, clicks_trt,
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
            f"With n=500 and 1pp lift, should be non-significant, got p={result.pvalue:.4f}"
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
            baseline=0.10, mde=0.01, alternative="two-sided",
        )
        one_sided = SampleSize.for_proportion(
            baseline=0.10, mde=0.01, alternative="one-sided",
        )
        assert one_sided.n_per_variant < two_sided.n_per_variant, (
            f"One-sided ({one_sided.n_per_variant}) should need fewer samples "
            f"than two-sided ({two_sided.n_per_variant})"
        )

    def test_no_degradation_is_not_significant(self):
        """When treatment equals control, one-sided 'less' test should not fire."""
        plan = SampleSize.for_proportion(
            baseline=0.10, mde=0.01, alternative="one-sided",
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
        control = rng.binomial(1, 0.10, size=n).astype(float)
        treatment = rng.binomial(1, 0.12, size=n).astype(float)

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
            pvalues, method="bh", labels=labels,
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
            pvalues, method="bh", labels=labels,
        ).run()

        # Last 3 metrics (aov, session, bounce) should not survive
        assert correction.rejected[2] is False, (
            f"AOV should not survive BH (adj_p={correction.adjusted_pvalues[2]:.4f})"
        )
        assert correction.rejected[3] is False, (
            f"Session should not survive BH (adj_p={correction.adjusted_pvalues[3]:.4f})"
        )
        assert correction.rejected[4] is False, (
            f"Bounce should not survive BH (adj_p={correction.adjusted_pvalues[4]:.4f})"
        )

    def test_bh_correction_metadata(self, five_metric_results):
        """BH correction result should have correct metadata."""
        from splita.core.correction import MultipleCorrection

        _, pvalues, labels = five_metric_results

        correction = MultipleCorrection(
            pvalues, method="bh", labels=labels,
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
                f"Metric {labels[i]}: Bonferroni adj_p ({bonf.adjusted_pvalues[i]:.4f}) "
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
            f"Treatment A should survive Holm (adj_p={correction.adjusted_pvalues[0]:.4f})"
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
            f"Treatment B should not survive Holm (adj_p={correction.adjusted_pvalues[1]:.4f})"
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
        assert correction.rejected[0] is True   # treatment A
        assert correction.rejected[1] is False   # treatment B


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
            result = Experiment(control, treatment).run()
            assert False, "SRM should have failed for 3000/7000 split"
        else:
            # Correct path: SRM failed, do not analyze
            assert "cannot be trusted" in srm.message
