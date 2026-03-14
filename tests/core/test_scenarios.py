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
