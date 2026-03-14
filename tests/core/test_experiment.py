"""Tests for splita.core.experiment — Experiment class."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy.stats import mannwhitneyu as sp_mannwhitneyu
from scipy.stats import norm, ttest_ind

from splita._types import ExperimentResult
from splita.core.experiment import Experiment

# ═══════════════════════════════════════════════════════════════════════
# Basic functionality
# ═══════════════════════════════════════════════════════════════════════


class TestBasicFunctionality:
    """Auto-detection and method dispatch."""

    def test_conversion_auto_detects_ztest(self):
        """Binary data → metric='conversion', method='ztest'."""
        rng = np.random.default_rng(42)
        ctrl = rng.binomial(1, 0.10, size=500)
        trt = rng.binomial(1, 0.12, size=500)
        result = Experiment(ctrl, trt).run()
        assert result.metric == "conversion"
        assert result.method == "ztest"

    def test_continuous_auto_detects_ttest(self):
        """Normal data → metric='continuous', method='ttest'."""
        rng = np.random.default_rng(42)
        ctrl = rng.normal(10, 2, size=200)
        trt = rng.normal(10.5, 2, size=200)
        result = Experiment(ctrl, trt).run()
        assert result.metric == "continuous"
        assert result.method == "ttest"

    def test_ratio_auto_detects_delta(self):
        """Denominators provided → metric='ratio', method='delta'."""
        rng = np.random.default_rng(42)
        n = 300
        ctrl_den = rng.poisson(10, size=n).astype(float) + 1
        trt_den = rng.poisson(10, size=n).astype(float) + 1
        ctrl_num = rng.binomial(ctrl_den.astype(int), 0.10).astype(float)
        trt_num = rng.binomial(trt_den.astype(int), 0.12).astype(float)

        result = Experiment(
            ctrl_num,
            trt_num,
            control_denominator=ctrl_den,
            treatment_denominator=trt_den,
        ).run()
        assert result.metric == "ratio"
        assert result.method == "delta"

    def test_explicit_mannwhitney_on_continuous(self):
        """Force mannwhitney on continuous data."""
        rng = np.random.default_rng(42)
        ctrl = rng.normal(10, 2, size=100)
        trt = rng.normal(11, 2, size=100)
        result = Experiment(ctrl, trt, method="mannwhitney").run()
        assert result.method == "mannwhitney"

    def test_explicit_bootstrap_on_conversion(self):
        """Force bootstrap on binary data."""
        rng = np.random.default_rng(42)
        ctrl = rng.binomial(1, 0.10, size=200)
        trt = rng.binomial(1, 0.15, size=200)
        result = Experiment(ctrl, trt, method="bootstrap", random_state=42).run()
        assert result.method == "bootstrap"

    def test_explicit_continuous_metric(self):
        """Passing metric='continuous' explicitly on continuous data."""
        rng = np.random.default_rng(42)
        ctrl = rng.normal(10, 2, size=200)
        trt = rng.normal(10.5, 2, size=200)
        result = Experiment(ctrl, trt, metric="continuous").run()
        assert result.metric == "continuous"
        assert result.method == "ttest"

    def test_explicit_conversion_metric(self):
        """Passing metric='conversion' explicitly on binary data."""
        rng = np.random.default_rng(42)
        ctrl = rng.binomial(1, 0.10, size=500)
        trt = rng.binomial(1, 0.12, size=500)
        result = Experiment(ctrl, trt, metric="conversion").run()
        assert result.metric == "conversion"
        assert result.method == "ztest"

    def test_chisquare_equivalent_to_ztest_two_sided(self):
        """Chi-square p-value ~= ztest p-value for two-sided test."""
        rng = np.random.default_rng(42)
        ctrl = rng.binomial(1, 0.10, size=1000)
        trt = rng.binomial(1, 0.13, size=1000)

        z_result = Experiment(ctrl, trt, method="ztest").run()
        chi_result = Experiment(ctrl, trt, method="chisquare").run()

        assert abs(z_result.pvalue - chi_result.pvalue) < 1e-10


# ═══════════════════════════════════════════════════════════════════════
# Statistical correctness
# ═══════════════════════════════════════════════════════════════════════


class TestStatisticalCorrectness:
    """Verify computed values match scipy / manual calculation."""

    def test_ztest_matches_manual(self):
        """Z-test p-value matches manual calculation with scipy.stats.norm."""
        rng = np.random.default_rng(1)
        ctrl = rng.binomial(1, 0.10, size=500)
        trt = rng.binomial(1, 0.14, size=500)

        result = Experiment(ctrl, trt, method="ztest").run()

        # manual
        p1, p2 = ctrl.mean(), trt.mean()
        n1, n2 = len(ctrl), len(trt)
        p_pool = (ctrl.sum() + trt.sum()) / (n1 + n2)
        se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
        z = (p2 - p1) / se
        pval = float(2 * norm.sf(abs(z)))

        assert abs(result.pvalue - pval) < 1e-12

    def test_ttest_matches_scipy(self):
        """T-test p-value matches scipy.stats.ttest_ind."""
        rng = np.random.default_rng(2)
        ctrl = rng.normal(10, 3, size=150)
        trt = rng.normal(11, 3, size=150)

        result = Experiment(ctrl, trt, method="ttest").run()
        _, expected_p = ttest_ind(trt, ctrl, equal_var=False)

        assert abs(result.pvalue - expected_p) < 1e-12

    def test_mannwhitney_matches_scipy(self):
        """Mann-Whitney p-value matches scipy.stats.mannwhitneyu."""
        rng = np.random.default_rng(3)
        ctrl = rng.normal(10, 2, size=80)
        trt = rng.normal(11, 2, size=80)

        result = Experiment(ctrl, trt, method="mannwhitney").run()
        sp_res = sp_mannwhitneyu(trt, ctrl, alternative="two-sided")

        assert abs(result.pvalue - sp_res.pvalue) < 1e-12

    def test_known_significant_result(self):
        """Large effect size → significant=True."""
        rng = np.random.default_rng(4)
        ctrl = rng.normal(10, 1, size=500)
        trt = rng.normal(12, 1, size=500)  # big difference

        result = Experiment(ctrl, trt).run()
        assert result.significant is True
        assert result.pvalue < 0.001

    def test_known_nonsignificant_result(self):
        """Tiny effect size with small n → significant=False."""
        rng = np.random.default_rng(5)
        ctrl = rng.normal(10, 5, size=20)
        trt = rng.normal(10.1, 5, size=20)  # tiny difference

        result = Experiment(ctrl, trt).run()
        assert result.significant is False

    def test_ci_contains_true_effect(self):
        """CI brackets the true effect for known data."""
        rng = np.random.default_rng(6)
        true_effect = 2.0
        ctrl = rng.normal(10, 1, size=1000)
        trt = rng.normal(10 + true_effect, 1, size=1000)

        result = Experiment(ctrl, trt).run()
        assert result.ci_lower <= true_effect <= result.ci_upper

    def test_bootstrap_reproducibility(self):
        """Same random_state → identical result."""
        rng = np.random.default_rng(7)
        ctrl = rng.normal(10, 2, size=100)
        trt = rng.normal(11, 2, size=100)

        r1 = Experiment(ctrl, trt, method="bootstrap", random_state=99).run()
        r2 = Experiment(ctrl, trt, method="bootstrap", random_state=99).run()

        assert r1.pvalue == r2.pvalue
        assert r1.ci_lower == r2.ci_lower
        assert r1.ci_upper == r2.ci_upper


# ═══════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════


class TestValidation:
    """Input validation raises appropriate errors."""

    def test_too_few_observations(self):
        """< 2 per group raises ValueError."""
        with pytest.raises(ValueError, match="at least 2 elements"):
            Experiment([1], [1, 2, 3])

    def test_invalid_alpha_zero(self):
        """alpha=0 raises ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            Experiment([0, 1, 0], [1, 0, 1], alpha=0.0)

    def test_invalid_alpha_one(self):
        """alpha=1 raises ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            Experiment([0, 1, 0], [1, 0, 1], alpha=1.0)

    def test_invalid_alpha_negative(self):
        """alpha=-0.1 raises ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            Experiment([0, 1, 0], [1, 0, 1], alpha=-0.1)

    def test_invalid_alpha_above_one(self):
        """alpha=1.5 raises ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            Experiment([0, 1, 0], [1, 0, 1], alpha=1.5)

    def test_invalid_metric_string(self):
        """Invalid metric raises ValueError with suggestion."""
        with pytest.raises(ValueError, match="metric"):
            Experiment([0, 1, 0], [1, 0, 1], metric="binary")

    def test_invalid_method_string(self):
        """Invalid method raises ValueError with suggestion."""
        with pytest.raises(ValueError, match="method"):
            Experiment([0, 1, 0], [1, 0, 1], method="anova")

    def test_ratio_without_denominators(self):
        """metric='ratio' without denominators raises ValueError."""
        with pytest.raises(ValueError, match="denominator"):
            Experiment([1, 2, 3], [4, 5, 6], metric="ratio")

    def test_denominator_length_mismatch(self):
        """Denominator length != array length raises ValueError."""
        with pytest.raises(ValueError, match="same length"):
            Experiment(
                [1, 2, 3],
                [4, 5, 6],
                metric="ratio",
                control_denominator=[10, 20],  # wrong length
                treatment_denominator=[10, 20, 30],
            )

    def test_chisquare_with_one_sided(self):
        """Chi-square + alternative='greater' raises ValueError."""
        with pytest.raises(ValueError, match="two-sided"):
            Experiment(
                [0, 1, 0, 1],
                [1, 1, 0, 1],
                method="chisquare",
                alternative="greater",
            ).run()

    def test_nan_in_data(self):
        """NaN values emit warning and are dropped."""
        data_with_nan = [0, 1, float("nan"), 0, 1, 0]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            exp = Experiment(data_with_nan, [0, 1, 0, 1, 0])
            nan_warnings = [x for x in w if "NaN" in str(x.message)]
            assert len(nan_warnings) >= 1
        result = exp.run()
        assert result.control_n == 5  # 6 - 1 NaN


# ═══════════════════════════════════════════════════════════════════════
# Skewness warning
# ═══════════════════════════════════════════════════════════════════════


class TestSkewnessWarning:
    """High skewness emits RuntimeWarning."""

    def test_high_skewness_warning(self):
        """Continuous data with |skew| > 2 emits RuntimeWarning."""
        rng = np.random.default_rng(10)
        # exponential data is right-skewed; scale up for |skew| > 2
        ctrl = rng.exponential(1.0, size=200) ** 3
        trt = rng.exponential(1.0, size=200) ** 3

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Experiment(ctrl, trt)
            skew_warnings = [
                x
                for x in w
                if issubclass(x.category, RuntimeWarning)
                and "skewness" in str(x.message)
            ]
            assert len(skew_warnings) >= 1


# ═══════════════════════════════════════════════════════════════════════
# Properties
# ═══════════════════════════════════════════════════════════════════════


class TestProperties:
    """Result type and field completeness."""

    def test_idempotent(self):
        """Calling .run() twice gives consistent results."""
        rng = np.random.default_rng(20)
        ctrl = rng.normal(10, 2, size=100)
        trt = rng.normal(11, 2, size=100)
        exp = Experiment(ctrl, trt)
        r1 = exp.run()
        r2 = exp.run()
        assert r1.pvalue == r2.pvalue
        assert r1.ci_lower == r2.ci_lower

    def test_result_is_experiment_result(self):
        """Result is an ExperimentResult instance."""
        result = Experiment([0, 1, 0, 1], [1, 1, 0, 1]).run()
        assert isinstance(result, ExperimentResult)

    def test_result_fields_populated(self):
        """All fields are non-None."""
        result = Experiment([0, 1, 0, 1, 0], [1, 1, 0, 1, 1]).run()
        assert result.control_mean is not None
        assert result.treatment_mean is not None
        assert result.lift is not None
        assert result.relative_lift is not None
        assert result.pvalue is not None
        assert result.statistic is not None
        assert result.ci_lower is not None
        assert result.ci_upper is not None
        assert result.significant is not None
        assert result.alpha is not None
        assert result.method is not None
        assert result.metric is not None
        assert result.control_n is not None
        assert result.treatment_n is not None
        assert result.power is not None
        assert result.effect_size is not None

    def test_to_dict_works(self):
        """result.to_dict() returns a plain dict."""
        result = Experiment([0, 1, 0, 1, 0], [1, 1, 0, 1, 1]).run()
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "pvalue" in d
        assert "significant" in d
        assert isinstance(d["significant"], bool)


# ═══════════════════════════════════════════════════════════════════════
# Real-world scenarios
# ═══════════════════════════════════════════════════════════════════════


class TestRealWorldScenarios:
    """Realistic A/B test scenarios."""

    def test_ecommerce_ab_test(self):
        """10% baseline, 12% treatment, n=5000 → significant."""
        rng = np.random.default_rng(100)
        ctrl = rng.binomial(1, 0.10, size=5000)
        trt = rng.binomial(1, 0.12, size=5000)

        result = Experiment(ctrl, trt).run()
        assert result.metric == "conversion"
        assert result.significant is True
        assert result.pvalue < 0.05

    def test_revenue_test(self):
        """Control $25 std=$40, trt $26 std=$40, n=1000."""
        rng = np.random.default_rng(101)
        ctrl = rng.normal(25, 40, size=1000)
        trt = rng.normal(26, 40, size=1000)

        result = Experiment(ctrl, trt).run()
        assert result.metric == "continuous"
        # $1 diff with std=$40 and n=1000 is a very small effect
        # p-value should be > 0.05 most of the time, but allow some leeway
        assert result.lift == pytest.approx(trt.mean() - ctrl.mean(), rel=1e-10)
        # With only $1 diff at std=$40, n=1000, we can't assert significance
        # but we can assert the result is structurally valid
        assert 0 <= result.pvalue <= 1
        assert isinstance(result.significant, bool)
        assert result.method == "ttest"
        assert result.metric == "continuous"

    def test_ctr_ratio_test(self):
        """Clicks/pageviews with delta method."""
        rng = np.random.default_rng(102)
        n = 1000
        ctrl_pvs = rng.poisson(20, size=n).astype(float) + 1
        trt_pvs = rng.poisson(20, size=n).astype(float) + 1
        ctrl_clicks = rng.binomial(ctrl_pvs.astype(int), 0.05).astype(float)
        trt_clicks = rng.binomial(trt_pvs.astype(int), 0.06).astype(float)

        result = Experiment(
            ctrl_clicks,
            trt_clicks,
            control_denominator=ctrl_pvs,
            treatment_denominator=trt_pvs,
        ).run()
        assert result.metric == "ratio"
        assert result.method == "delta"
        assert result.control_n == n
        assert result.treatment_n == n


# ═══════════════════════════════════════════════════════════════════════
# One-sided alternative branches
# ═══════════════════════════════════════════════════════════════════════


class TestOneSidedAlternatives:
    """Cover the alternative='greater' and 'less' branches for all methods."""

    # ── ztest ──

    def test_ztest_alternative_greater(self):
        """Conversion data, alternative='greater', treatment > control."""
        rng = np.random.default_rng(50)
        ctrl = rng.binomial(1, 0.10, size=1000)
        trt = rng.binomial(1, 0.15, size=1000)

        two_sided = Experiment(ctrl, trt, method="ztest", alternative="two-sided").run()
        greater = Experiment(ctrl, trt, method="ztest", alternative="greater").run()

        # One-sided p-value should be smaller when effect is in the correct direction
        assert greater.pvalue <= two_sided.pvalue + 1e-12
        # CI: upper bound is inf for "greater"
        assert greater.ci_upper == float("inf")
        assert math.isfinite(greater.ci_lower)

    def test_ztest_alternative_less(self):
        """Conversion data, alternative='less', treatment < control."""
        rng = np.random.default_rng(51)
        ctrl = rng.binomial(1, 0.15, size=1000)
        trt = rng.binomial(1, 0.10, size=1000)

        result = Experiment(ctrl, trt, method="ztest", alternative="less").run()

        assert result.ci_lower == float("-inf")
        assert math.isfinite(result.ci_upper)
        assert result.pvalue < 0.05

    # ── ttest ──

    def test_ttest_alternative_greater(self):
        """Continuous data, alternative='greater'."""
        rng = np.random.default_rng(52)
        ctrl = rng.normal(10, 2, size=200)
        trt = rng.normal(12, 2, size=200)

        result = Experiment(ctrl, trt, method="ttest", alternative="greater").run()

        assert result.ci_upper == float("inf")
        assert math.isfinite(result.ci_lower)
        assert result.pvalue < 0.05

    def test_ttest_alternative_less(self):
        """Continuous data, alternative='less'."""
        rng = np.random.default_rng(53)
        ctrl = rng.normal(12, 2, size=200)
        trt = rng.normal(10, 2, size=200)

        result = Experiment(ctrl, trt, method="ttest", alternative="less").run()

        assert result.ci_lower == float("-inf")
        assert math.isfinite(result.ci_upper)
        assert result.pvalue < 0.05

    # ── mannwhitney ──

    def test_mannwhitney_alternative_greater(self):
        """Mann-Whitney with alternative='greater'."""
        rng = np.random.default_rng(54)
        ctrl = rng.normal(10, 2, size=100)
        trt = rng.normal(12, 2, size=100)

        result = Experiment(
            ctrl, trt, method="mannwhitney", alternative="greater"
        ).run()

        assert result.ci_upper == float("inf")
        assert math.isfinite(result.ci_lower)
        assert result.pvalue < 0.05

    def test_mannwhitney_alternative_less(self):
        """Mann-Whitney with alternative='less'."""
        rng = np.random.default_rng(55)
        ctrl = rng.normal(12, 2, size=100)
        trt = rng.normal(10, 2, size=100)

        result = Experiment(ctrl, trt, method="mannwhitney", alternative="less").run()

        assert result.ci_lower == float("-inf")
        assert math.isfinite(result.ci_upper)
        assert result.pvalue < 0.05

    # ── delta ──

    def test_delta_alternative_greater(self):
        """Ratio data, alternative='greater'."""
        rng = np.random.default_rng(56)
        n = 500
        ctrl_den = rng.poisson(10, size=n).astype(float) + 1
        trt_den = rng.poisson(10, size=n).astype(float) + 1
        ctrl_num = rng.binomial(ctrl_den.astype(int), 0.05).astype(float)
        trt_num = rng.binomial(trt_den.astype(int), 0.10).astype(float)

        result = Experiment(
            ctrl_num,
            trt_num,
            control_denominator=ctrl_den,
            treatment_denominator=trt_den,
            alternative="greater",
        ).run()

        assert result.ci_upper == float("inf")
        assert math.isfinite(result.ci_lower)
        assert result.method == "delta"

    def test_delta_alternative_less(self):
        """Ratio data, alternative='less'."""
        rng = np.random.default_rng(57)
        n = 500
        ctrl_den = rng.poisson(10, size=n).astype(float) + 1
        trt_den = rng.poisson(10, size=n).astype(float) + 1
        ctrl_num = rng.binomial(ctrl_den.astype(int), 0.10).astype(float)
        trt_num = rng.binomial(trt_den.astype(int), 0.05).astype(float)

        result = Experiment(
            ctrl_num,
            trt_num,
            control_denominator=ctrl_den,
            treatment_denominator=trt_den,
            alternative="less",
        ).run()

        assert result.ci_lower == float("-inf")
        assert math.isfinite(result.ci_upper)
        assert result.method == "delta"

    # ── bootstrap ──

    def test_bootstrap_alternative_greater(self):
        """Bootstrap with alternative='greater'."""
        rng = np.random.default_rng(58)
        ctrl = rng.normal(10, 2, size=200)
        trt = rng.normal(12, 2, size=200)

        result = Experiment(
            ctrl,
            trt,
            method="bootstrap",
            alternative="greater",
            random_state=58,
        ).run()

        assert result.ci_upper == float("inf")
        assert math.isfinite(result.ci_lower)

    def test_bootstrap_alternative_less(self):
        """Bootstrap with alternative='less'."""
        rng = np.random.default_rng(59)
        ctrl = rng.normal(12, 2, size=200)
        trt = rng.normal(10, 2, size=200)

        result = Experiment(
            ctrl,
            trt,
            method="bootstrap",
            alternative="less",
            random_state=59,
        ).run()

        assert result.ci_lower == float("-inf")
        assert math.isfinite(result.ci_upper)


# ═══════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases for robustness."""

    def test_zero_variance_data(self):
        """Identical data in both groups should not crash, should be non-significant."""
        result = Experiment([5, 5, 5, 5, 5], [5, 5, 5, 5, 5]).run()
        assert result.significant is False

    def test_n_equals_2(self):
        """Minimum sample size of 2 per group should work."""
        result = Experiment([0, 1], [0, 1]).run()
        assert isinstance(result, ExperimentResult)
        assert result.control_n == 2
        assert result.treatment_n == 2

    def test_n_bootstrap_too_low_raises(self):
        """n_bootstrap=50 raises ValueError."""
        with pytest.raises(ValueError, match="n_bootstrap"):
            Experiment([0, 1, 0], [1, 0, 1], n_bootstrap=50)

    def test_mannwhitney_large_n_subsampling(self):
        """n1=n2=250 triggers subsampling (250*250=62500 > 50000)."""
        rng = np.random.default_rng(60)
        ctrl = rng.normal(10, 2, size=250)
        trt = rng.normal(11, 2, size=250)

        result = Experiment(ctrl, trt, method="mannwhitney", random_state=60).run()
        assert result.method == "mannwhitney"
        assert result.pvalue < 0.05

    def test_chisquare_alternative_less_raises(self):
        """Chi-square with alternative='less' raises ValueError."""
        with pytest.raises(ValueError, match="two-sided"):
            Experiment(
                [0, 1, 0, 1],
                [1, 1, 0, 1],
                method="chisquare",
                alternative="less",
            ).run()

    def test_invalid_alternative_raises(self):
        """alternative='both' raises ValueError."""
        with pytest.raises(ValueError, match="alternative"):
            Experiment([0, 1, 0], [1, 0, 1], alternative="both")

    def test_ztest_se_zero(self):
        """All-zero data where pooled SE = 0 should not crash."""
        result = Experiment([0, 0, 0, 0, 0], [0, 0, 0, 0, 0], method="ztest").run()
        assert result.pvalue >= 0
        assert result.significant is False

    def test_bootstrap_pvalue_correctness(self):
        """Data with known large effect should give p-value < 0.05."""
        rng = np.random.default_rng(61)
        ctrl = rng.normal(0, 1, size=500)
        trt = rng.normal(3, 1, size=500)  # very large effect

        result = Experiment(ctrl, trt, method="bootstrap", random_state=61).run()
        assert result.pvalue < 0.05
        assert result.significant is True

    def test_delta_zero_effect(self):
        """Ratio data with identical ratios should be non-significant."""
        rng = np.random.default_rng(62)
        n = 500
        ctrl_den = rng.poisson(10, size=n).astype(float) + 1
        trt_den = rng.poisson(10, size=n).astype(float) + 1
        # Same rate (0.10) for both groups
        ctrl_num = rng.binomial(ctrl_den.astype(int), 0.10).astype(float)
        trt_num = rng.binomial(trt_den.astype(int), 0.10).astype(float)

        result = Experiment(
            ctrl_num,
            trt_num,
            control_denominator=ctrl_den,
            treatment_denominator=trt_den,
        ).run()
        # With same rate, should typically not be significant
        assert result.method == "delta"
        assert result.pvalue > 0.01  # not strongly significant


# ═══════════════════════════════════════════════════════════════════════
# Properties — one-sided vs two-sided
# ═══════════════════════════════════════════════════════════════════════


class TestOneSidedProperties:
    """Property: one-sided p-value <= two-sided p-value in correct direction."""

    def test_one_sided_pvalue_less_than_two_sided(self):
        """One-sided p should be <= two-sided p in correct direction."""
        rng = np.random.default_rng(70)
        ctrl = rng.normal(10, 2, size=300)
        trt = rng.normal(11, 2, size=300)  # treatment > control

        two = Experiment(ctrl, trt, method="ttest", alternative="two-sided").run()
        one = Experiment(ctrl, trt, method="ttest", alternative="greater").run()

        assert one.pvalue <= two.pvalue + 1e-12


# ═══════════════════════════════════════════════════════════════════════
# Review fixes — OR Expert + Security
# ═══════════════════════════════════════════════════════════════════════


class TestReviewFixes:
    """Tests for fixes from OR Expert and Security reviews."""

    def test_n_bootstrap_too_high_raises(self):
        """n_bootstrap=2_000_000 raises ValueError."""
        with pytest.raises(ValueError, match="n_bootstrap"):
            Experiment([0, 1, 0], [1, 0, 1], n_bootstrap=2_000_000)

    def test_inf_in_data_warning_and_removal(self):
        """Array with inf values emits warning and removes inf."""
        data_with_inf = [1.0, 2.0, float("inf"), 3.0, float("-inf")]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            exp = Experiment(data_with_inf, [1.0, 2.0, 3.0, 4.0, 5.0])
            inf_warnings = [x for x in w if "infinite" in str(x.message).lower()]
            assert len(inf_warnings) >= 1
        result = exp.run()
        assert result.control_n == 3  # 5 - 2 inf values

    def test_2d_array_in_experiment_raises(self):
        """2D array input raises ValueError."""
        data_2d = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="1-D array"):
            Experiment(data_2d, [1, 2, 3])

    def test_delta_zero_denominator_sum_raises(self):
        """All-zero denominators raise ValueError."""
        ctrl_num = [1.0, 2.0, 3.0]
        trt_num = [2.0, 3.0, 4.0]
        ctrl_den = [0.0, 0.0, 0.0]  # sum is zero
        trt_den = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match="sum to zero"):
            Experiment(
                ctrl_num,
                trt_num,
                metric="ratio",
                control_denominator=ctrl_den,
                treatment_denominator=trt_den,
            ).run()

    def test_random_state_generator(self):
        """Experiment accepts np.random.Generator for random_state."""
        rng = np.random.default_rng(42)
        ctrl = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        trt = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        result = Experiment(ctrl, trt, method="bootstrap", random_state=rng).run()
        assert isinstance(result, ExperimentResult)
        assert 0 <= result.pvalue <= 1
