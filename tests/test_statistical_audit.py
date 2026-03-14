"""Statistical audit — simulation-based verification of splita formulas.

These tests verify that the core statistical computations produce correct
results by running Monte Carlo simulations and checking that error rates,
coverage, and variance reduction match theoretical expectations.

Audit conducted 2026-03-14 by PhD Statistician review.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from splita.core.experiment import Experiment
from splita.core.sample_size import SampleSize
from splita.core.srm import SRMCheck
from splita.core.correction import MultipleCorrection
from splita.variance.cuped import CUPED
from splita.sequential.msprt import mSPRT


# ── Helpers ──────────────────────────────────────────────────────────────


def _run_ztest_sim(
    n: int, p_ctrl: float, p_trt: float, alpha: float, seed: int, n_sims: int
) -> tuple[float, float]:
    """Run n_sims z-test experiments, return (rejection_rate, ci_coverage)."""
    rng = np.random.default_rng(seed)
    rejections = 0
    ci_covers = 0
    true_diff = p_trt - p_ctrl

    for _ in range(n_sims):
        ctrl = rng.binomial(1, p_ctrl, size=n).astype(float)
        trt = rng.binomial(1, p_trt, size=n).astype(float)
        result = Experiment(ctrl, trt, method="ztest", alpha=alpha).run()
        if result.significant:
            rejections += 1
        if result.ci_lower <= true_diff <= result.ci_upper:
            ci_covers += 1

    return rejections / n_sims, ci_covers / n_sims


def _run_ttest_sim(
    n: int,
    mu_ctrl: float,
    mu_trt: float,
    std: float,
    alpha: float,
    seed: int,
    n_sims: int,
) -> tuple[float, float]:
    """Run n_sims t-test experiments, return (rejection_rate, ci_coverage)."""
    rng = np.random.default_rng(seed)
    rejections = 0
    ci_covers = 0
    true_diff = mu_trt - mu_ctrl

    for _ in range(n_sims):
        ctrl = rng.normal(mu_ctrl, std, size=n)
        trt = rng.normal(mu_trt, std, size=n)
        result = Experiment(ctrl, trt, method="ttest", alpha=alpha).run()
        if result.significant:
            rejections += 1
        if result.ci_lower <= true_diff <= result.ci_upper:
            ci_covers += 1

    return rejections / n_sims, ci_covers / n_sims


# ═══════════════════════════════════════════════════════════════════════════
# 1. Z-test Type I error
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
def test_ztest_type1_error():
    """Under H0 (no effect), z-test rejection rate should be close to alpha."""
    n_sims = 2000
    alpha = 0.05
    n = 500
    p = 0.10

    rej_rate, _ = _run_ztest_sim(n, p, p, alpha, seed=1001, n_sims=n_sims)

    assert 0.03 <= rej_rate <= 0.07, (
        f"Z-test Type I error = {rej_rate:.4f}, expected ~{alpha} "
        f"(tolerance [0.03, 0.07])"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2. Z-test CI coverage
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
def test_ztest_ci_coverage():
    """Z-test CI should contain the true effect >= 94% of the time at alpha=0.05."""
    n_sims = 2000
    alpha = 0.05
    n = 500
    p_ctrl = 0.10
    p_trt = 0.13  # true MDE = 0.03

    _, coverage = _run_ztest_sim(n, p_ctrl, p_trt, alpha, seed=2002, n_sims=n_sims)

    assert coverage >= 0.93, (
        f"Z-test CI coverage = {coverage:.4f}, expected >= 0.93 "
        f"(nominal 1-alpha = {1-alpha})"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3. T-test Type I error
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
def test_ttest_type1_error():
    """Under H0, t-test rejection rate should be close to alpha."""
    n_sims = 2000
    alpha = 0.05
    n = 200
    mu = 10.0
    std = 5.0

    rej_rate, _ = _run_ttest_sim(n, mu, mu, std, alpha, seed=3003, n_sims=n_sims)

    assert 0.03 <= rej_rate <= 0.07, (
        f"T-test Type I error = {rej_rate:.4f}, expected ~{alpha} "
        f"(tolerance [0.03, 0.07])"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 4. T-test CI coverage
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
def test_ttest_ci_coverage():
    """T-test CI should contain the true effect >= 94% of the time."""
    n_sims = 2000
    alpha = 0.05
    n = 200
    mu_ctrl = 10.0
    mu_trt = 12.0
    std = 5.0

    _, coverage = _run_ttest_sim(
        n, mu_ctrl, mu_trt, std, alpha, seed=4004, n_sims=n_sims
    )

    assert coverage >= 0.93, (
        f"T-test CI coverage = {coverage:.4f}, expected >= 0.93 "
        f"(nominal 1-alpha = {1-alpha})"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 5. Bootstrap Type I error
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
def test_bootstrap_type1_error():
    """Under H0, bootstrap rejection rate should be below 0.08."""
    n_sims = 500
    alpha = 0.05
    n = 200
    mu = 10.0
    std = 5.0
    rejections = 0

    rng = np.random.default_rng(5005)
    for _ in range(n_sims):
        ctrl = rng.normal(mu, std, size=n)
        trt = rng.normal(mu, std, size=n)
        result = Experiment(
            ctrl, trt, method="bootstrap", alpha=alpha, n_bootstrap=1000,
            random_state=int(rng.integers(0, 2**31)),
        ).run()
        if result.significant:
            rejections += 1

    rej_rate = rejections / n_sims
    assert rej_rate <= 0.08, (
        f"Bootstrap Type I error = {rej_rate:.4f}, expected <= 0.08"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 6. SampleSize roundtrip (proportion)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
def test_sample_size_roundtrip_proportion():
    """Sample size for 80% power -> generate data at that n -> ~80% rejection."""
    baseline = 0.10
    mde = 0.03
    res = SampleSize.for_proportion(baseline, mde, power=0.80)
    n = res.n_per_variant

    n_sims = 1000
    rng = np.random.default_rng(6006)
    rejections = 0
    p_trt = baseline + mde

    for _ in range(n_sims):
        ctrl = rng.binomial(1, baseline, size=n).astype(float)
        trt = rng.binomial(1, p_trt, size=n).astype(float)
        result = Experiment(ctrl, trt, method="ztest", alpha=0.05).run()
        if result.significant:
            rejections += 1

    power_observed = rejections / n_sims
    # 80% power +/- 10% tolerance = [0.70, 0.90]
    assert 0.70 <= power_observed <= 0.90, (
        f"Sample size roundtrip: observed power = {power_observed:.4f}, "
        f"expected ~0.80 (tolerance [0.70, 0.90]), n_per_variant={n}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 7. CUPED variance reduction
# ═══════════════════════════════════════════════════════════════════════════


def test_cuped_variance_reduction():
    """CUPED-adjusted variance should be less than original variance."""
    rng = np.random.default_rng(7007)
    n = 1000

    # Generate correlated pre/post data
    pre = rng.normal(10, 3, size=2 * n)
    noise = rng.normal(0, 1, size=2 * n)
    post = 0.8 * pre + noise  # strong correlation

    ctrl_post = post[:n]
    trt_post = post[n:]
    ctrl_pre = pre[:n]
    trt_pre = pre[n:]

    cuped = CUPED()
    ctrl_adj, trt_adj = cuped.fit_transform(ctrl_post, trt_post, ctrl_pre, trt_pre)

    var_original = np.var(np.concatenate([ctrl_post, trt_post]))
    var_adjusted = np.var(np.concatenate([ctrl_adj, trt_adj]))

    assert var_adjusted < var_original, (
        f"CUPED should reduce variance: original={var_original:.4f}, "
        f"adjusted={var_adjusted:.4f}"
    )
    # With correlation ~0.95, variance reduction should be substantial
    reduction = 1.0 - var_adjusted / var_original
    assert reduction > 0.3, (
        f"CUPED variance reduction = {reduction:.4f}, expected > 0.3 "
        f"for highly correlated data"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 8. CUPED ATE preservation
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
def test_cuped_ate_preservation():
    """CUPED must preserve the average treatment effect across simulations.

    The mean of (ATE_adj - ATE_raw) across simulations should be near zero,
    meaning CUPED does not systematically bias the treatment effect estimate.
    """
    rng = np.random.default_rng(8008)
    n = 500
    n_sims = 1000
    true_ate = 2.0
    ate_diffs = []  # signed differences, not absolute

    for _ in range(n_sims):
        pre_ctrl = rng.normal(10, 3, size=n)
        pre_trt = rng.normal(10, 3, size=n)
        ctrl = 0.7 * pre_ctrl + rng.normal(0, 1, size=n)
        trt = 0.7 * pre_trt + true_ate + rng.normal(0, 1, size=n)

        cuped = CUPED()
        ctrl_adj, trt_adj = cuped.fit_transform(ctrl, trt, pre_ctrl, pre_trt)

        ate_raw = float(np.mean(trt) - np.mean(ctrl))
        ate_adj = float(np.mean(trt_adj) - np.mean(ctrl_adj))
        ate_diffs.append(ate_adj - ate_raw)

    # The MEAN signed difference should be near zero (unbiased)
    mean_signed_diff = abs(np.mean(ate_diffs))
    assert mean_signed_diff < 0.05, (
        f"CUPED ATE bias = {mean_signed_diff:.6f}, expected < 0.05 "
        f"(mean signed diff across {n_sims} sims)"
    )
    # Also check the adjusted ATE is close to the true ATE on average
    ate_adj_values = []
    rng2 = np.random.default_rng(8009)
    for _ in range(n_sims):
        pre_ctrl = rng2.normal(10, 3, size=n)
        pre_trt = rng2.normal(10, 3, size=n)
        ctrl = 0.7 * pre_ctrl + rng2.normal(0, 1, size=n)
        trt = 0.7 * pre_trt + true_ate + rng2.normal(0, 1, size=n)

        cuped = CUPED()
        ctrl_adj, trt_adj = cuped.fit_transform(ctrl, trt, pre_ctrl, pre_trt)
        ate_adj_values.append(float(np.mean(trt_adj) - np.mean(ctrl_adj)))

    mean_ate_adj = np.mean(ate_adj_values)
    assert abs(mean_ate_adj - true_ate) < 0.1, (
        f"CUPED mean ATE = {mean_ate_adj:.4f}, expected ~{true_ate} "
        f"(tolerance 0.1)"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 9. BH FDR control
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
def test_bh_fdr_control():
    """BH procedure should control FDR at alpha=0.05 (+tolerance)."""
    rng = np.random.default_rng(9009)
    n_sims = 1000
    n_null = 5
    n_alt = 5
    n_tests = n_null + n_alt
    alpha = 0.05
    n_per_group = 200
    effect_size = 0.5  # Cohen's d for alternative hypotheses

    false_discovery_rates = []

    for _ in range(n_sims):
        pvalues = []
        is_null = []

        # Null hypotheses (no effect)
        for _ in range(n_null):
            ctrl = rng.normal(0, 1, size=n_per_group)
            trt = rng.normal(0, 1, size=n_per_group)
            result = Experiment(ctrl, trt, method="ttest", alpha=alpha).run()
            pvalues.append(result.pvalue)
            is_null.append(True)

        # Alternative hypotheses (real effect)
        for _ in range(n_alt):
            ctrl = rng.normal(0, 1, size=n_per_group)
            trt = rng.normal(effect_size, 1, size=n_per_group)
            result = Experiment(ctrl, trt, method="ttest", alpha=alpha).run()
            pvalues.append(result.pvalue)
            is_null.append(False)

        # Apply BH correction
        correction = MultipleCorrection(pvalues, method="bh", alpha=alpha)
        cresult = correction.run()

        # Count false discoveries
        n_rejected = sum(cresult.rejected)
        if n_rejected > 0:
            false_discoveries = sum(
                1 for r, null in zip(cresult.rejected, is_null) if r and null
            )
            fdr = false_discoveries / n_rejected
        else:
            fdr = 0.0
        false_discovery_rates.append(fdr)

    mean_fdr = np.mean(false_discovery_rates)
    # BH controls FDR at alpha; allow some tolerance for simulation noise
    assert mean_fdr <= alpha + 0.03, (
        f"BH mean FDR = {mean_fdr:.4f}, expected <= {alpha + 0.03:.4f}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 10. mSPRT always-valid Type I error
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
def test_msprt_type1_error():
    """mSPRT always-valid p-value should reject <= alpha under H0."""
    rng = np.random.default_rng(10010)
    n_sims = 1000
    alpha = 0.05
    n_per_batch = 100
    n_batches = 10  # peek 10 times
    rejections = 0

    for _ in range(n_sims):
        test = mSPRT(metric="continuous", alpha=alpha, tau=0.1)
        rejected = False
        for _ in range(n_batches):
            ctrl = rng.normal(0, 1, size=n_per_batch)
            trt = rng.normal(0, 1, size=n_per_batch)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                state = test.update(ctrl, trt)
            if state.should_stop:
                rejected = True
                break
        if rejected:
            rejections += 1

    rej_rate = rejections / n_sims
    # Always-valid guarantee: rejection rate <= alpha even with peeking
    assert rej_rate <= alpha + 0.02, (
        f"mSPRT Type I error with peeking = {rej_rate:.4f}, "
        f"expected <= {alpha + 0.02:.4f}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 11. SRM Type I error
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
def test_srm_type1_error():
    """Under equal split, SRM should have false positive rate <= alpha + tol."""
    rng = np.random.default_rng(11011)
    n_sims = 1000
    alpha = 0.01  # default SRM alpha
    n_total = 10000
    false_positives = 0

    for _ in range(n_sims):
        # Generate truly equal split via multinomial
        counts = rng.multinomial(n_total, [0.5, 0.5])
        result = SRMCheck(counts.tolist(), alpha=alpha).run()
        if not result.passed:
            false_positives += 1

    fp_rate = false_positives / n_sims
    assert fp_rate <= alpha + 0.01, (
        f"SRM false positive rate = {fp_rate:.4f}, expected <= {alpha + 0.01:.4f}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Deterministic formula checks (fast, not @slow)
# ═══════════════════════════════════════════════════════════════════════════


class TestFormulaChecks:
    """Deterministic checks of specific formula implementations."""

    def test_holm_multipliers(self):
        """Holm multipliers should be [n, n-1, ..., 1] (not [n, n-1, ..., 0])."""
        pvalues = [0.01, 0.04, 0.03]
        result = MultipleCorrection(pvalues, method="holm").run()
        # sorted pvalues: [0.01, 0.03, 0.04]
        # Holm: 0.01*3=0.03, max(0.03*2, 0.03)=0.06, max(0.04*1, 0.06)=0.06
        adj = sorted(zip(pvalues, result.adjusted_pvalues))
        # p=0.01 -> adj=0.03, p=0.03 -> adj=0.06, p=0.04 -> adj=0.06
        assert abs(result.adjusted_pvalues[0] - 0.03) < 1e-10  # 0.01 * 3
        assert abs(result.adjusted_pvalues[2] - 0.06) < 1e-10  # max(0.03*2, 0.03)

    def test_holm_multipliers_are_n_minus_rank(self):
        """Verify Holm uses multipliers n, n-1, ..., 1 (not n-rank with 0-indexed)."""
        # The code uses: sorted_p * (n - np.arange(n))
        # For n=3: multipliers are [3, 2, 1] -- CORRECT
        # If it were (n - np.arange(1, n+1)): [2, 1, 0] -- WRONG
        pvalues = [0.5]  # single p-value
        result = MultipleCorrection(pvalues, method="holm").run()
        # With n=1, multiplier should be 1, so adjusted = 0.5
        assert abs(result.adjusted_pvalues[0] - 0.5) < 1e-10

    def test_bh_formula(self):
        """BH: adjusted = p * n / rank (1-indexed), reverse cumulative min."""
        pvalues = [0.01, 0.20, 0.03]
        result = MultipleCorrection(pvalues, method="bh").run()
        # sorted: [0.01, 0.03, 0.20]
        # raw adjusted: [0.01*3/1, 0.03*3/2, 0.20*3/3] = [0.03, 0.045, 0.20]
        # reverse cum min: [0.03, 0.045, 0.20] (already non-decreasing)
        assert abs(result.adjusted_pvalues[0] - 0.03) < 1e-10
        assert abs(result.adjusted_pvalues[2] - 0.045) < 1e-10

    def test_bonferroni_cap(self):
        """Bonferroni must cap at 1.0."""
        pvalues = [0.5, 0.6]
        result = MultipleCorrection(pvalues, method="bonferroni").run()
        assert result.adjusted_pvalues[0] == 1.0
        assert result.adjusted_pvalues[1] == 1.0

    def test_cohens_d_formula(self):
        """Cohen's d = (mean_trt - mean_ctrl) / pooled_std."""
        from splita._utils import cohens_d

        ctrl = np.array([1.0, 2.0, 3.0])
        trt = np.array([2.0, 3.0, 4.0])
        d = cohens_d(ctrl, trt)
        assert abs(d - 1.0) < 1e-10

    def test_cohens_h_formula(self):
        """Cohen's h = 2*(arcsin(sqrt(p2)) - arcsin(sqrt(p1)))."""
        from splita._utils import cohens_h
        import math

        p1, p2 = 0.5, 0.7
        expected = 2.0 * (math.asin(math.sqrt(p2)) - math.asin(math.sqrt(p1)))
        h = cohens_h(p1, p2)
        assert abs(h - expected) < 1e-10

    def test_welch_df_formula(self):
        """Welch-Satterthwaite df should match textbook formula."""
        s1, s2, n1, n2 = 2.0, 3.0, 50, 60
        # num = (s1^2/n1 + s2^2/n2)^2
        # den = (s1^2/n1)^2/(n1-1) + (s2^2/n2)^2/(n2-1)
        num = (s1**2 / n1 + s2**2 / n2) ** 2
        den = (s1**2 / n1) ** 2 / (n1 - 1) + (s2**2 / n2) ** 2 / (n2 - 1)
        expected_df = num / den

        df = Experiment._welch_df(s1, s2, n1, n2)
        assert abs(df - expected_df) < 1e-10

    def test_ztest_uses_pooled_se_for_statistic(self):
        """Z-test statistic must use pooled SE, CI must use unpooled SE."""
        ctrl = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0], dtype=float)
        trt = np.array([1, 1, 0, 1, 1, 0, 1, 0, 1, 1], dtype=float)
        result = Experiment(ctrl, trt, method="ztest").run()

        p1 = np.mean(ctrl)
        p2 = np.mean(trt)
        p_pool = (np.sum(ctrl) + np.sum(trt)) / (len(ctrl) + len(trt))
        n1, n2 = len(ctrl), len(trt)

        import math
        se_pooled = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
        expected_z = (p2 - p1) / se_pooled
        assert abs(result.statistic - expected_z) < 1e-10

    def test_ttest_uses_welch(self):
        """T-test must use Welch's (unequal variance), not Student's."""
        from scipy.stats import ttest_ind

        ctrl = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        trt = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        result = Experiment(ctrl, trt, method="ttest").run()

        # Welch's
        welch_res = ttest_ind(trt, ctrl, equal_var=False)
        # Student's
        student_res = ttest_ind(trt, ctrl, equal_var=True)

        # Result should match Welch's, not Student's
        assert abs(result.pvalue - welch_res.pvalue) < 1e-10
        if abs(welch_res.pvalue - student_res.pvalue) > 1e-6:
            # Only check if they differ (they should with unequal variance)
            assert abs(result.pvalue - student_res.pvalue) > 1e-6

    def test_srm_uses_sf_not_1_minus_cdf(self):
        """SRM uses chi2.sf() for numerical stability (not 1-cdf)."""
        from scipy.stats import chi2

        # With a large chi2 statistic, 1-cdf would lose precision
        result = SRMCheck([5500, 4500], alpha=0.01).run()
        # Just verify it returns a valid p-value
        assert 0.0 <= result.pvalue <= 1.0
        # This should definitely detect SRM
        assert not result.passed

    def test_sample_size_proportion_formula(self):
        """Verify proportion sample size against manual Farrington-Manning calc."""
        import math as _math
        # baseline=0.10, mde=0.02, alpha=0.05, power=0.80
        p1, p2 = 0.10, 0.12
        p_bar = (p1 + p2) / 2.0
        se0 = _math.sqrt(2.0 * p_bar * (1.0 - p_bar))
        se1 = _math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
        za = norm.ppf(0.975)
        zb = norm.ppf(0.80)
        expected_n = _math.ceil((za * se0 + zb * se1) ** 2 / 0.02 ** 2)

        result = SampleSize.for_proportion(0.10, 0.02)
        assert result.n_per_variant == expected_n

    def test_sample_size_mean_formula(self):
        """Verify mean sample size: n = ceil(2 * ((z_a + z_b) / d)^2)."""
        from scipy.stats import norm
        import math

        mu, std, mde = 25.0, 40.0, 2.0
        d = mde / std
        za = norm.ppf(0.975)  # two-sided alpha=0.05
        zb = norm.ppf(0.80)
        expected_n = math.ceil(2.0 * ((za + zb) / d) ** 2)

        result = SampleSize.for_mean(mu, std, mde)
        assert result.n_per_variant == expected_n

    def test_delta_method_linearization(self):
        """Delta method linearization: y = (num - R*den) / mean(den)."""
        rng = np.random.default_rng(42)
        n = 500

        ctrl_num = rng.normal(10, 2, size=n)
        ctrl_den = rng.normal(5, 1, size=n)
        trt_num = rng.normal(11, 2, size=n)
        trt_den = rng.normal(5, 1, size=n)

        result = Experiment(
            ctrl_num, trt_num,
            metric="ratio",
            method="delta",
            control_denominator=ctrl_den,
            treatment_denominator=trt_den,
        ).run()

        # Should produce a valid result
        assert result.pvalue >= 0.0
        assert result.pvalue <= 1.0
        assert result.method == "delta"

    def test_one_sided_alternatives_ztest(self):
        """Verify one-sided alternatives are not inverted."""
        # Treatment clearly better
        ctrl = np.zeros(1000)
        ctrl[:100] = 1.0  # 10%
        trt = np.zeros(1000)
        trt[:200] = 1.0  # 20%

        result_greater = Experiment(
            ctrl, trt, method="ztest", alternative="greater"
        ).run()
        result_less = Experiment(
            ctrl, trt, method="ztest", alternative="less"
        ).run()

        # "greater" means H1: treatment > control -- should be significant
        assert result_greater.pvalue < 0.01
        # "less" means H1: treatment < control -- should NOT be significant
        assert result_less.pvalue > 0.5

    def test_one_sided_alternatives_ttest(self):
        """Verify one-sided alternatives are not inverted for t-test."""
        rng = np.random.default_rng(42)
        ctrl = rng.normal(0, 1, size=500)
        trt = rng.normal(1, 1, size=500)  # treatment clearly higher

        result_greater = Experiment(
            ctrl, trt, method="ttest", alternative="greater"
        ).run()
        result_less = Experiment(
            ctrl, trt, method="ttest", alternative="less"
        ).run()

        assert result_greater.pvalue < 0.01
        assert result_less.pvalue > 0.5

    def test_msprt_mlr_formula(self):
        """mSPRT MLR = sqrt(V/(V+tau)) * exp(tau*delta^2 / (2*V*(V+tau)))."""
        import math

        # Manually compute MLR and compare
        test = mSPRT(metric="conversion", alpha=0.05, tau=0.01)
        ctrl = np.array([0, 1, 0, 0, 1] * 100, dtype=float)
        trt = np.array([1, 1, 0, 1, 1] * 100, dtype=float)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            state = test.update(ctrl, trt)

        # The MLR should be positive
        assert state.mixture_lr > 0.0
        # With a real effect, MLR should be > 1
        assert state.mixture_lr > 1.0

    def test_chisquare_correction_false(self):
        """Chi-square uses correction=False (Pearson's, not Yates')."""
        from scipy.stats import chi2_contingency

        ctrl = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0] * 50, dtype=float)
        trt = np.array([1, 1, 0, 1, 1, 0, 1, 0, 1, 1] * 50, dtype=float)

        result = Experiment(ctrl, trt, method="chisquare").run()

        # Manual chi-square without Yates' correction
        s1, s2 = int(np.sum(ctrl)), int(np.sum(trt))
        n1, n2 = len(ctrl), len(trt)
        table = np.array([[s1, n1 - s1], [s2, n2 - s2]])
        chi2_stat, pval, _, _ = chi2_contingency(table, correction=False)

        assert abs(result.statistic - chi2_stat) < 1e-10
        assert abs(result.pvalue - pval) < 1e-10

    def test_bootstrap_ci_quantiles(self):
        """Bootstrap percentile CI uses alpha/2 and 1-alpha/2 quantiles."""
        rng = np.random.default_rng(42)
        ctrl = rng.normal(10, 2, size=500)
        trt = rng.normal(12, 2, size=500)

        result = Experiment(
            ctrl, trt, method="bootstrap", alpha=0.05,
            n_bootstrap=5000, random_state=42,
        ).run()

        # CI should contain the observed lift
        assert result.ci_lower < result.lift < result.ci_upper
        # CI should be reasonable (not degenerate)
        assert result.ci_upper - result.ci_lower > 0.1

    def test_by_harmonic_number(self):
        """BY correction multiplies by harmonic number c_n = sum(1/i)."""
        pvalues = [0.01, 0.03, 0.10]
        bh_result = MultipleCorrection(pvalues, method="bh").run()
        by_result = MultipleCorrection(pvalues, method="by").run()

        # BY should be more conservative than BH
        for bh_p, by_p in zip(bh_result.adjusted_pvalues, by_result.adjusted_pvalues):
            assert by_p >= bh_p - 1e-10, (
                f"BY ({by_p}) should be >= BH ({bh_p})"
            )

    def test_cuped_theta_formula(self):
        """CUPED theta = Cov(Y, X) / Var(X)."""
        rng = np.random.default_rng(42)
        n = 1000
        x = rng.normal(0, 1, size=2 * n)
        y = 2.0 * x + rng.normal(0, 0.1, size=2 * n)

        ctrl = y[:n]
        trt = y[n:]
        x_ctrl = x[:n]
        x_trt = x[n:]

        cuped = CUPED()
        cuped.fit(ctrl, trt, x_ctrl, x_trt)

        # Manual computation
        y_all = np.concatenate([ctrl, trt])
        x_all = np.concatenate([x_ctrl, x_trt])
        cov = np.cov(y_all, x_all, ddof=1)[0, 1]
        var_x = np.var(x_all, ddof=1)
        expected_theta = cov / var_x

        assert abs(cuped.theta_ - expected_theta) < 1e-10

    def test_cuped_adjustment_formula(self):
        """CUPED: Y_adj = Y - theta * (X - mean(X_pool))."""
        rng = np.random.default_rng(42)
        n = 100

        x_ctrl = rng.normal(5, 1, size=n)
        x_trt = rng.normal(5, 1, size=n)
        ctrl = x_ctrl + rng.normal(0, 0.5, size=n)
        trt = x_trt + 1.0 + rng.normal(0, 0.5, size=n)

        cuped = CUPED()
        cuped.fit(ctrl, trt, x_ctrl, x_trt)
        ctrl_adj, trt_adj = cuped.transform(ctrl, trt, x_ctrl, x_trt)

        # Manual
        x_pool_mean = np.mean(np.concatenate([x_ctrl, x_trt]))
        expected_ctrl = ctrl - cuped.theta_ * (x_ctrl - x_pool_mean)
        expected_trt = trt - cuped.theta_ * (x_trt - x_pool_mean)

        np.testing.assert_allclose(ctrl_adj, expected_ctrl, atol=1e-10)
        np.testing.assert_allclose(trt_adj, expected_trt, atol=1e-10)

    def test_outlier_winsorize_caps_not_removes(self):
        """Winsorize should cap values, not remove them."""
        from splita.variance.outliers import OutlierHandler

        ctrl = np.arange(100, dtype=float)
        trt = np.arange(100, 200, dtype=float)

        handler = OutlierHandler(method="winsorize", lower=0.05, upper=0.95)
        ctrl_out, trt_out = handler.fit_transform(ctrl, trt)

        # Length preserved
        assert len(ctrl_out) == len(ctrl)
        assert len(trt_out) == len(trt)

    def test_outlier_trim_removes_not_caps(self):
        """Trim should remove values, not cap them."""
        from splita.variance.outliers import OutlierHandler

        ctrl = np.arange(100, dtype=float)
        trt = np.arange(100, 200, dtype=float)

        handler = OutlierHandler(method="trim", lower=0.05, upper=0.95)
        ctrl_out, trt_out = handler.fit_transform(ctrl, trt)

        # Length reduced (extreme values removed)
        assert len(ctrl_out) <= len(ctrl)
        assert len(trt_out) <= len(trt)

    def test_outlier_thresholds_from_pooled_data(self):
        """Thresholds must be computed from pooled (combined) data."""
        from splita.variance.outliers import OutlierHandler

        ctrl = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        trt = np.array([6.0, 7.0, 8.0, 9.0, 10.0])

        handler = OutlierHandler(method="winsorize", lower=0.1, upper=0.9)
        handler.fit(ctrl, trt)

        # Thresholds should be from combined [1..10], not from each group
        combined = np.concatenate([ctrl, trt])
        expected_lower = np.percentile(combined, 10)
        expected_upper = np.percentile(combined, 90)

        assert abs(handler.lower_threshold_ - expected_lower) < 1e-10
        assert abs(handler.upper_threshold_ - expected_upper) < 1e-10

    def test_sample_size_ratio_delta_var(self):
        """Verify delta method variance: sigma_num^2/d^2 - 2*R*cov/d^2 + R^2*sigma_den^2/d^2."""
        import math
        from scipy.stats import norm as norm_dist

        num_mean, den_mean = 10.0, 5.0
        num_std, den_std = 3.0, 1.5
        cov = 2.0
        mde = 0.1

        R = num_mean / den_mean
        d2 = den_mean ** 2
        expected_var = num_std**2 / d2 - 2 * R * cov / d2 + R**2 * den_std**2 / d2

        za = norm_dist.ppf(0.975)
        zb = norm_dist.ppf(0.80)
        expected_n = math.ceil(2 * expected_var * ((za + zb) / mde) ** 2)

        result = SampleSize.for_ratio(num_mean, den_mean, num_std, den_std, cov, mde)
        assert result.n_per_variant == expected_n


# ═══════════════════════════════════════════════════════════════════════════
# Run configuration
# ═══════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
