"""Experiment — central A/B test analysis class.

Accepts raw data, infers metric type, selects the appropriate statistical
test, and exposes a ``.run()`` method that returns
:class:`~splita._types.ExperimentResult`.
"""

from __future__ import annotations

import math
import warnings
from typing import Literal

import numpy as np
from scipy.stats import chi2_contingency, mannwhitneyu, norm, ttest_ind

from splita._types import ExperimentResult
from splita._utils import (
    auto_detect_metric,
    cohens_d,
    cohens_h,
    ensure_rng,
    pooled_proportion,
    relative_lift,
)
from splita._validation import (
    check_array_like,
    check_in_range,
    check_one_of,
    check_same_length,
    format_error,
)

_VALID_METRICS = ["auto", "conversion", "continuous", "ratio"]
_VALID_METHODS = [
    "auto",
    "ttest",
    "mannwhitney",
    "ztest",
    "chisquare",
    "delta",
    "bootstrap",
]
_VALID_ALTERNATIVES = ["two-sided", "greater", "less"]

ArrayLike = list | tuple | np.ndarray


class Experiment:
    """Run a frequentist A/B test on two groups of observations.

    Parameters
    ----------
    control : array-like
        Observations from the control group.
    treatment : array-like
        Observations from the treatment group.
    metric : {'auto', 'conversion', 'continuous', 'ratio'}, default 'auto'
        Type of metric.  ``'auto'`` infers from the data.
    method : str, default 'auto'
        Statistical test to use.  One of ``'auto'``, ``'ttest'``,
        ``'mannwhitney'``, ``'ztest'``, ``'chisquare'``, ``'delta'``,
        or ``'bootstrap'``.  ``'auto'`` selects based on metric type.
    alpha : float, default 0.05
        Significance level.
    alternative : {'two-sided', 'greater', 'less'}, default 'two-sided'
        Direction of the test.
    control_denominator : array-like or None, default None
        Denominator values for the control group (required for ratio metrics).
    treatment_denominator : array-like or None, default None
        Denominator values for the treatment group (required for ratio metrics).
    n_bootstrap : int, default 2000
        Number of bootstrap iterations (only used when ``method='bootstrap'``).
    random_state : int, Generator, or None, default None
        Seed for reproducibility of bootstrap resampling.

    Examples
    --------
    >>> import numpy as np
    >>> ctrl = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0])
    >>> trt  = np.array([1, 1, 0, 1, 1, 0, 1, 0, 1, 1])
    >>> result = Experiment(ctrl, trt).run()
    >>> result.metric
    'conversion'
    """

    def __init__(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
        *,
        metric: Literal["auto", "conversion", "continuous", "ratio"] = "auto",
        method: Literal[
            "auto",
            "ttest",
            "mannwhitney",
            "ztest",
            "chisquare",
            "delta",
            "bootstrap",
        ] = "auto",
        alpha: float = 0.05,
        alternative: Literal["two-sided", "greater", "less"] = "two-sided",
        control_denominator: ArrayLike | None = None,
        treatment_denominator: ArrayLike | None = None,
        n_bootstrap: int = 2000,
        random_state: int | np.random.Generator | None = None,
    ):
        # ── validate categorical args ───────────────────────────────
        check_one_of(metric, "metric", _VALID_METRICS)
        check_one_of(method, "method", _VALID_METHODS)
        check_one_of(alternative, "alternative", _VALID_ALTERNATIVES)

        # ── validate alpha ──────────────────────────────────────────
        check_in_range(
            alpha,
            "alpha",
            0.0,
            1.0,
            hint="typical values are 0.05, 0.01, or 0.10",
        )

        # ── validate n_bootstrap ────────────────────────────────────
        if n_bootstrap < 100:
            raise ValueError(
                format_error(
                    f"`n_bootstrap` must be >= 100, got {n_bootstrap}.",
                    "too few bootstrap iterations for reliable inference.",
                    "typical values are 1000-10000.",
                )
            )
        if n_bootstrap > 1_000_000:
            raise ValueError(
                format_error(
                    f"`n_bootstrap` must be at most 1,000,000, got {n_bootstrap}.",
                    "very large bootstrap counts cause excessive memory allocation.",
                    "2,000-10,000 is sufficient for most use cases.",
                )
            )

        # ── convert & clean arrays ──────────────────────────────────
        self._control = check_array_like(
            control,
            "control",
            min_length=2,
        )
        self._treatment = check_array_like(
            treatment,
            "treatment",
            min_length=2,
        )

        # ── denominators (ratio metric) ─────────────────────────────
        self._control_denom: np.ndarray | None = None
        self._treatment_denom: np.ndarray | None = None

        if metric == "ratio" and (control_denominator is None or treatment_denominator is None):
            raise ValueError(
                format_error(
                    "`control_denominator` and `treatment_denominator` are required "
                    "when metric='ratio'.",
                    "ratio metrics need both a numerator and denominator per user.",
                    "pass arrays of denominator values (e.g. pageviews per user).",
                )
            )

        if control_denominator is not None:
            self._control_denom = check_array_like(
                control_denominator,
                "control_denominator",
                min_length=2,
            )
            check_same_length(
                self._control,
                self._control_denom,
                "control",
                "control_denominator",
            )

        if treatment_denominator is not None:
            self._treatment_denom = check_array_like(
                treatment_denominator,
                "treatment_denominator",
                min_length=2,
            )
            check_same_length(
                self._treatment,
                self._treatment_denom,
                "treatment",
                "treatment_denominator",
            )

        # ── store config ────────────────────────────────────────────
        self._alpha = alpha
        self._alternative = alternative
        self._n_bootstrap = n_bootstrap
        self._rng = ensure_rng(random_state)
        self._random_state = random_state
        self._user_method = method  # track what the user originally chose

        # ── infer metric ────────────────────────────────────────────
        if metric == "auto":
            if self._control_denom is not None:
                self._metric = "ratio"
            else:
                self._metric = auto_detect_metric(np.concatenate([self._control, self._treatment]))
        else:
            self._metric = metric

        # ── infer method ────────────────────────────────────────────
        if method == "auto":
            if self._metric == "conversion":
                self._method = "ztest"
            elif self._metric == "continuous":
                self._method = "ttest"
                # check skewness and warn
                self._warn_skewness()
            else:  # ratio
                self._method = "delta"
        else:
            self._method = method

        # ── advisory: sample size ────────────────────────────────────
        from splita._advisory import advise_sample_size

        advise_sample_size(len(self._control), len(self._treatment), self._metric)

    # ── private helpers ──────────────────────────────────────────────

    def _warn_skewness(self) -> None:
        """Emit a RuntimeWarning if either group has high skewness."""
        from scipy.stats import skew as scipy_skew

        for arr, label in [
            (self._control, "Control"),
            (self._treatment, "Treatment"),
        ]:
            if np.std(arr) == 0.0:
                continue  # zero-variance data has no meaningful skewness
            s = float(scipy_skew(arr))
            if abs(s) > 1.5:
                warnings.warn(
                    f"{label} data has high skewness ({s:.1f}). "
                    "The t-test assumes approximate normality via CLT. "
                    "Hint: consider method='mannwhitney' or method='bootstrap' "
                    "for heavy-tailed data.",
                    RuntimeWarning,
                    stacklevel=4,
                )

    def _compute_power(self, effect_size: float, n1: int, n2: int) -> float:
        """Approximate post-hoc power using the normal approximation.

        Notes
        -----
        Post-hoc (observed) power is a deterministic function of the p-value
        and sample size. It does not provide information beyond what is already
        in the p-value (Hoenig & Heisey, 2001). It is included because many
        A/B testing platforms report it, but should be interpreted with caution.
        For prospective power analysis, use ``SampleSize.for_proportion`` or
        ``SampleSize.for_mean`` instead.
        """
        if self._alternative == "two-sided":
            z_alpha = norm.ppf(1 - self._alpha / 2)
        else:
            z_alpha = norm.ppf(1 - self._alpha)

        n_harmonic = 2.0 * n1 * n2 / (n1 + n2)

        if self._metric == "conversion":
            # Cohen's h: power = Phi(|h| * sqrt(n) - z_alpha)
            power_estimate = float(norm.cdf(abs(effect_size) * math.sqrt(n_harmonic) - z_alpha))
        else:
            # Cohen's d: power = Phi(|d| * sqrt(n/2) - z_alpha)
            power_estimate = float(norm.cdf(abs(effect_size) * math.sqrt(n_harmonic / 2) - z_alpha))
        return max(0.0, min(1.0, power_estimate))

    def _build_ci(
        self,
        diff: float,
        se: float,
        crit_value: float,
    ) -> tuple[float, float]:
        """Build confidence interval based on alternative hypothesis."""
        if self._alternative == "two-sided":
            return (diff - crit_value * se, diff + crit_value * se)
        elif self._alternative == "greater":
            return (diff - crit_value * se, float("inf"))
        else:  # less
            return (float("-inf"), diff + crit_value * se)

    @staticmethod
    def _welch_df(s1: float, s2: float, n1: int, n2: int) -> float:
        """Welch-Satterthwaite degrees of freedom."""
        num = (s1**2 / n1 + s2**2 / n2) ** 2
        denom = (s1**2 / n1) ** 2 / (n1 - 1) + (s2**2 / n2) ** 2 / (n2 - 1)
        return num / denom if denom > 0 else float(n1 + n2 - 2)

    # ── test implementations ─────────────────────────────────────────

    def _run_ztest(self) -> ExperimentResult:
        """Two-proportion z-test."""
        ctrl, trt = self._control, self._treatment
        n1, n2 = len(ctrl), len(trt)
        p1, p2 = float(np.mean(ctrl)), float(np.mean(trt))
        p_pool = pooled_proportion(ctrl, trt)

        se = math.sqrt(p_pool * (1 - p_pool) * (1.0 / n1 + 1.0 / n2))
        z = 0.0 if se == 0 else (p2 - p1) / se

        # p-value
        if self._alternative == "two-sided":
            pval = float(2 * norm.sf(abs(z)))
        elif self._alternative == "greater":
            pval = float(norm.sf(z))
        else:
            pval = float(norm.cdf(z))

        # CI (unpooled SE)
        se_unpooled = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        if self._alternative == "two-sided":
            z_crit = float(norm.ppf(1 - self._alpha / 2))
        else:
            z_crit = float(norm.ppf(1 - self._alpha))

        diff = p2 - p1
        ci_lower, ci_upper = self._build_ci(diff, se_unpooled, z_crit)

        effect = cohens_h(p1, p2)

        return ExperimentResult(
            control_mean=p1,
            treatment_mean=p2,
            lift=diff,
            relative_lift=relative_lift(p1, p2),
            pvalue=pval,
            statistic=z,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            significant=pval < self._alpha,
            alpha=self._alpha,
            method="ztest",
            metric=self._metric,
            control_n=n1,
            treatment_n=n2,
            power=self._compute_power(effect, n1, n2),
            effect_size=effect,
        )

    def _run_ttest(self) -> ExperimentResult:
        """Welch's t-test for continuous data."""
        ctrl, trt = self._control, self._treatment
        n1, n2 = len(ctrl), len(trt)
        mean_ctrl, mean_trt = float(np.mean(ctrl)), float(np.mean(trt))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            res = ttest_ind(trt, ctrl, equal_var=False, alternative=self._alternative)
        t_stat = float(0.0 if math.isnan(res.statistic) else res.statistic)
        pval = float(1.0 if math.isnan(res.pvalue) else res.pvalue)

        # CI (Welch's)
        s1 = float(np.std(ctrl, ddof=1))
        s2 = float(np.std(trt, ddof=1))
        se = math.sqrt(s1**2 / n1 + s2**2 / n2)

        df = self._welch_df(s1, s2, n1, n2) if se > 0 else float(n1 + n2 - 2)

        from scipy.stats import t as t_dist

        diff = mean_trt - mean_ctrl
        if self._alternative == "two-sided":
            t_crit = float(t_dist.ppf(1 - self._alpha / 2, df))
        else:
            t_crit = float(t_dist.ppf(1 - self._alpha, df))
        ci_lower, ci_upper = self._build_ci(diff, se, t_crit)

        effect = cohens_d(ctrl, trt)

        return ExperimentResult(
            control_mean=mean_ctrl,
            treatment_mean=mean_trt,
            lift=diff,
            relative_lift=relative_lift(mean_ctrl, mean_trt),
            pvalue=pval,
            statistic=t_stat,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            significant=pval < self._alpha,
            alpha=self._alpha,
            method="ttest",
            metric=self._metric,
            control_n=n1,
            treatment_n=n2,
            power=self._compute_power(effect, n1, n2),
            effect_size=effect,
        )

    def _run_mannwhitney(self) -> ExperimentResult:
        """Mann-Whitney U test with Hodges-Lehmann CI."""
        ctrl, trt = self._control, self._treatment
        n1, n2 = len(ctrl), len(trt)
        mean_ctrl, mean_trt = float(np.mean(ctrl)), float(np.mean(trt))

        res = mannwhitneyu(trt, ctrl, alternative=self._alternative)
        u_stat = float(res.statistic)
        pval = float(res.pvalue)

        # Hodges-Lehmann estimator and Moses CI
        rng = ensure_rng(self._random_state)

        if n1 * n2 > 50_000:
            # Subsample pairwise diffs for large samples (approximation)
            max_pairs = 50_000
            idx1 = rng.integers(0, n2, size=max_pairs)
            idx2 = rng.integers(0, n1, size=max_pairs)
            diffs = trt[idx1] - ctrl[idx2]
        else:
            diffs = (trt[:, None] - ctrl[None, :]).ravel()

        diffs_sorted = np.sort(diffs)
        point_est = float(np.median(diffs_sorted))

        # Moses CI using normal approximation to Mann-Whitney distribution
        m = n1 * n2 / 2.0
        se_U = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)

        if self._alternative == "two-sided":
            z_crit = float(norm.ppf(1 - self._alpha / 2))
        else:
            z_crit = float(norm.ppf(1 - self._alpha))

        c_alpha = int(m - z_crit * se_U)
        total_pairs = len(diffs_sorted)

        c_alpha = max(0, min(c_alpha, total_pairs - 1))

        if self._alternative == "two-sided":
            ci_lower = float(diffs_sorted[c_alpha])
            ci_upper = float(diffs_sorted[total_pairs - 1 - c_alpha])
        elif self._alternative == "greater":
            ci_lower = float(diffs_sorted[c_alpha])
            ci_upper = float("inf")
        else:
            ci_lower = float("-inf")
            ci_upper = float(diffs_sorted[total_pairs - 1 - c_alpha])

        effect = cohens_d(ctrl, trt)

        return ExperimentResult(
            control_mean=mean_ctrl,
            treatment_mean=mean_trt,
            lift=point_est,
            relative_lift=relative_lift(mean_ctrl, mean_trt),
            pvalue=pval,
            statistic=u_stat,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            significant=pval < self._alpha,
            alpha=self._alpha,
            method="mannwhitney",
            metric=self._metric,
            control_n=n1,
            treatment_n=n2,
            power=self._compute_power(effect, n1, n2),
            effect_size=effect,
        )

    def _run_chisquare(self) -> ExperimentResult:
        """Chi-square test for proportions (two-sided only)."""
        if self._alternative != "two-sided":
            raise ValueError(
                format_error(
                    "`chisquare` only supports alternative='two-sided'.",
                    f"got alternative={self._alternative!r}.",
                    "use method='ztest' for one-sided proportion tests.",
                )
            )

        ctrl, trt = self._control, self._treatment
        n1, n2 = len(ctrl), len(trt)
        p1, p2 = float(np.mean(ctrl)), float(np.mean(trt))

        # 2x2 contingency table
        s1, s2 = int(np.sum(ctrl)), int(np.sum(trt))
        table = np.array([[s1, n1 - s1], [s2, n2 - s2]])
        chi2, pval, _, _ = chi2_contingency(table, correction=False)

        # CI same as ztest (unpooled)
        se_unpooled = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        z_crit = float(norm.ppf(1 - self._alpha / 2))
        diff = p2 - p1
        ci_lower = diff - z_crit * se_unpooled
        ci_upper = diff + z_crit * se_unpooled

        effect = cohens_h(p1, p2)

        return ExperimentResult(
            control_mean=p1,
            treatment_mean=p2,
            lift=diff,
            relative_lift=relative_lift(p1, p2),
            pvalue=float(pval),
            statistic=float(chi2),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            significant=pval < self._alpha,
            alpha=self._alpha,
            method="chisquare",
            metric=self._metric,
            control_n=n1,
            treatment_n=n2,
            power=self._compute_power(effect, n1, n2),
            effect_size=effect,
        )

    def _run_delta(self) -> ExperimentResult:
        """Delta method for ratio metrics (Deng et al. 2018)."""
        ctrl_num, trt_num = self._control, self._treatment
        ctrl_den, trt_den = self._control_denom, self._treatment_denom

        n1, n2 = len(ctrl_num), len(trt_num)

        # Validate denominator sums are nonzero
        if np.sum(ctrl_den) == 0 or np.sum(trt_den) == 0:
            raise ValueError(
                format_error(
                    "Denominator values must not sum to zero.",
                    f"control denominator sum: {np.sum(ctrl_den)}, "
                    f"treatment denominator sum: {np.sum(trt_den)}.",
                    "check your denominator data for all-zero arrays.",
                )
            )

        # group-level ratios
        r_ctrl = float(np.sum(ctrl_num)) / float(np.sum(ctrl_den))
        r_trt = float(np.sum(trt_num)) / float(np.sum(trt_den))

        # linearized metric per user
        mean_den_ctrl = float(np.mean(ctrl_den))
        mean_den_trt = float(np.mean(trt_den))

        y_ctrl = (ctrl_num - r_ctrl * ctrl_den) / mean_den_ctrl
        y_trt = (trt_num - r_trt * trt_den) / mean_den_trt

        # t-test on linearized values
        res = ttest_ind(y_trt, y_ctrl, equal_var=False, alternative=self._alternative)
        t_stat = float(res.statistic)
        pval = float(res.pvalue)

        # CI
        s1 = float(np.std(y_ctrl, ddof=1))
        s2 = float(np.std(y_trt, ddof=1))
        se = math.sqrt(s1**2 / n1 + s2**2 / n2)

        diff = r_trt - r_ctrl

        from scipy.stats import t as t_dist

        df = self._welch_df(s1, s2, n1, n2) if se > 0 else float(n1 + n2 - 2)

        if self._alternative == "two-sided":
            t_crit = float(t_dist.ppf(1 - self._alpha / 2, df))
        else:
            t_crit = float(t_dist.ppf(1 - self._alpha, df))
        ci_lower, ci_upper = self._build_ci(diff, se, t_crit)

        # effect size: Cohen's d on linearized values (for power calculation)
        effect = cohens_d(y_ctrl, y_trt)

        return ExperimentResult(
            control_mean=r_ctrl,
            treatment_mean=r_trt,
            lift=diff,
            relative_lift=relative_lift(r_ctrl, r_trt),
            pvalue=pval,
            statistic=t_stat,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            significant=pval < self._alpha,
            alpha=self._alpha,
            method="delta",
            metric=self._metric,
            control_n=n1,
            treatment_n=n2,
            power=self._compute_power(effect, n1, n2),
            effect_size=effect,
        )

    def _run_bootstrap(self) -> ExperimentResult:
        """Bootstrap test for the difference in means."""
        ctrl, trt = self._control, self._treatment
        n1, n2 = len(ctrl), len(trt)
        mean_ctrl, mean_trt = float(np.mean(ctrl)), float(np.mean(trt))

        rng = ensure_rng(self._random_state)

        # Vectorized bootstrap resampling
        boot_ctrl = rng.choice(ctrl, size=(self._n_bootstrap, n1), replace=True)
        boot_trt = rng.choice(trt, size=(self._n_bootstrap, n2), replace=True)
        diffs = np.mean(boot_trt, axis=1) - np.mean(boot_ctrl, axis=1)

        observed_diff = mean_trt - mean_ctrl

        # Shifted bootstrap p-value (center diffs under H0)
        centered_diffs = diffs - np.mean(diffs)
        if self._alternative == "two-sided":
            pval = float(np.mean(np.abs(centered_diffs) >= abs(observed_diff)))
        elif self._alternative == "greater":
            pval = float(np.mean(centered_diffs >= observed_diff))
        else:  # less
            pval = float(np.mean(centered_diffs <= observed_diff))
        # Clamp to avoid p=0 (minimum is 1/n_bootstrap)
        pval = max(pval, 1.0 / len(diffs))

        # CI: percentile method — uses empirical quantiles of the bootstrap
        # distribution rather than diff +/- crit*se, so _build_ci does not apply.
        if self._alternative == "two-sided":
            ci_lower = float(np.percentile(diffs, 100 * self._alpha / 2))
            ci_upper = float(np.percentile(diffs, 100 * (1 - self._alpha / 2)))
        elif self._alternative == "greater":
            ci_lower = float(np.percentile(diffs, 100 * self._alpha))
            ci_upper = float("inf")
        else:
            ci_lower = float("-inf")
            ci_upper = float(np.percentile(diffs, 100 * (1 - self._alpha)))

        effect = cohens_d(ctrl, trt)

        return ExperimentResult(
            control_mean=mean_ctrl,
            treatment_mean=mean_trt,
            lift=observed_diff,
            relative_lift=relative_lift(mean_ctrl, mean_trt),
            pvalue=pval,
            statistic=observed_diff,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            significant=pval < self._alpha,
            alpha=self._alpha,
            method="bootstrap",
            metric=self._metric,
            control_n=n1,
            treatment_n=n2,
            power=self._compute_power(effect, n1, n2),
            effect_size=effect,
        )

    # ── public API ───────────────────────────────────────────────────

    def run(self) -> ExperimentResult:
        """Execute the statistical test and return results.

        Returns a fresh :class:`~splita._types.ExperimentResult` on every
        call (idempotent).

        Returns
        -------
        ExperimentResult
            Frozen dataclass with all test outputs.

        Raises
        ------
        ValueError
            If the method/alternative combination is invalid
            (e.g. chi-square with one-sided test).
        """
        from splita._advisory import advise_method_choice, info

        n1, n2 = len(self._control), len(self._treatment)
        info(f"Auto-detected metric: {self._metric}")
        info(f"Selected method: {self._method}")
        info(f"Running {self._method} test on {n1}+{n2} observations")

        # Always explain auto-selected method (not just in verbose mode)
        if self._user_method == "auto":
            _method_reasons = {
                ("conversion", "ztest"): (
                    "Auto-selected z-test because the metric is binary "
                    "(conversion). The z-test is the standard and most "
                    "powerful test for comparing two proportions."
                ),
                ("continuous", "ttest"): (
                    "Auto-selected Welch's t-test because the metric is "
                    "continuous. The t-test is the standard choice and does "
                    "not assume equal variances."
                ),
                ("ratio", "delta"): (
                    "Auto-selected delta method because the metric is a ratio. "
                    "The delta method correctly handles the covariance between "
                    "numerator and denominator (Deng et al. 2018)."
                ),
            }
            reason = _method_reasons.get((self._metric, self._method))
            if reason:
                info(reason)

        # Advisory: warn if user explicitly chose a sub-optimal method
        if self._user_method != "auto":
            combined = np.concatenate([self._control, self._treatment])
            advise_method_choice(
                combined,
                self._method,
                self._metric,
                n1 + n2,
            )

        # Advisory: ratio metric without delta method
        from splita._advisory import (
            advise_bootstrap_iterations,
            advise_large_effect,
            advise_large_sample_practical_significance,
            advise_ratio_without_delta,
        )

        advise_ratio_without_delta(self._metric, self._method)

        # Advisory: low bootstrap iterations
        if self._method == "bootstrap":
            advise_bootstrap_iterations(self._n_bootstrap)

        dispatch = {
            "ztest": self._run_ztest,
            "ttest": self._run_ttest,
            "mannwhitney": self._run_mannwhitney,
            "chisquare": self._run_chisquare,
            "delta": self._run_delta,
            "bootstrap": self._run_bootstrap,
        }
        result = dispatch[self._method]()

        # Post-result advisories
        advise_large_effect(result.effect_size)
        advise_large_sample_practical_significance(
            result.control_n + result.treatment_n,
            result.pvalue,
            result.effect_size,
        )

        return result
