"""Simulate A/B tests before running them.

Generates synthetic data and runs the test many times to show expected
power, typical p-values, and likely CI widths for a given baseline,
effect size, and sample size.

Examples
--------
>>> result = simulate(0.10, 0.02, 5000, random_state=42)
>>> 0.0 <= result.estimated_power <= 1.0
True
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, ttest_ind

from splita._types import SimulationResult
from splita._utils import ensure_rng

ArrayLike = list | tuple | np.ndarray


def simulate(
    baseline: float,
    mde: float,
    n_per_variant: int,
    *,
    metric: str = "conversion",
    n_simulations: int = 1000,
    alpha: float = 0.05,
    random_state: int | np.random.Generator | None = None,
) -> SimulationResult:
    """Simulate what would happen if you ran this A/B test.

    Generates synthetic data, runs the test ``n_simulations`` times,
    shows expected power, typical p-values, likely CI widths.

    Parameters
    ----------
    baseline : float
        Baseline metric value (e.g. 0.10 for 10% conversion).
    mde : float
        Minimum detectable effect (absolute). For conversion, the
        treatment rate is ``baseline + mde``.
    n_per_variant : int
        Number of observations per variant.
    metric : {'conversion', 'continuous'}, default 'conversion'
        Type of metric to simulate.
    n_simulations : int, default 1000
        Number of Monte Carlo simulations.
    alpha : float, default 0.05
        Significance level.
    random_state : int, Generator, or None, default None
        Seed for reproducibility.

    Returns
    -------
    SimulationResult
        Simulation results including power, typical p-values, and
        CI widths.

    Raises
    ------
    ValueError
        If parameters are invalid.
    """
    if n_per_variant < 2:
        raise ValueError(f"`n_per_variant` must be >= 2, got {n_per_variant}.")
    if n_simulations < 10:
        raise ValueError(f"`n_simulations` must be >= 10, got {n_simulations}.")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"`alpha` must be in (0, 1), got {alpha}.")
    if metric not in ("conversion", "continuous"):
        raise ValueError(f"`metric` must be 'conversion' or 'continuous', got {metric!r}.")

    rng = ensure_rng(random_state)

    pvalues = np.empty(n_simulations)
    lifts = np.empty(n_simulations)
    ci_widths = np.empty(n_simulations)

    treatment_value = baseline + mde

    for i in range(n_simulations):
        if metric == "conversion":
            ctrl = rng.binomial(1, baseline, size=n_per_variant).astype(float)
            trt = rng.binomial(1, treatment_value, size=n_per_variant).astype(float)

            p1, p2 = ctrl.mean(), trt.mean()
            p_pool = (ctrl.sum() + trt.sum()) / (2 * n_per_variant)
            se = np.sqrt(p_pool * (1 - p_pool) * 2.0 / n_per_variant)
            if se > 0:
                z = (p2 - p1) / se
                pval = float(2 * norm.sf(abs(z)))
            else:
                pval = 1.0

            # CI (unpooled)
            se_ci = np.sqrt(p1 * (1 - p1) / n_per_variant + p2 * (1 - p2) / n_per_variant)
            z_crit = float(norm.ppf(1 - alpha / 2))
            ci_w = 2 * z_crit * se_ci

            pvalues[i] = pval
            lifts[i] = p2 - p1
            ci_widths[i] = ci_w
        else:
            # continuous
            std = max(abs(baseline) * 0.5, 1.0)
            ctrl = rng.normal(baseline, std, size=n_per_variant)
            trt = rng.normal(treatment_value, std, size=n_per_variant)

            res = ttest_ind(trt, ctrl, equal_var=False)
            pval = float(res.pvalue)

            diff = trt.mean() - ctrl.mean()
            s1 = float(np.std(ctrl, ddof=1))
            s2 = float(np.std(trt, ddof=1))
            se = np.sqrt(s1**2 / n_per_variant + s2**2 / n_per_variant)
            from scipy.stats import t as t_dist

            df_num = (s1**2 / n_per_variant + s2**2 / n_per_variant) ** 2
            df_den = (s1**2 / n_per_variant) ** 2 / max(n_per_variant - 1, 1) + (
                s2**2 / n_per_variant
            ) ** 2 / max(n_per_variant - 1, 1)
            df = df_num / df_den if df_den > 0 else 2 * n_per_variant - 2
            t_crit = float(t_dist.ppf(1 - alpha / 2, df))
            ci_w = 2 * t_crit * se

            pvalues[i] = pval
            lifts[i] = diff
            ci_widths[i] = ci_w

    significant_count = int(np.sum(pvalues < alpha))
    estimated_power = significant_count / n_simulations
    false_negative_rate = 1.0 - estimated_power

    # Recommendation
    if estimated_power >= 0.8:
        recommendation = (
            f"Good power ({estimated_power:.0%}). This experiment is well-sized "
            f"to detect an effect of {mde} with {n_per_variant} users per variant."
        )
    elif estimated_power >= 0.5:
        recommendation = (
            f"Moderate power ({estimated_power:.0%}). Consider increasing "
            f"sample size to improve detection reliability."
        )
    else:
        recommendation = (
            f"Low power ({estimated_power:.0%}). This experiment is unlikely "
            f"to detect an effect of {mde}. Increase sample size substantially "
            f"or use variance reduction (CUPED)."
        )

    return SimulationResult(
        estimated_power=estimated_power,
        median_pvalue=float(np.median(pvalues)),
        median_lift=float(np.median(lifts)),
        ci_width_median=float(np.median(ci_widths)),
        significant_rate=estimated_power,
        false_negative_rate=false_negative_rate,
        recommendation=recommendation,
    )
