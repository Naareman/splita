"""Compare two experiment results.

Test whether two treatment effects are significantly different from
each other. Useful for answering: "Did the effect change between v1
and v2?"

Examples
--------
>>> from splita._types import ExperimentResult
>>> a = ExperimentResult(
...     control_mean=0.10, treatment_mean=0.12, lift=0.02,
...     relative_lift=0.20, pvalue=0.003, statistic=2.97,
...     ci_lower=0.007, ci_upper=0.033, significant=True,
...     alpha=0.05, method="ztest", metric="conversion",
...     control_n=5000, treatment_n=5000, power=0.82,
...     effect_size=0.15,
... )
>>> b = ExperimentResult(
...     control_mean=0.10, treatment_mean=0.11, lift=0.01,
...     relative_lift=0.10, pvalue=0.15, statistic=1.44,
...     ci_lower=-0.004, ci_upper=0.024, significant=False,
...     alpha=0.05, method="ztest", metric="conversion",
...     control_n=5000, treatment_n=5000, power=0.35,
...     effect_size=0.07,
... )
>>> result = compare(a, b)
>>> result.direction
'a_larger'
"""

from __future__ import annotations

import math

from scipy.stats import norm

from splita._types import ComparisonResult, ExperimentResult


def _se_from_result(result: ExperimentResult) -> float:
    """Estimate standard error from the confidence interval."""
    ci_width = result.ci_upper - result.ci_lower
    # For two-sided CI at given alpha: width = 2 * z_crit * se
    z_crit = float(norm.ppf(1 - result.alpha / 2))
    if z_crit > 0:
        return ci_width / (2 * z_crit)
    return 0.0


def compare(
    result_a: ExperimentResult,
    result_b: ExperimentResult,
    *,
    alpha: float = 0.05,
) -> ComparisonResult:
    """Test whether two treatment effects are significantly different.

    Uses a z-test on the difference of effects with pooled standard
    error, assuming the two experiments are independent.

    Parameters
    ----------
    result_a : ExperimentResult
        Result from the first experiment.
    result_b : ExperimentResult
        Result from the second experiment.
    alpha : float, default 0.05
        Significance level for the comparison.

    Returns
    -------
    ComparisonResult
        Comparison result with z-test on the difference of effects.

    Raises
    ------
    TypeError
        If inputs are not ExperimentResult instances.
    ValueError
        If alpha is out of range.
    """
    if not isinstance(result_a, ExperimentResult):
        raise TypeError(f"`result_a` must be an ExperimentResult, got {type(result_a).__name__}.")
    if not isinstance(result_b, ExperimentResult):
        raise TypeError(f"`result_b` must be an ExperimentResult, got {type(result_b).__name__}.")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"`alpha` must be in (0, 1), got {alpha}.")

    effect_a = result_a.lift
    effect_b = result_b.lift

    se_a = _se_from_result(result_a)
    se_b = _se_from_result(result_b)

    difference = effect_b - effect_a
    se_diff = math.sqrt(se_a**2 + se_b**2)

    if se_diff > 0:
        z = difference / se_diff
        pvalue = float(2 * norm.sf(abs(z)))
    else:
        z = 0.0
        pvalue = 1.0 if difference == 0 else 0.0

    significant = pvalue < alpha

    z_crit = float(norm.ppf(1 - alpha / 2))
    ci_lower = difference - z_crit * se_diff
    ci_upper = difference + z_crit * se_diff

    direction = ("b_larger" if difference > 0 else "a_larger") if significant else "equivalent"

    return ComparisonResult(
        effect_a=effect_a,
        effect_b=effect_b,
        difference=difference,
        se=se_diff,
        pvalue=pvalue,
        significant=significant,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        direction=direction,
    )
