"""Counterfactual projections for experiment results.

Provides :func:`what_if` which projects what would happen under different
experimental conditions (sample size, alpha, power) using the observed
effect size from a completed or in-progress experiment.

Examples
--------
>>> from splita import Experiment, what_if
>>> import numpy as np
>>> rng = np.random.default_rng(42)
>>> ctrl = rng.binomial(1, 0.10, 1000).astype(float)
>>> trt = rng.binomial(1, 0.12, 1000).astype(float)
>>> result = Experiment(ctrl, trt).run()
>>> w = what_if(result, n=10000)
>>> w.projected_n
10000
"""

from __future__ import annotations

import math

from scipy.stats import norm

from splita._types import WhatIfResult


def what_if(
    result: object,
    *,
    n: int | None = None,
    alpha: float | None = None,
    power: float | None = None,
) -> WhatIfResult:
    """Project what would happen under different conditions.

    Uses the observed effect size from the result to project significance,
    power, and p-values at different sample sizes or significance levels.

    Parameters
    ----------
    result : ExperimentResult or similar
        Must have ``control_n``, ``treatment_n``, ``lift``, ``pvalue``,
        ``significant``, and ``alpha`` attributes. Also needs either
        ``control_mean`` and ``treatment_mean`` or ``effect_size``.
    n : int or None, default None
        Projected total sample size (split equally). If ``None``, uses
        the original total.
    alpha : float or None, default None
        Projected significance level. If ``None``, uses the original.
    power : float or None, default None
        Not used for projection but included in the message if provided
        (reserved for future use).

    Returns
    -------
    WhatIfResult
        Original vs. projected statistics and a human-readable message.

    Raises
    ------
    ValueError
        If the result object is missing required attributes.

    Examples
    --------
    >>> from splita._types import ExperimentResult
    >>> r = ExperimentResult(
    ...     control_mean=0.10, treatment_mean=0.12,
    ...     lift=0.02, relative_lift=0.2, pvalue=0.12,
    ...     statistic=1.5, ci_lower=-0.005, ci_upper=0.045,
    ...     significant=False, alpha=0.05, method="ztest",
    ...     metric="conversion", control_n=1000,
    ...     treatment_n=1000, power=0.45, effect_size=0.06,
    ... )
    >>> w = what_if(r, n=10000)
    >>> w.projected_n
    10000
    >>> w.projected_significant
    True
    """
    # Extract attributes from result
    _require_attrs(result, ["control_n", "treatment_n", "lift", "pvalue", "significant", "alpha"])

    original_control_n = result.control_n  # type: ignore[attr-defined]
    original_treatment_n = result.treatment_n  # type: ignore[attr-defined]
    original_n = original_control_n + original_treatment_n
    original_lift = result.lift  # type: ignore[attr-defined]
    original_pvalue = result.pvalue  # type: ignore[attr-defined]
    original_significant = result.significant  # type: ignore[attr-defined]
    original_alpha = result.alpha  # type: ignore[attr-defined]

    # Determine projected parameters
    projected_n = n if n is not None else original_n
    projected_alpha = alpha if alpha is not None else original_alpha

    # Estimate standard error from original data
    # se = lift / z_stat, where z_stat = norm.ppf(1 - pvalue/2) * sign(lift)
    if original_pvalue >= 1.0 or original_pvalue == 0.0:
        # Can't infer z from p=0 or p=1; use a fallback
        if original_pvalue == 0.0:
            original_z = 10.0  # very significant
        else:
            original_z = 0.0
    else:
        original_z = float(norm.ppf(1.0 - original_pvalue / 2.0))
        if original_lift < 0:
            original_z = -original_z

    # se at original n
    if original_z != 0.0:
        original_se = abs(original_lift / original_z)
    else:
        # No effect detected, estimate se from means if available
        original_se = _estimate_se_from_result(result, original_n)

    # Project to new n: se scales as 1/sqrt(n)
    if original_n > 0 and projected_n > 0:
        scale_factor = math.sqrt(original_n / projected_n)
        projected_se = original_se * scale_factor
    else:
        projected_se = original_se

    # Projected z and p-value
    if projected_se > 0:
        projected_z = original_lift / projected_se
        projected_pvalue = float(2.0 * (1.0 - norm.cdf(abs(projected_z))))
    else:  # pragma: no cover
        projected_pvalue = 0.0 if original_lift != 0 else 1.0

    projected_significant = projected_pvalue < projected_alpha

    # Projected power: power at projected n for the observed effect
    projected_power = _compute_power(original_lift, projected_se, projected_alpha)

    # Build message
    message = _build_message(
        original_n=original_n,
        projected_n=projected_n,
        original_pvalue=original_pvalue,
        projected_pvalue=projected_pvalue,
        original_significant=original_significant,
        projected_significant=projected_significant,
        projected_power=projected_power,
        original_alpha=original_alpha,
        projected_alpha=projected_alpha,
    )

    return WhatIfResult(
        original_n=int(original_n),
        projected_n=int(projected_n),
        original_pvalue=float(original_pvalue),
        projected_pvalue=float(projected_pvalue),
        original_significant=bool(original_significant),
        projected_significant=bool(projected_significant),
        projected_power=float(projected_power),
        message=message,
    )


def _require_attrs(obj: object, attrs: list[str]) -> None:
    """Raise ValueError if obj is missing any of the listed attributes."""
    missing = [a for a in attrs if not hasattr(obj, a)]
    if missing:
        raise ValueError(
            f"Result object is missing required attributes: {missing}.\n"
            "  Detail: what_if() needs control_n, treatment_n, lift, "
            "pvalue, significant, and alpha.\n"
            "  Hint: pass an ExperimentResult from Experiment().run()."
        )


def _estimate_se_from_result(result: object, n: int) -> float:
    """Fallback SE estimation when z=0."""
    # Try to use control_mean/treatment_mean for variance estimate
    if hasattr(result, "control_mean") and hasattr(result, "treatment_mean"):
        cm = result.control_mean  # type: ignore[attr-defined]
        tm = result.treatment_mean  # type: ignore[attr-defined]
        # Assume proportions: var = p*(1-p)
        var_c = cm * (1.0 - cm) if 0 < cm < 1 else max(cm, 0.01)
        var_t = tm * (1.0 - tm) if 0 < tm < 1 else max(tm, 0.01)
        n_per = n / 2 if n > 0 else 1
        return math.sqrt(var_c / n_per + var_t / n_per)
    return 1.0


def _compute_power(effect: float, se: float, alpha: float) -> float:
    """Compute power for a two-sided z-test."""
    if se <= 0:
        return 1.0 if effect != 0 else 0.0
    z_alpha = float(norm.ppf(1.0 - alpha / 2.0))
    noncentrality = abs(effect) / se
    power = float(norm.cdf(noncentrality - z_alpha) + norm.cdf(-noncentrality - z_alpha))
    return min(max(power, 0.0), 1.0)


def _build_message(
    *,
    original_n: int,
    projected_n: int,
    original_pvalue: float,
    projected_pvalue: float,
    original_significant: bool,
    projected_significant: bool,
    projected_power: float,
    original_alpha: float,
    projected_alpha: float,
) -> str:
    """Build a human-readable what-if message."""
    parts: list[str] = []

    if projected_n != original_n:
        parts.append(f"With {projected_n:,} total users instead of {original_n:,}:")
    elif projected_alpha != original_alpha:
        parts.append(f"With alpha={projected_alpha} instead of {original_alpha}:")
    else:
        parts.append("Under the same conditions:")

    parts.append(f"  p-value would be {projected_pvalue:.4f} (was {original_pvalue:.4f})")

    if projected_significant and not original_significant:
        parts.append("  The result WOULD be statistically significant.")
    elif not projected_significant and original_significant:
        parts.append("  The result would NOT be statistically significant.")
    elif projected_significant:
        parts.append("  The result would remain statistically significant.")
    else:
        parts.append("  The result would still not be statistically significant.")

    parts.append(f"  Projected power: {projected_power:.1%}")

    return "\n".join(parts)
