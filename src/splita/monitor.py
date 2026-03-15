"""Real-time experiment monitoring dashboard data.

Provides :func:`monitor` which returns a snapshot of the current state of
an experiment: effect size, SRM status, guardrail health, projected
completion, and an actionable recommendation.

Examples
--------
>>> import numpy as np
>>> from splita import monitor
>>> rng = np.random.default_rng(42)
>>> ctrl = rng.binomial(1, 0.10, 500).astype(float)
>>> trt = rng.binomial(1, 0.12, 500).astype(float)
>>> result = monitor(ctrl, trt)
>>> result.recommendation in ('continue', 'stop_winner', 'stop_harm', 'investigate')
True
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.stats import chi2, norm

from splita._types import MonitorResult


def _srm_check(n_control: int, n_treatment: int, alpha: float = 0.01) -> bool:
    """Quick SRM check: return True if passed (no mismatch)."""
    total = n_control + n_treatment
    if total == 0:
        return True
    expected = total / 2.0
    chi2_stat = (n_control - expected) ** 2 / expected + (n_treatment - expected) ** 2 / expected
    pvalue = float(1.0 - chi2.cdf(chi2_stat, df=1))
    return pvalue >= alpha


def _ztest_two_sample(control: np.ndarray, treatment: np.ndarray) -> tuple[float, float, float]:
    """Return (lift, z_stat, pvalue) for a two-sample z-test."""
    n_c, n_t = len(control), len(treatment)
    mean_c, mean_t = float(np.mean(control)), float(np.mean(treatment))
    lift = mean_t - mean_c
    var_c = float(np.var(control, ddof=1))
    var_t = float(np.var(treatment, ddof=1))
    se = math.sqrt(var_c / n_c + var_t / n_t)
    if se == 0.0:
        return lift, 0.0, 1.0
    z = lift / se
    pvalue = float(2.0 * (1.0 - norm.cdf(abs(z))))
    return lift, z, pvalue


def _check_guardrails(
    control: np.ndarray,
    treatment: np.ndarray,
    guardrails: Sequence[dict],
) -> list[dict]:
    """Evaluate each guardrail metric.

    Each guardrail dict must have ``'name'`` and ``'threshold'`` keys,
    and optionally ``'direction'`` (``'lower'`` or ``'upper'``, default
    ``'lower'`` meaning treatment should not decrease below threshold).
    """
    results: list[dict] = []
    for g in guardrails:
        name = g["name"]
        threshold = g["threshold"]
        direction = g.get("direction", "lower")
        lift, _, pvalue = _ztest_two_sample(control, treatment)
        if direction == "lower":
            passed = lift >= threshold or pvalue > 0.05
        else:
            passed = lift <= threshold or pvalue > 0.05
        results.append({"name": name, "passed": bool(passed), "value": float(lift)})
    return results


def _predict_significance(z_stat: float, current_n: int, target_n: int) -> tuple[float, bool]:
    """Project p-value at target_n assuming effect size stays constant.

    The z-statistic scales as sqrt(n), so at target_n the projected z is
    z_current * sqrt(target_n / current_n).
    """
    if current_n == 0 or target_n == 0:
        return 1.0, False
    projected_z = z_stat * math.sqrt(target_n / current_n)
    projected_p = float(2.0 * (1.0 - norm.cdf(abs(projected_z))))
    return projected_p, projected_p < 0.05


def monitor(
    control: np.ndarray | list | tuple,
    treatment: np.ndarray | list | tuple,
    *,
    timestamps: np.ndarray | list | tuple | None = None,
    guardrails: Sequence[dict] | None = None,
    target_n: int | None = None,
    daily_users: int | None = None,
) -> MonitorResult:
    """Get a real-time monitoring snapshot for an experiment.

    Parameters
    ----------
    control : array-like
        Observations from the control group so far.
    treatment : array-like
        Observations from the treatment group so far.
    timestamps : array-like or None, default None
        Per-observation timestamps (not used for calculation yet, reserved
        for future trend analysis).
    guardrails : sequence of dict or None, default None
        Each dict must have ``'name'`` (str) and ``'threshold'`` (float).
        Optional ``'direction'`` (``'lower'`` or ``'upper'``).
    target_n : int or None, default None
        Target total sample size for the experiment.
    daily_users : int or None, default None
        Expected daily user count across both variants, used to estimate
        days remaining.

    Returns
    -------
    MonitorResult
        Snapshot with current effect, SRM status, guardrail health,
        projected completion, and recommendation.

    Examples
    --------
    >>> import numpy as np
    >>> ctrl = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0], dtype=float)
    >>> trt = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
    >>> r = monitor(ctrl, trt, target_n=1000, daily_users=20)
    >>> r.recommendation
    'continue'
    """
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)

    n_c, n_t = len(control), len(treatment)
    current_n = n_c + n_t

    # Core test
    lift, z_stat, pvalue = _ztest_two_sample(control, treatment)

    # SRM
    srm_passed = _srm_check(n_c, n_t)

    # Guardrails
    guardrail_status: list[dict] = []
    if guardrails is not None:
        guardrail_status = _check_guardrails(control, treatment, guardrails)

    # Days remaining
    days_remaining: int | None = None
    if target_n is not None and daily_users is not None and daily_users > 0:
        remaining_users = max(0, target_n - current_n)
        days_remaining = math.ceil(remaining_users / daily_users)

    # Projected significance
    predicted_significant = False
    if target_n is not None and current_n > 0:
        _, predicted_significant = _predict_significance(z_stat, current_n, target_n)

    # Recommendation
    recommendation = _make_recommendation(
        pvalue=pvalue,
        lift=lift,
        srm_passed=srm_passed,
        guardrail_status=guardrail_status,
        current_n=current_n,
        target_n=target_n,
    )

    return MonitorResult(
        current_lift=float(lift),
        current_pvalue=float(pvalue),
        current_n=int(current_n),
        srm_passed=srm_passed,
        guardrail_status=guardrail_status,
        days_remaining=days_remaining,
        predicted_significant=predicted_significant,
        recommendation=recommendation,
    )


def _make_recommendation(
    *,
    pvalue: float,
    lift: float,
    srm_passed: bool,
    guardrail_status: list[dict],
    current_n: int,
    target_n: int | None,
) -> str:
    """Determine the recommended action."""
    # SRM failure means something is wrong with the experiment setup
    if not srm_passed:
        return "investigate"

    # Any guardrail failure means potential harm
    if any(not g["passed"] for g in guardrail_status):
        return "stop_harm"

    # If significant and positive, consider stopping early
    if pvalue < 0.05 and lift > 0:
        if target_n is None or current_n >= target_n:
            return "stop_winner"
        # Significant but haven't reached target — still continue
        return "continue"

    # If significant and negative, stop for harm
    if pvalue < 0.05 and lift < 0:
        return "stop_harm"

    return "continue"
