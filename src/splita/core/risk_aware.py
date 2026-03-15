"""RiskAwareDecision — constrained decision-making for A/B tests.

Supports user-specified tradeoff bounds across multiple metrics.

References
----------
.. [1] Spotify (2024).  "Risk-aware experimentation: balancing metric
       tradeoffs in constrained decision making."
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm, ttest_ind

from splita._types import RiskAwareResult
from splita._validation import (
    check_array_like,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


class RiskAwareDecision:
    """Multi-metric constrained decision framework.

    Enables shipping decisions that respect user-specified bounds on
    metric tradeoffs. For example: "ship if revenue increases by at
    least 1%, AND latency does not degrade by more than 5ms."

    Parameters
    ----------
    alpha : float, default 0.05
        Significance level for per-metric tests.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> rd = RiskAwareDecision()
    >>> rd.add_metric("revenue", rng.normal(10, 2, 500),
    ...               rng.normal(10.5, 2, 500), min_acceptable=0.0)
    >>> rd.add_metric("latency", rng.normal(100, 10, 500),
    ...               rng.normal(101, 10, 500), max_acceptable=5.0)
    >>> result = rd.decide()
    >>> result.decision in ("ship", "hold", "investigate")
    True
    """

    def __init__(self, *, alpha: float = 0.05) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(
                format_error(
                    f"`alpha` must be in (0, 1), got {alpha}.",
                    hint="typical values are 0.05, 0.01, or 0.10.",
                )
            )
        self._alpha = alpha
        self._metrics: dict[str, dict] = {}

    def add_metric(
        self,
        name: str,
        control: ArrayLike,
        treatment: ArrayLike,
        min_acceptable: float | None = None,
        max_acceptable: float | None = None,
    ) -> RiskAwareDecision:
        """Add a metric with optional constraint bounds.

        Parameters
        ----------
        name : str
            Name of the metric.
        control : array-like
            Control group observations.
        treatment : array-like
            Treatment group observations.
        min_acceptable : float or None
            Minimum acceptable lift (lower CI must exceed this).
        max_acceptable : float or None
            Maximum acceptable lift (upper CI must not exceed this).

        Returns
        -------
        RiskAwareDecision
            Self, for method chaining.

        Raises
        ------
        ValueError
            If the metric name is empty or already added.
        """
        if not name:
            raise ValueError(
                format_error(
                    "`name` can't be empty.",
                    hint="provide a descriptive metric name.",
                )
            )

        if name in self._metrics:
            raise ValueError(
                format_error(
                    f"Metric '{name}' has already been added.",
                    hint="use a unique name for each metric.",
                )
            )

        ctrl = check_array_like(control, f"control ({name})", min_length=2)
        trt = check_array_like(treatment, f"treatment ({name})", min_length=2)

        self._metrics[name] = {
            "control": ctrl,
            "treatment": trt,
            "min_acceptable": min_acceptable,
            "max_acceptable": max_acceptable,
        }
        return self

    def decide(self) -> RiskAwareResult:
        """Evaluate all metrics and produce a constrained decision.

        Returns
        -------
        RiskAwareResult
            Decision with constraint analysis.

        Raises
        ------
        RuntimeError
            If no metrics have been added.
        """
        if not self._metrics:
            raise RuntimeError(
                format_error(
                    "can't decide without any metrics.",
                    detail="no metrics have been added yet.",
                    hint="call add_metric() at least once before decide().",
                )
            )

        violations: list[str] = []
        metric_details: dict[str, dict] = {}
        any_significant_positive = False

        for name, info in self._metrics.items():
            ctrl = info["control"]
            trt = info["treatment"]
            min_acc = info["min_acceptable"]
            max_acc = info["max_acceptable"]

            # Welch's t-test
            res = ttest_ind(trt, ctrl, equal_var=False)
            lift = float(np.mean(trt) - np.mean(ctrl))
            pvalue = float(res.pvalue)
            significant = pvalue < self._alpha

            # CI
            n_c, n_t = len(ctrl), len(trt)
            se = math.sqrt(float(np.var(ctrl, ddof=1)) / n_c + float(np.var(trt, ddof=1)) / n_t)
            z_crit = float(norm.ppf(1.0 - self._alpha / 2.0))
            ci_lower = lift - z_crit * se
            ci_upper = lift + z_crit * se

            # Check constraints
            violated = False
            if min_acc is not None and ci_lower < min_acc:
                violated = True
            if max_acc is not None and ci_upper > max_acc:
                violated = True

            if violated:
                violations.append(name)

            if significant and lift > 0:
                any_significant_positive = True

            metric_details[name] = {
                "lift": lift,
                "pvalue": pvalue,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "significant": significant,
            }

        # Decision logic
        constraints_met = len(violations) == 0

        if constraints_met and any_significant_positive:
            decision = "ship"
        elif not constraints_met:
            decision = "hold"
        else:
            decision = "investigate"

        return RiskAwareResult(
            decision=decision,
            constraints_met=constraints_met,
            violations=violations,
            metric_details=metric_details,
        )
