"""GuardrailMonitor — monitor safety metrics during experiments.

Runs statistical tests on guardrail metrics and recommends whether
to continue, caution, or stop an experiment based on whether any
guardrail has been breached.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from splita._types import GuardrailResult
from splita._validation import (
    check_array_like,
    check_in_range,
    check_one_of,
    format_error,
)
from splita.core.experiment import Experiment

ArrayLike = list | tuple | np.ndarray

_VALID_DIRECTIONS = ["increase", "decrease", "any"]


class GuardrailMonitor:
    """Monitor safety metrics and recommend stopping if guardrails breach.

    Adds one or more guardrail metrics, then runs a Bonferroni-corrected
    significance test for each. If any metric degrades significantly in
    the specified direction, the monitor recommends stopping.

    Parameters
    ----------
    alpha : float, default 0.05
        Family-wise significance level. Bonferroni correction is applied
        across all guardrails.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> ctrl = rng.normal(100, 10, 500)
    >>> trt = rng.normal(100, 10, 500)
    >>> mon = GuardrailMonitor(alpha=0.05)
    >>> mon.add_guardrail("latency", ctrl, trt, direction="increase")
    >>> result = mon.check()
    >>> result.all_passed
    True
    """

    def __init__(self, *, alpha: float = 0.05):
        check_in_range(
            alpha,
            "alpha",
            0.0,
            1.0,
            hint="typical values are 0.05, 0.01, or 0.10",
        )
        self._alpha = alpha
        self._guardrails: list[dict] = []

    def add_guardrail(
        self,
        name: str,
        control: ArrayLike,
        treatment: ArrayLike,
        direction: Literal["increase", "decrease", "any"] = "any",
        threshold: float | None = None,
    ) -> None:
        """Register a guardrail metric.

        Parameters
        ----------
        name : str
            Human-readable name for the guardrail.
        control : array-like
            Control group observations for this metric.
        treatment : array-like
            Treatment group observations for this metric.
        direction : {'increase', 'decrease', 'any'}, default 'any'
            Which direction is "bad". ``'increase'`` means a significant
            increase is a breach; ``'decrease'`` means a significant
            decrease is a breach; ``'any'`` means either direction.
        threshold : float or None, default None
            If given, the guardrail also breaches when the absolute lift
            exceeds this value, regardless of statistical significance.

        Raises
        ------
        ValueError
            If *name* is empty, *direction* is invalid, or arrays are
            too short.
        """
        if not name or not name.strip():
            raise ValueError(
                format_error(
                    "`name` can't be empty.",
                    "guardrail name is required for identification.",
                )
            )

        check_one_of(direction, "direction", _VALID_DIRECTIONS)

        ctrl = check_array_like(control, "control", min_length=2)
        trt = check_array_like(treatment, "treatment", min_length=2)

        if threshold is not None and threshold < 0:
            raise ValueError(
                format_error(
                    f"`threshold` must be >= 0, got {threshold}.",
                    "threshold represents an absolute magnitude.",
                    "use a positive value like 0.01 or 0.05.",
                )
            )

        self._guardrails.append(
            {
                "name": name,
                "control": ctrl,
                "treatment": trt,
                "direction": direction,
                "threshold": threshold,
            }
        )

    def check(self) -> GuardrailResult:
        """Run all guardrail checks and return the aggregated result.

        Returns
        -------
        GuardrailResult
            Aggregated result with per-guardrail details.

        Raises
        ------
        ValueError
            If no guardrails have been added.
        """
        if len(self._guardrails) == 0:
            raise ValueError(
                format_error(
                    "No guardrails have been added.",
                    "call add_guardrail() at least once before check().",
                )
            )

        n_guardrails = len(self._guardrails)
        adjusted_alpha = self._alpha / n_guardrails  # Bonferroni

        results = []
        for g in self._guardrails:
            exp = Experiment(g["control"], g["treatment"], alpha=adjusted_alpha)
            exp_result = exp.run()

            pvalue = exp_result.pvalue
            lift = exp_result.lift
            direction = g["direction"]

            # Determine if breached by direction
            breached_by_direction = False
            if pvalue < adjusted_alpha and (
                direction == "any"
                or (direction == "increase" and lift > 0)
                or (direction == "decrease" and lift < 0)
            ):
                breached_by_direction = True

            # Determine if breached by threshold
            breached_by_threshold = False
            if g["threshold"] is not None and abs(lift) > g["threshold"]:
                breached_by_threshold = True

            passed = not (breached_by_direction or breached_by_threshold)

            results.append(
                {
                    "name": g["name"],
                    "passed": passed,
                    "pvalue": float(pvalue),
                    "lift": float(lift),
                    "direction": direction,
                }
            )

        all_passed = all(r["passed"] for r in results)
        n_failed = sum(1 for r in results if not r["passed"])

        if all_passed:
            recommendation = "safe"
            message = f"All {n_guardrails} guardrail(s) passed."
        elif n_failed == n_guardrails:
            recommendation = "stop"
            failed_names = [r["name"] for r in results if not r["passed"]]
            message = (
                f"All {n_guardrails} guardrail(s) breached: "
                f"{', '.join(failed_names)}. Recommend stopping."
            )
        else:
            recommendation = "warning"
            failed_names = [r["name"] for r in results if not r["passed"]]
            message = (
                f"{n_failed} of {n_guardrails} guardrail(s) breached: "
                f"{', '.join(failed_names)}. Proceed with caution."
            )

        return GuardrailResult(
            all_passed=all_passed,
            guardrail_results=results,
            message=message,
            recommendation=recommendation,
        )
