"""FunnelExperiment -- Analyse treatment effects at each step of a conversion funnel.

Computes per-step conversion rate tests and conditional conversion rates
(step N given step N-1) to identify where the treatment helps or hurts.

References
----------
.. [1] Kohavi, R., Tang, D. & Xu, Y. "Trustworthy Online Controlled
       Experiments: A Practical Guide to A/B Testing." Cambridge, 2020.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from splita._types import FunnelResult
from splita._validation import (
    check_in_range,
    check_is_integer,
    format_error,
)


def _proportion_ztest(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    """Two-proportion z-test.  Returns (z-statistic, two-sided p-value)."""
    p1 = x1 / n1 if n1 > 0 else 0.0
    p2 = x2 / n2 if n2 > 0 else 0.0
    p_pool = (x1 + x2) / (n1 + n2) if (n1 + n2) > 0 else 0.0
    se = np.sqrt(p_pool * (1 - p_pool) * (1.0 / n1 + 1.0 / n2)) if (n1 > 0 and n2 > 0) else 0.0
    if se == 0.0:
        return 0.0, 1.0
    z = (p2 - p1) / se
    pvalue = float(2 * norm.sf(abs(z)))
    return float(z), pvalue


class FunnelExperiment:
    """Analyse treatment effect at each step of a conversion funnel.

    Parameters
    ----------
    alpha : float, default 0.05
        Significance level for each per-step test.
    step_names : list of str, optional
        Default names for steps added without an explicit name.

    Examples
    --------
    >>> exp = FunnelExperiment(step_names=["landing", "cart", "checkout", "purchase"])
    >>> exp.add_step(900, 1000, 920, 1000, name="landing")
    >>> exp.add_step(400, 900, 450, 920, name="cart")
    >>> exp.add_step(200, 400, 240, 450, name="checkout")
    >>> exp.add_step(150, 200, 190, 240, name="purchase")
    >>> result = exp.run()
    >>> len(result.step_results) == 4
    True
    """

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        step_names: list[str] | None = None,
    ) -> None:
        check_in_range(
            alpha,
            "alpha",
            0.0,
            1.0,
            hint="typical values are 0.05, 0.01, or 0.10",
        )
        self._alpha = alpha
        self._step_names = step_names
        self._steps: list[dict] = []

    def add_step(
        self,
        control_converted: int,
        control_total: int,
        treatment_converted: int,
        treatment_total: int,
        name: str | None = None,
    ) -> None:
        """Add a funnel step.

        Parameters
        ----------
        control_converted : int
            Number of control users who converted at this step.
        control_total : int
            Number of control users who entered this step.
        treatment_converted : int
            Number of treatment users who converted at this step.
        treatment_total : int
            Number of treatment users who entered this step.
        name : str, optional
            Name for this step.  Falls back to ``step_names[i]`` or ``step_N``.
        """
        for val, val_name in [
            (control_converted, "control_converted"),
            (control_total, "control_total"),
            (treatment_converted, "treatment_converted"),
            (treatment_total, "treatment_total"),
        ]:
            check_is_integer(val, val_name, min_value=0)

        if control_converted > control_total:
            raise ValueError(
                format_error(
                    "`control_converted` can't exceed `control_total`.",
                    detail=f"got {control_converted} converted out of {control_total} total.",
                    hint="converted count must be <= total count.",
                )
            )
        if treatment_converted > treatment_total:
            raise ValueError(
                format_error(
                    "`treatment_converted` can't exceed `treatment_total`.",
                    detail=f"got {treatment_converted} converted out of {treatment_total} total.",
                    hint="converted count must be <= total count.",
                )
            )

        idx = len(self._steps)
        if name is None:
            if self._step_names and idx < len(self._step_names):
                name = self._step_names[idx]
            else:
                name = f"step_{idx}"

        self._steps.append(
            {
                "name": name,
                "control_converted": int(control_converted),
                "control_total": int(control_total),
                "treatment_converted": int(treatment_converted),
                "treatment_total": int(treatment_total),
            }
        )

    def run(self) -> FunnelResult:
        """Execute the funnel analysis and return results.

        Returns
        -------
        FunnelResult
            Frozen dataclass with per-step results, bottleneck step,
            and overall lift.

        Raises
        ------
        ValueError
            If fewer than one step has been added.
        """
        if not self._steps:
            raise ValueError(
                format_error(
                    "No funnel steps have been added.",
                    hint="call add_step() at least once before run().",
                )
            )

        step_results = []
        min_lift = float("inf")
        bottleneck_name = self._steps[0]["name"]

        for i, s in enumerate(self._steps):
            cc = s["control_converted"]
            ct = s["control_total"]
            tc = s["treatment_converted"]
            tt = s["treatment_total"]

            c_rate = cc / ct if ct > 0 else 0.0
            t_rate = tc / tt if tt > 0 else 0.0
            lift = t_rate - c_rate

            _, pvalue = _proportion_ztest(cc, ct, tc, tt)

            # Conditional conversion (step i given step i-1)
            cond_lift = 0.0
            cond_pvalue = 1.0
            if i > 0:
                prev = self._steps[i - 1]
                # Conditional: converted at i / converted at i-1
                prev_cc = prev["control_converted"]
                prev_tc = prev["treatment_converted"]
                if prev_cc > 0 and prev_tc > 0:
                    cond_c_rate = cc / prev_cc
                    cond_t_rate = tc / prev_tc
                    cond_lift = cond_t_rate - cond_c_rate
                    _, cond_pvalue = _proportion_ztest(cc, prev_cc, tc, prev_tc)

            step_result = {
                "name": s["name"],
                "control_rate": float(c_rate),
                "treatment_rate": float(t_rate),
                "lift": float(lift),
                "pvalue": float(pvalue),
                "significant": float(pvalue) < self._alpha,
                "conditional_lift": float(cond_lift),
                "conditional_pvalue": float(cond_pvalue),
            }
            step_results.append(step_result)

            if lift < min_lift:
                min_lift = lift
                bottleneck_name = s["name"]

        # Overall lift: last step rate difference
        first = self._steps[0]
        last = self._steps[-1]
        overall_c = (
            (last["control_converted"] / first["control_total"])
            if first["control_total"] > 0
            else 0.0
        )
        overall_t = (
            (last["treatment_converted"] / first["treatment_total"])
            if first["treatment_total"] > 0
            else 0.0
        )
        overall_lift = float(overall_t - overall_c)

        return FunnelResult(
            step_results=step_results,
            bottleneck_step=bottleneck_name,
            overall_lift=overall_lift,
        )
