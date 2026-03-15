"""Dynamic causal effects with doubly robust estimation (Shi, Deng et al. JASA 2022).

Estimates time-varying treatment effects and accounts for time-varying
confounding using a doubly robust estimator at each time period.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import linregress, norm

from splita._types import DynamicResult
from splita._validation import (
    check_array_like,
    check_same_length,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


class DynamicCausalEffect:
    """Estimate time-varying treatment effects with doubly robust estimation.

    Implements a simplified version of the dynamic causal framework from
    Shi, Deng et al. (JASA 2022).  At each time period, the treatment
    effect is estimated using a doubly robust estimator that combines an
    outcome model with an inverse-propensity weighting correction.

    Parameters
    ----------
    alpha : float, default 0.05
        Significance level for per-period tests and the overall test.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> T = 5
    >>> outcomes = [rng.normal(10 + t * 0.5, 1, 100) for t in range(T)]
    >>> treatments = [rng.binomial(1, 0.5, 100) for _ in range(T)]
    >>> timestamps = list(range(T))
    >>> result = DynamicCausalEffect().fit(outcomes, treatments, timestamps)
    >>> isinstance(result.effects_over_time, list)
    True
    """

    def __init__(self, *, alpha: float = 0.05) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(
                format_error(
                    f"`alpha` must be in (0, 1), got {alpha}.",
                    "alpha controls the significance level.",
                    "typical values are 0.05, 0.01, or 0.10.",
                )
            )
        self._alpha = alpha

    def fit(
        self,
        outcomes: list[ArrayLike],
        treatments: list[ArrayLike],
        timestamps: list | np.ndarray,
    ) -> DynamicResult:
        """Estimate the treatment effect at each time period.

        Parameters
        ----------
        outcomes : list of array-like
            Outcome arrays, one per time period.
        treatments : list of array-like
            Binary treatment arrays (0/1), one per time period.
        timestamps : list or array-like
            Time labels for each period. Must have the same length as
            *outcomes* and *treatments*.

        Returns
        -------
        DynamicResult
            Per-period effects, cumulative effect, p-value, and trend.

        Raises
        ------
        ValueError
            If inputs have inconsistent lengths or fewer than 2 periods.
        """
        if not isinstance(outcomes, list):
            raise TypeError(
                format_error(
                    "`outcomes` must be a list of arrays, one per period.",
                    f"got type {type(outcomes).__name__}.",
                )
            )
        if not isinstance(treatments, list):  # pragma: no cover
            raise TypeError(
                format_error(
                    "`treatments` must be a list of arrays, one per period.",
                    f"got type {type(treatments).__name__}.",
                )
            )

        n_periods = len(outcomes)
        if n_periods < 2:
            raise ValueError(
                format_error(
                    "Need at least 2 time periods for dynamic effect estimation.",
                    f"got {n_periods} period(s).",
                    "provide outcome and treatment arrays for multiple time periods.",
                )
            )

        ts = np.asarray(timestamps)
        if len(ts) != n_periods:
            raise ValueError(
                format_error(
                    "`timestamps` must have the same length as `outcomes`.",
                    f"outcomes has {n_periods} periods, timestamps has {len(ts)}.",
                )
            )
        if len(treatments) != n_periods:  # pragma: no cover
            raise ValueError(
                format_error(
                    "`treatments` must have the same length as `outcomes`.",
                    f"outcomes has {n_periods} periods, treatments has {len(treatments)}.",
                )
            )

        effects_over_time: list[dict] = []
        effect_values: list[float] = []

        for t_idx in range(n_periods):
            y = check_array_like(outcomes[t_idx], f"outcomes[{t_idx}]", min_length=2)
            w = check_array_like(treatments[t_idx], f"treatments[{t_idx}]", min_length=2)
            check_same_length(y, w, f"outcomes[{t_idx}]", f"treatments[{t_idx}]")

            unique_w = np.unique(w)
            if not np.all(np.isin(unique_w, [0.0, 1.0])):
                raise ValueError(
                    format_error(
                        f"`treatments[{t_idx}]` must contain only 0 and 1.",
                        f"found unique values: {unique_w.tolist()}.",
                        "encode treatment as 0 (control) and 1 (treatment).",
                    )
                )

            n_trt = int(np.sum(w == 1.0))
            n_ctrl = int(np.sum(w == 0.0))
            if n_trt == 0 or n_ctrl == 0:  # pragma: no cover
                raise ValueError(
                    format_error(
                        f"Period {t_idx} must have both treatment and control units.",
                        f"got {n_trt} treated and {n_ctrl} control units.",
                    )
                )

            # Doubly robust estimator:
            # 1. Propensity score (simple proportion)
            e_hat = n_trt / len(w)

            # 2. Outcome model: group means
            mu1_hat = float(np.mean(y[w == 1.0]))
            mu0_hat = float(np.mean(y[w == 0.0]))

            # 3. DR estimator: tau_DR = E[w*(y - mu1)/e + mu1] - E[(1-w)*(y - mu0)/(1-e) + mu0]
            dr_scores = (
                w * (y - mu1_hat) / e_hat
                + mu1_hat
                - (1.0 - w) * (y - mu0_hat) / (1.0 - e_hat)
                - mu0_hat
            )
            effect = float(np.mean(dr_scores))
            se = float(np.std(dr_scores, ddof=1) / np.sqrt(len(dr_scores)))

            if se > 0:
                z = effect / se
                pval = float(2 * norm.sf(abs(z)))
            else:
                pval = 1.0 if effect == 0 else 0.0  # pragma: no cover

            effects_over_time.append(
                {
                    "period": int(ts[t_idx])
                    if np.issubdtype(ts.dtype, np.integer)
                    else float(ts[t_idx]),
                    "effect": effect,
                    "se": se,
                    "pvalue": pval,
                }
            )
            effect_values.append(effect)

        # Cumulative effect: sum of per-period effects
        cumulative_effect = float(np.sum(effect_values))

        # Overall test: combine per-period z-stats (Stouffer's method)
        z_scores = []
        for rec in effects_over_time:
            if rec["se"] > 0:
                z_scores.append(rec["effect"] / rec["se"])
        if len(z_scores) > 0:
            combined_z = float(np.sum(z_scores) / np.sqrt(len(z_scores)))
            overall_pvalue = float(2 * norm.sf(abs(combined_z)))
        else:  # pragma: no cover
            overall_pvalue = 1.0

        # Trend detection via linear regression on effect values
        trend = self._detect_trend(effect_values)

        return DynamicResult(
            effects_over_time=effects_over_time,
            cumulative_effect=cumulative_effect,
            pvalue=overall_pvalue,
            trend=trend,
        )

    @staticmethod
    def _detect_trend(effects: list[float]) -> str:
        """Detect whether effects are stable, increasing, or decreasing.

        Parameters
        ----------
        effects : list of float
            Effect estimates over time.

        Returns
        -------
        str
            One of ``'stable'``, ``'increasing'``, ``'decreasing'``.
        """
        if len(effects) < 3:  # pragma: no cover
            return "stable"

        x = np.arange(len(effects), dtype=float)
        result = linregress(x, effects)
        slope = result.slope
        pvalue = result.pvalue

        if pvalue < 0.10:
            return "increasing" if slope > 0 else "decreasing"
        return "stable"
