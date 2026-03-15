"""Switchback experiment analysis.

Handles time-based switchback designs where treatment and control alternate
across time periods.  Analysis is performed at the period level by
averaging outcomes within each period and comparing treatment vs control
period means.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import t as t_dist

from splita._types import SwitchbackResult
from splita._validation import (
    check_array_like,
    check_in_range,
    check_same_length,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


class SwitchbackExperiment:
    """Analyse a switchback (time-alternating) experiment.

    In a switchback design, the entire population alternates between
    treatment and control across discrete time periods.  This class
    averages observations within each period and then performs a
    two-sample t-test on the period-level means.

    Parameters
    ----------
    outcomes : array-like
        Outcome values for all observations (unit level).
    treatments : array-like
        Binary treatment indicator for each observation (0 or 1).
    time_periods : array-like
        Time-period label for each observation.
    alpha : float, default 0.05
        Significance level.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n_per_period = 50
    >>> outcomes, treatments, periods = [], [], []
    >>> for t in range(20):
    ...     trt = t % 2
    ...     y = rng.normal(10 + 2 * trt, 1, n_per_period)
    ...     outcomes.extend(y)
    ...     treatments.extend([trt] * n_per_period)
    ...     periods.extend([t] * n_per_period)
    >>> result = SwitchbackExperiment(
    ...     outcomes, treatments, periods
    ... ).run()
    >>> result.n_periods
    20
    """

    def __init__(
        self,
        outcomes: ArrayLike,
        treatments: ArrayLike,
        time_periods: ArrayLike,
        *,
        alpha: float = 0.05,
    ) -> None:
        check_in_range(
            alpha,
            "alpha",
            0.0,
            1.0,
            hint="typical values are 0.05, 0.01, or 0.10",
        )

        self._outcomes = check_array_like(outcomes, "outcomes", min_length=2)
        self._treatments = check_array_like(treatments, "treatments", min_length=2)

        check_same_length(self._outcomes, self._treatments, "outcomes", "treatments")

        # Time periods — allow non-numeric
        tp = np.asarray(time_periods)
        if tp.ndim != 1:
            raise ValueError(
                format_error(
                    "`time_periods` must be a 1-D array.",
                    f"got {tp.ndim}-D array with shape {tp.shape}.",
                )
            )
        if len(tp) != len(self._outcomes):
            raise ValueError(
                format_error(
                    "`outcomes` and `time_periods` must have the same length.",
                    f"outcomes has {len(self._outcomes)} elements, "
                    f"time_periods has {len(tp)} elements.",
                )
            )

        # Validate treatments are 0/1
        unique_trt = np.unique(self._treatments)
        if not np.all(np.isin(unique_trt, [0.0, 1.0])):
            raise ValueError(
                format_error(
                    "`treatments` must contain only 0 and 1.",
                    f"found unique values: {unique_trt.tolist()}.",
                    "encode control as 0 and treatment as 1.",
                )
            )

        self._time_periods = tp
        self._alpha = alpha

    def run(self) -> SwitchbackResult:
        """Run the switchback analysis.

        Returns
        -------
        SwitchbackResult
            Period-level inference results.

        Raises
        ------
        ValueError
            If there are fewer than 2 treatment or control periods.
        """
        unique_periods = np.unique(self._time_periods)

        ctrl_period_means = []
        trt_period_means = []

        for p in unique_periods:
            mask = self._time_periods == p
            period_outcomes = self._outcomes[mask]
            period_treatments = self._treatments[mask]

            # Determine if this is a treatment or control period
            # (majority rule — in a proper switchback all units in a period
            # get the same treatment, but we handle mixed gracefully)
            trt_frac = float(np.mean(period_treatments))
            period_mean = float(np.mean(period_outcomes))

            if trt_frac > 0.5:
                trt_period_means.append(period_mean)
            else:
                ctrl_period_means.append(period_mean)

        ctrl_means = np.array(ctrl_period_means)
        trt_means = np.array(trt_period_means)

        n_ctrl = len(ctrl_means)
        n_trt = len(trt_means)

        if n_ctrl < 2:
            raise ValueError(
                format_error(
                    "Need at least 2 control periods for inference.",
                    f"got {n_ctrl} control period(s).",
                    "increase the number of switchback periods.",
                )
            )
        if n_trt < 2:
            raise ValueError(
                format_error(
                    "Need at least 2 treatment periods for inference.",
                    f"got {n_trt} treatment period(s).",
                    "increase the number of switchback periods.",
                )
            )

        mean_ctrl = float(np.mean(ctrl_means))
        mean_trt = float(np.mean(trt_means))
        lift = mean_trt - mean_ctrl

        # Welch's t-test on period means
        s_ctrl = float(np.std(ctrl_means, ddof=1))
        s_trt = float(np.std(trt_means, ddof=1))
        se = float(np.sqrt(s_ctrl**2 / n_ctrl + s_trt**2 / n_trt))

        if se > 0:
            t_stat = lift / se
            # Welch-Satterthwaite df
            num = (s_ctrl**2 / n_ctrl + s_trt**2 / n_trt) ** 2
            denom = (s_ctrl**2 / n_ctrl) ** 2 / (n_ctrl - 1) + (s_trt**2 / n_trt) ** 2 / (n_trt - 1)
            df = num / denom if denom > 0 else float(n_ctrl + n_trt - 2)
            pvalue = float(2 * t_dist.sf(abs(t_stat), df))
            t_crit = float(t_dist.ppf(1 - self._alpha / 2, df))
            ci_lower = lift - t_crit * se
            ci_upper = lift + t_crit * se
        else:
            pvalue = 1.0 if lift == 0 else 0.0
            ci_lower = lift
            ci_upper = lift

        return SwitchbackResult(
            lift=lift,
            pvalue=pvalue,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            significant=pvalue < self._alpha,
            n_periods=len(unique_periods),
            n_treatment_periods=n_trt,
            n_control_periods=n_ctrl,
        )
