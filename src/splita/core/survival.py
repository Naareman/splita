"""Survival experiment — time-to-event analysis for A/B tests.

Implements Kaplan-Meier survival estimation and the log-rank test
(Cox 1972, Kaplan & Meier 1958) for comparing time-to-event outcomes
such as churn, conversion delay, or time-to-purchase.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2, norm

from splita._types import SurvivalResult
from splita._validation import (
    check_array_like,
    check_in_range,
    check_same_length,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


def _kaplan_meier(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute Kaplan-Meier survival curve.

    Parameters
    ----------
    times : np.ndarray
        Observed times.
    events : np.ndarray
        Event indicators (1 = event, 0 = censored).

    Returns
    -------
    unique_times : np.ndarray
        Sorted unique event times.
    survival : np.ndarray
        Survival probability at each time.
    """
    order = np.argsort(times)
    times = times[order]
    events = events[order]

    unique_times = np.unique(times[events == 1])
    survival = np.ones(len(unique_times))
    len(times)
    s = 1.0

    for i, t in enumerate(unique_times):
        # Number who left the risk set before this time (censored or events at earlier times)
        n_left = int(np.sum(times < t))
        n_at_risk_t = len(times) - n_left
        n_events_t = int(np.sum((times == t) & (events == 1)))
        if n_at_risk_t > 0:
            s *= 1.0 - n_events_t / n_at_risk_t
        survival[i] = s

    return unique_times, survival


def _median_survival(km_times: np.ndarray, km_survival: np.ndarray) -> float | None:
    """Extract median survival time from a Kaplan-Meier curve.

    Parameters
    ----------
    km_times : np.ndarray
        Sorted unique event times.
    km_survival : np.ndarray
        Survival probabilities.

    Returns
    -------
    float or None
        Median survival time, or None if survival never drops below 0.5.
    """
    below = np.where(km_survival <= 0.5)[0]
    if len(below) == 0:
        return None  # pragma: no cover
    return float(km_times[below[0]])


class SurvivalExperiment:
    """Time-to-event analysis for comparing survival between two groups.

    Uses the Kaplan-Meier estimator for survival curves and the log-rank
    test for statistical significance (Cox 1972).

    Parameters
    ----------
    alpha : float, default 0.05
        Significance level for the log-rank test.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> time_ctrl = rng.exponential(10, 100)
    >>> event_ctrl = rng.binomial(1, 0.8, 100)
    >>> time_trt = rng.exponential(15, 100)
    >>> event_trt = rng.binomial(1, 0.8, 100)
    >>> exp = SurvivalExperiment()
    >>> exp.fit(time_ctrl, event_ctrl, time_trt, event_trt)  # doctest: +ELLIPSIS
    <splita.core.survival.SurvivalExperiment object at ...>
    >>> r = exp.result()
    >>> r.hazard_ratio > 0
    True
    """

    def __init__(self, *, alpha: float = 0.05) -> None:
        check_in_range(
            alpha,
            "alpha",
            0.0,
            1.0,
            hint="typical values are 0.05, 0.01, or 0.10",
        )
        self._alpha = alpha
        self._result: SurvivalResult | None = None

    def fit(
        self,
        time_control: ArrayLike,
        event_control: ArrayLike,
        time_treatment: ArrayLike,
        event_treatment: ArrayLike,
    ) -> SurvivalExperiment:
        """Fit the survival model and run the log-rank test.

        Parameters
        ----------
        time_control : array-like
            Observed times for the control group.
        event_control : array-like
            Event indicators for the control group (1 = event, 0 = censored).
        time_treatment : array-like
            Observed times for the treatment group.
        event_treatment : array-like
            Event indicators for the treatment group (1 = event, 0 = censored).

        Returns
        -------
        SurvivalExperiment
            The fitted estimator (self).

        Raises
        ------
        TypeError
            If inputs cannot be converted to numeric arrays.
        ValueError
            If arrays have fewer than 2 elements or mismatched lengths.
        """
        t_c = check_array_like(time_control, "time_control", min_length=2)
        e_c = check_array_like(event_control, "event_control", min_length=2)
        t_t = check_array_like(time_treatment, "time_treatment", min_length=2)
        e_t = check_array_like(event_treatment, "event_treatment", min_length=2)

        check_same_length(t_c, e_c, "time_control", "event_control")
        check_same_length(t_t, e_t, "time_treatment", "event_treatment")

        # Validate times are non-negative
        if np.any(t_c < 0) or np.any(t_t < 0):
            raise ValueError(
                format_error(
                    "Survival times must be non-negative.",
                    "found negative values in the time arrays.",
                    "ensure all times are >= 0.",
                )
            )

        # Validate events are 0 or 1
        for arr, name in [(e_c, "event_control"), (e_t, "event_treatment")]:
            unique_vals = np.unique(arr)
            if not np.all(np.isin(unique_vals, [0.0, 1.0])):
                raise ValueError(
                    format_error(
                        f"`{name}` must contain only 0 and 1.",
                        f"found values: {unique_vals.tolist()}.",
                        "use 1 for events and 0 for censored observations.",
                    )
                )

        # Kaplan-Meier curves
        km_times_c, km_surv_c = _kaplan_meier(t_c, e_c)
        km_times_t, km_surv_t = _kaplan_meier(t_t, e_t)

        median_ctrl = _median_survival(km_times_c, km_surv_c)
        median_trt = _median_survival(km_times_t, km_surv_t)

        # Log-rank test
        all_times = np.concatenate([t_c, t_t])
        all_events = np.concatenate([e_c, e_t])
        # Group indicator: 0 = control, 1 = treatment
        np.concatenate([np.zeros(len(t_c)), np.ones(len(t_t))])

        # Get unique event times
        event_times = np.unique(all_times[all_events == 1])

        observed_ctrl = 0.0
        expected_ctrl = 0.0
        variance_sum = 0.0

        n_events_ctrl = int(np.sum(e_c))
        n_events_trt = int(np.sum(e_t))

        for t in event_times:
            # Number at risk in each group
            at_risk_c = float(np.sum(t_c >= t))
            at_risk_t = float(np.sum(t_t >= t))
            at_risk_total = at_risk_c + at_risk_t

            if at_risk_total == 0:  # pragma: no cover
                continue

            # Events at this time
            events_c = float(np.sum((t_c == t) & (e_c == 1)))
            events_t = float(np.sum((t_t == t) & (e_t == 1)))
            events_total = events_c + events_t

            # Expected events in control under null
            expected_c = events_total * at_risk_c / at_risk_total

            observed_ctrl += events_c
            expected_ctrl += expected_c

            # Variance under null (hypergeometric)
            if at_risk_total > 1:
                v = (
                    events_total
                    * at_risk_c
                    * at_risk_t
                    * (at_risk_total - events_total)
                    / (at_risk_total**2 * (at_risk_total - 1))
                )
                variance_sum += v

        # Chi-square statistic for log-rank test
        if variance_sum > 0:
            chi2_stat = (observed_ctrl - expected_ctrl) ** 2 / variance_sum
            logrank_pvalue = float(chi2.sf(chi2_stat, df=1))
        else:  # pragma: no cover
            chi2_stat = 0.0
            logrank_pvalue = 1.0

        # Hazard ratio estimate (Mantel-Haenszel style)
        # HR = (O_trt / E_trt) / (O_ctrl / E_ctrl)
        observed_trt = n_events_trt
        expected_trt = (n_events_ctrl + n_events_trt) - expected_ctrl

        if expected_trt > 0 and expected_ctrl > 0 and observed_ctrl > 0:
            hr = (observed_trt / expected_trt) / (observed_ctrl / expected_ctrl)
        else:  # pragma: no cover
            hr = 1.0

        # CI for log(HR) using 1/sqrt(events) approximation
        total_events = n_events_ctrl + n_events_trt
        if total_events > 0 and hr > 0:
            log_hr = np.log(hr)
            # SE of log(HR) approximation
            se_log_hr = np.sqrt(1.0 / max(n_events_ctrl, 1) + 1.0 / max(n_events_trt, 1))
            z_crit = float(norm.ppf(1 - self._alpha / 2))
            ci_lower = float(np.exp(log_hr - z_crit * se_log_hr))
            ci_upper = float(np.exp(log_hr + z_crit * se_log_hr))
        else:  # pragma: no cover
            ci_lower = 0.0
            ci_upper = float("inf")

        self._result = SurvivalResult(
            hazard_ratio=float(hr),
            logrank_pvalue=logrank_pvalue,
            significant=logrank_pvalue < self._alpha,
            median_survival_ctrl=median_ctrl,
            median_survival_trt=median_trt,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            alpha=self._alpha,
            n_ctrl=len(t_c),
            n_trt=len(t_t),
            n_events_ctrl=n_events_ctrl,
            n_events_trt=n_events_trt,
        )
        return self

    def result(self) -> SurvivalResult:
        """Return the survival analysis result.

        Returns
        -------
        SurvivalResult
            The log-rank test result and survival estimates.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called.
        """
        if self._result is None:
            raise RuntimeError(
                format_error(
                    "SurvivalExperiment must be fitted before calling result().",
                    "call .fit() with time and event data first.",
                )
            )
        return self._result
