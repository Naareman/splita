"""Nonstationary adjustment for A/B tests.

Bias-corrected treatment effect estimator via time-series decomposition,
inspired by Chen et al. (Management Science, 2024).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from splita._types import NonstationaryAdjResult
from splita._validation import check_array_like, check_same_length, format_error

ArrayLike = list | tuple | np.ndarray


class NonstationaryAdjustment:
    """Nonstationary adjustment for time-varying treatment effects.

    Decomposes the treatment effect time series to remove bias from
    non-stationarity in the underlying metric (e.g., day-of-week effects,
    trends).

    Parameters
    ----------
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n = 100
    >>> timestamps = np.arange(n, dtype=float)
    >>> trend = 0.01 * timestamps
    >>> control = 10 + trend + rng.normal(0, 1, n)
    >>> treatment = 12 + trend + rng.normal(0, 1, n)
    >>> adj = NonstationaryAdjustment()
    >>> r = adj.fit_transform(control, treatment, timestamps)
    >>> abs(r.ate_corrected - 2.0) < 1.0
    True
    """

    def __init__(self, *, alpha: float = 0.05) -> None:
        if not 0 < alpha < 1:
            raise ValueError(
                format_error(
                    f"`alpha` must be in (0, 1), got {alpha}.",
                    "alpha represents the significance level.",
                    "typical values are 0.05, 0.01, or 0.10.",
                )
            )
        self._alpha = alpha

    def fit_transform(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
        timestamps: ArrayLike,
    ) -> NonstationaryAdjResult:
        """Compute the bias-corrected treatment effect.

        Parameters
        ----------
        control : array-like
            Control group outcomes, one per time period.
        treatment : array-like
            Treatment group outcomes, one per time period.
        timestamps : array-like
            Time indices corresponding to each observation.

        Returns
        -------
        NonstationaryAdjResult
            Frozen dataclass with corrected and naive estimates.

        Raises
        ------
        TypeError
            If inputs cannot be converted to numeric arrays.
        ValueError
            If arrays have fewer than 3 elements or differ in length.
        """
        c = check_array_like(control, "control", min_length=3)
        t = check_array_like(treatment, "treatment", min_length=3)
        ts = check_array_like(timestamps, "timestamps", min_length=3)
        check_same_length(c, t, "control", "treatment")
        check_same_length(c, ts, "control", "timestamps")

        n = len(c)

        # Naive ATE
        ate_naive = float(np.mean(t) - np.mean(c))

        # Fit linear trend on control to estimate non-stationarity
        # control_i = a + b * ts_i + eps_i
        ts_centered = ts - np.mean(ts)
        slope = float(np.sum(ts_centered * (c - np.mean(c))) / (np.sum(ts_centered**2) + 1e-12))

        # Predicted control trend
        control_trend = np.mean(c) + slope * ts_centered

        # Detrended control and treatment
        c_detrended = c - control_trend + np.mean(c)
        t_detrended = t - control_trend + np.mean(c)

        # Corrected ATE
        ate_corrected = float(np.mean(t_detrended) - np.mean(c_detrended))

        # Bias
        bias = ate_naive - ate_corrected

        # Standard error of the corrected estimator
        diff_detrended = t_detrended - c_detrended
        se = float(np.std(diff_detrended, ddof=1) / np.sqrt(n))

        # Inference
        if se > 0:
            z = ate_corrected / se
            pvalue = float(2.0 * norm.sf(abs(z)))
        else:
            pvalue = 1.0 if ate_corrected == 0 else 0.0

        z_crit = float(norm.ppf(1 - self._alpha / 2))
        ci_lower = ate_corrected - z_crit * se
        ci_upper = ate_corrected + z_crit * se

        return NonstationaryAdjResult(
            ate_corrected=ate_corrected,
            ate_naive=ate_naive,
            bias=bias,
            se=se,
            pvalue=pvalue,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        )
