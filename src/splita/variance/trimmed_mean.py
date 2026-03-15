"""TrimmedMeanEstimator — robust ATE estimation via trimmed means.

Trims a fixed fraction from each tail of the distribution, then
estimates the treatment effect on the remaining data. This reduces
sensitivity to outliers while preserving a valid standard error.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import t as t_dist

from splita._types import TrimmedMeanResult
from splita._validation import (
    check_array_like,
    check_in_range,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


class TrimmedMeanEstimator:
    """Estimate the ATE using trimmed means for robustness to outliers.

    Trims ``trim_fraction / 2`` from each tail of both the control
    and treatment distributions, then computes a t-test on the
    remaining observations.

    Parameters
    ----------
    trim_fraction : float, default 0.05
        Total fraction to trim (split equally between both tails).
        For example, 0.10 trims 5% from each tail.
    alpha : float, default 0.05
        Significance level for the confidence interval and test.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> ctrl = rng.normal(10, 2, 200)
    >>> trt = rng.normal(10.5, 2, 200)
    >>> result = TrimmedMeanEstimator(trim_fraction=0.10).fit_transform(ctrl, trt)
    >>> result.trim_fraction
    0.1
    """

    def __init__(self, *, trim_fraction: float = 0.05, alpha: float = 0.05):
        check_in_range(
            trim_fraction,
            "trim_fraction",
            0.0,
            1.0,
            hint="typical values are 0.05 or 0.10.",
        )
        check_in_range(
            alpha,
            "alpha",
            0.0,
            1.0,
            hint="typical values are 0.05, 0.01, or 0.10.",
        )
        self._trim_fraction = trim_fraction
        self._alpha = alpha

    def _trim(self, arr: np.ndarray) -> np.ndarray:
        """Trim the top and bottom `trim_fraction / 2` of sorted values."""
        n = len(arr)
        k = int(n * self._trim_fraction / 2)
        if k == 0:
            return arr.copy()
        sorted_arr = np.sort(arr)
        return sorted_arr[k : n - k]

    def fit_transform(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
    ) -> TrimmedMeanResult:
        """Trim both groups and estimate the treatment effect.

        Parameters
        ----------
        control : array-like
            Control group observations.
        treatment : array-like
            Treatment group observations.

        Returns
        -------
        TrimmedMeanResult
            Estimation result with ATE, SE, p-value, and confidence interval.

        Raises
        ------
        TypeError
            If inputs cannot be converted to numeric arrays.
        ValueError
            If inputs are too short or trimming removes all observations.
        """
        ctrl = check_array_like(control, "control", min_length=2)
        trt = check_array_like(treatment, "treatment", min_length=2)

        ctrl_trimmed = self._trim(ctrl)
        trt_trimmed = self._trim(trt)

        if len(ctrl_trimmed) < 2:
            raise ValueError(
                format_error(
                    "Trimming removed too many control observations.",
                    f"only {len(ctrl_trimmed)} remain after trimming "
                    f"{self._trim_fraction * 100:.0f}% from {len(ctrl)} original.",
                    "reduce trim_fraction or increase sample size.",
                )
            )
        if len(trt_trimmed) < 2:
            raise ValueError(
                format_error(
                    "Trimming removed too many treatment observations.",
                    f"only {len(trt_trimmed)} remain after trimming "
                    f"{self._trim_fraction * 100:.0f}% from {len(trt)} original.",
                    "reduce trim_fraction or increase sample size.",
                )
            )

        mean_c = float(np.mean(ctrl_trimmed))
        mean_t = float(np.mean(trt_trimmed))
        ate = mean_t - mean_c

        n_c = len(ctrl_trimmed)
        n_t = len(trt_trimmed)

        se_c = float(np.std(ctrl_trimmed, ddof=1)) / np.sqrt(n_c)
        se_t = float(np.std(trt_trimmed, ddof=1)) / np.sqrt(n_t)
        se = float(np.sqrt(se_c**2 + se_t**2))

        # Welch's degrees of freedom
        if se == 0:
            pvalue = 1.0
            df = max(n_c, n_t) - 1
        else:
            nu_c = se_c**2
            nu_t = se_t**2
            numerator = (nu_c + nu_t) ** 2
            denominator = nu_c**2 / (n_c - 1) + nu_t**2 / (n_t - 1)
            df = numerator / denominator if denominator > 0 else max(n_c, n_t) - 1
            t_stat = ate / se
            pvalue = float(2.0 * t_dist.sf(abs(t_stat), df))

        t_crit = float(t_dist.ppf(1 - self._alpha / 2, df))
        ci_lower = ate - t_crit * se
        ci_upper = ate + t_crit * se
        significant = pvalue < self._alpha

        return TrimmedMeanResult(
            ate=ate,
            se=se,
            pvalue=pvalue,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            significant=significant,
            n_trimmed_control=n_c,
            n_trimmed_treatment=n_t,
            trim_fraction=self._trim_fraction,
        )
