"""Carryover effect detector for A/B tests.

Compares pre/post metrics in the CONTROL group. If the control group
changes significantly after the experiment, this suggests carryover
contamination from the treatment group.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ttest_ind

from splita._types import CarryoverResult
from splita._validation import (
    check_array_like,
    check_in_range,
    check_is_integer,
)

ArrayLike = list | tuple | np.ndarray


class CarryoverDetector:
    """Detect carryover effects in A/B test experiments.

    Carryover occurs when the treatment effect "leaks" into the control
    group after the experiment ends. This detector compares pre- and
    post-experiment metrics in the control group.

    Parameters
    ----------
    washout_periods : int, default 3
        Number of time periods to skip between experiment end and
        post-measurement (informational only, not used in computation).
    alpha : float, default 0.05
        Significance level for the carryover detection test.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> ctrl_pre = rng.normal(10, 1, 200)
    >>> ctrl_post = rng.normal(10, 1, 200)  # no change
    >>> trt_pre = rng.normal(10, 1, 200)
    >>> trt_post = rng.normal(12, 1, 200)
    >>> detector = CarryoverDetector()
    >>> r = detector.detect(ctrl_pre, ctrl_post, trt_pre, trt_post)
    >>> r.has_carryover
    False
    """

    def __init__(
        self,
        *,
        washout_periods: int = 3,
        alpha: float = 0.05,
    ) -> None:
        check_is_integer(washout_periods, "washout_periods", min_value=0)
        check_in_range(
            alpha,
            "alpha",
            0.0,
            1.0,
            hint="typical values are 0.05, 0.01, or 0.10",
        )
        self._washout_periods = int(washout_periods)
        self._alpha = alpha

    def detect(
        self,
        control_pre: ArrayLike,
        control_post: ArrayLike,
        treatment_pre: ArrayLike,
        treatment_post: ArrayLike,
    ) -> CarryoverResult:
        """Detect carryover contamination.

        Parameters
        ----------
        control_pre : array-like
            Control group metrics before the experiment.
        control_post : array-like
            Control group metrics after the experiment.
        treatment_pre : array-like
            Treatment group metrics before the experiment.
        treatment_post : array-like
            Treatment group metrics after the experiment.

        Returns
        -------
        CarryoverResult
            Detection result with p-value and diagnostics.

        Raises
        ------
        TypeError
            If inputs cannot be converted to numeric arrays.
        ValueError
            If any array has fewer than 2 elements.
        """
        c_pre = check_array_like(control_pre, "control_pre", min_length=2)
        c_post = check_array_like(control_post, "control_post", min_length=2)
        _t_pre = check_array_like(treatment_pre, "treatment_pre", min_length=2)
        _t_post = check_array_like(treatment_post, "treatment_post", min_length=2)

        control_pre_mean = float(np.mean(c_pre))
        control_post_mean = float(np.mean(c_post))

        # Welch's t-test on control pre vs post
        _, pvalue = ttest_ind(c_pre, c_post, equal_var=False)
        pvalue = float(pvalue)

        has_carryover = pvalue < self._alpha

        if has_carryover:
            message = (
                f"Carryover detected: control group changed significantly "
                f"(p={pvalue:.4f} < {self._alpha}). "
                f"Control mean shifted from {control_pre_mean:.4f} to "
                f"{control_post_mean:.4f}."
            )
        else:
            message = (
                f"No carryover detected: control group stable "
                f"(p={pvalue:.4f} >= {self._alpha}). "
                f"Control mean was {control_pre_mean:.4f} pre and "
                f"{control_post_mean:.4f} post."
            )

        return CarryoverResult(
            has_carryover=has_carryover,
            control_change_pvalue=pvalue,
            control_pre_mean=control_pre_mean,
            control_post_mean=control_post_mean,
            message=message,
        )
