"""Difference-in-Differences (DiD) estimator.

Classic two-period, two-group DiD:
    ATT = (post_trt - pre_trt) - (post_ctrl - pre_ctrl)

Includes a parallel-trends check comparing pre-period means.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, ttest_ind

from splita._types import DiDResult
from splita._validation import check_array_like, check_in_range, format_error

ArrayLike = list | tuple | np.ndarray


class DifferenceInDifferences:
    """Difference-in-Differences estimator for causal inference.

    Estimates the average treatment effect on the treated (ATT) by
    comparing pre/post changes across treatment and control groups.

    Parameters
    ----------
    alpha : float, default 0.05
        Significance level for inference.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> pre_ctrl = rng.normal(10, 1, 100)
    >>> pre_trt = rng.normal(10, 1, 100)
    >>> post_ctrl = rng.normal(10, 1, 100)
    >>> post_trt = rng.normal(13, 1, 100)  # +3 effect
    >>> did = DifferenceInDifferences()
    >>> did.fit(pre_ctrl, pre_trt, post_ctrl, post_trt)  # doctest: +ELLIPSIS
    <splita.causal.did.DifferenceInDifferences object at ...>
    >>> r = did.result()
    >>> abs(r.att - 3.0) < 1.0
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
        self._result: DiDResult | None = None

    def fit(
        self,
        pre_control: ArrayLike,
        pre_treatment: ArrayLike,
        post_control: ArrayLike,
        post_treatment: ArrayLike,
    ) -> DifferenceInDifferences:
        """Fit the DiD model.

        Parameters
        ----------
        pre_control : array-like
            Control group outcomes in the pre-treatment period.
        pre_treatment : array-like
            Treatment group outcomes in the pre-treatment period.
        post_control : array-like
            Control group outcomes in the post-treatment period.
        post_treatment : array-like
            Treatment group outcomes in the post-treatment period.

        Returns
        -------
        DifferenceInDifferences
            The fitted estimator (self).

        Raises
        ------
        TypeError
            If inputs cannot be converted to numeric arrays.
        ValueError
            If any array has fewer than 2 elements.
        """
        pre_c = check_array_like(pre_control, "pre_control", min_length=2)
        pre_t = check_array_like(pre_treatment, "pre_treatment", min_length=2)
        post_c = check_array_like(post_control, "post_control", min_length=2)
        post_t = check_array_like(post_treatment, "post_treatment", min_length=2)

        # Group means
        mean_pre_c = float(np.mean(pre_c))
        mean_pre_t = float(np.mean(pre_t))
        mean_post_c = float(np.mean(post_c))
        mean_post_t = float(np.mean(post_t))

        # ATT = (post_trt - pre_trt) - (post_ctrl - pre_ctrl)
        att = (mean_post_t - mean_pre_t) - (mean_post_c - mean_pre_c)

        # SE via delta method:
        # Var(ATT) = Var(mean_post_t)/n_post_t + Var(mean_pre_t)/n_pre_t
        #          + Var(mean_post_c)/n_post_c + Var(mean_pre_c)/n_pre_c
        var_pre_c = float(np.var(pre_c, ddof=1))
        var_pre_t = float(np.var(pre_t, ddof=1))
        var_post_c = float(np.var(post_c, ddof=1))
        var_post_t = float(np.var(post_t, ddof=1))

        se = float(
            np.sqrt(
                var_post_t / len(post_t)
                + var_pre_t / len(pre_t)
                + var_post_c / len(post_c)
                + var_pre_c / len(pre_c)
            )
        )

        # z-test for ATT
        if se > 0:
            z = att / se
            pvalue = float(2 * norm.sf(abs(z)))
            z_crit = float(norm.ppf(1 - self._alpha / 2))
            ci_lower = att - z_crit * se
            ci_upper = att + z_crit * se
        else:
            pvalue = 1.0 if att == 0 else 0.0
            ci_lower = att
            ci_upper = att

        # Parallel trends check: test if pre-period means differ
        pre_trend_diff = mean_pre_t - mean_pre_c
        # t-test on pre-period means
        _, pt_pval = ttest_ind(pre_t, pre_c, equal_var=False)
        parallel_trends_pvalue = float(pt_pval)

        self._result = DiDResult(
            att=att,
            se=se,
            pvalue=pvalue,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            significant=pvalue < self._alpha,
            pre_trend_diff=pre_trend_diff,
            parallel_trends_pvalue=parallel_trends_pvalue,
        )
        return self

    def result(self) -> DiDResult:
        """Return the DiD result.

        Returns
        -------
        DiDResult
            The estimated treatment effect and diagnostics.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called.
        """
        if self._result is None:
            raise RuntimeError(
                format_error(
                    "DifferenceInDifferences must be fitted before calling result().",
                    "call .fit() with pre/post data first.",
                )
            )
        return self._result
