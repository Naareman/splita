"""PostStratification -- Post-experiment stratification for variance reduction.

Computes within-stratum average treatment effects, then combines them
with variance-optimal weights (inverse-variance weighting) for a more
precise estimate of the overall ATE.

References
----------
.. [1] Miratrix, L. W., Sekhon, J. S. & Yu, B. "Adjusting treatment
       effect estimates by post-stratification in randomized experiments."
       JRSS-B, 75(2), 2013.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from splita._types import PostStratResult
from splita._validation import (
    check_array_like,
    check_in_range,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


class PostStratification:
    """Post-experiment stratification for variance reduction.

    Splits control and treatment data by stratum labels, estimates
    within-stratum ATEs, and combines them using inverse-variance
    (variance-optimal) weights.

    Parameters
    ----------
    alpha : float, default 0.05
        Significance level for the two-sided test.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> ctrl = np.concatenate([rng.normal(5, 1, 200), rng.normal(20, 1, 200)])
    >>> trt = np.concatenate([rng.normal(5.5, 1, 200), rng.normal(20.5, 1, 200)])
    >>> strata_c = np.array([0]*200 + [1]*200)
    >>> strata_t = np.array([0]*200 + [1]*200)
    >>> result = PostStratification().fit_transform(ctrl, trt, strata_c, strata_t)
    >>> result.n_strata
    2
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

    @staticmethod
    def _validate_strata(strata: ArrayLike, name: str, expected_len: int) -> np.ndarray:
        """Validate and convert strata labels to a numpy array."""
        if not isinstance(strata, (list, tuple, np.ndarray)):
            raise TypeError(
                format_error(
                    f"`{name}` must be array-like (list, tuple, or ndarray), "
                    f"got type {type(strata).__name__}.",
                )
            )
        arr = np.asarray(strata)
        if arr.ndim != 1:
            raise ValueError(
                format_error(
                    f"`{name}` must be a 1-D array, got {arr.ndim}-D.",
                    hint="pass a flat list or 1-D array of stratum labels.",
                )
            )
        if len(arr) != expected_len:
            raise ValueError(
                format_error(
                    f"`{name}` must have the same length as its data array.",
                    detail=f"expected {expected_len} elements, got {len(arr)}.",
                )
            )
        return arr

    def fit_transform(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
        control_strata: ArrayLike,
        treatment_strata: ArrayLike,
    ) -> PostStratResult:
        """Run post-stratification and return results.

        Parameters
        ----------
        control : array-like
            Control group observations.
        treatment : array-like
            Treatment group observations.
        control_strata : array-like
            Stratum labels for each control observation.
        treatment_strata : array-like
            Stratum labels for each treatment observation.

        Returns
        -------
        PostStratResult
            Frozen dataclass with ATE, SE, CI, and variance reduction.
        """
        ctrl = check_array_like(control, "control", min_length=2)
        trt = check_array_like(treatment, "treatment", min_length=2)
        ctrl_strata = self._validate_strata(control_strata, "control_strata", len(ctrl))
        trt_strata = self._validate_strata(treatment_strata, "treatment_strata", len(trt))

        # Find common strata
        common = sorted(set(ctrl_strata) & set(trt_strata), key=str)
        if not common:
            raise ValueError(
                format_error(
                    "Control and treatment strata must share at least one label.",
                    detail=f"control labels: {sorted(set(ctrl_strata), key=str)}, "
                    f"treatment labels: {sorted(set(trt_strata), key=str)}.",
                    hint="ensure both groups use the same stratum naming.",
                )
            )

        # Compute within-stratum ATEs and variances
        ates = []
        variances = []
        stratum_ns = []

        for label in common:
            y_c = ctrl[ctrl_strata == label]
            y_t = trt[trt_strata == label]

            if len(y_c) < 2 or len(y_t) < 2:
                raise ValueError(
                    format_error(
                        f"Stratum {label!r} must have at least 2 observations per group.",
                        detail=f"control has {len(y_c)}, treatment has {len(y_t)}.",
                    )
                )

            ate_s = float(np.mean(y_t) - np.mean(y_c))
            var_s = float(np.var(y_c, ddof=1) / len(y_c) + np.var(y_t, ddof=1) / len(y_t))

            ates.append(ate_s)
            variances.append(var_s)
            stratum_ns.append(len(y_c) + len(y_t))

        ates = np.array(ates)
        variances = np.array(variances)

        # Inverse-variance (variance-optimal) weights
        inv_var = 1.0 / variances
        opt_weights = inv_var / inv_var.sum()

        # Combined ATE and variance
        ate = float(np.sum(opt_weights * ates))
        var_ate = float(np.sum(opt_weights**2 * variances))
        se = float(np.sqrt(var_ate))

        # z-test
        z = ate / se if se > 0 else 0.0
        pvalue = float(2 * norm.sf(abs(z)))

        z_crit = float(norm.ppf(1 - self._alpha / 2))
        ci_lower = ate - z_crit * se
        ci_upper = ate + z_crit * se

        # Variance reduction vs naive (unstratified)
        naive_var = float(np.var(ctrl, ddof=1) / len(ctrl) + np.var(trt, ddof=1) / len(trt))
        variance_reduction = 1.0 - var_ate / naive_var if naive_var > 0 else 0.0

        return PostStratResult(
            ate=ate,
            se=se,
            pvalue=pvalue,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            significant=pvalue < self._alpha,
            variance_reduction=float(max(0.0, variance_reduction)),
            n_strata=len(common),
        )
