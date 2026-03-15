"""RandomizationValidator -- Check covariate balance between experiment groups.

Computes the standardised mean difference (SMD) for each covariate and
runs a chi-squared omnibus test for overall balance.  An SMD > 0.1 for
any covariate flags potential randomisation failure.

References
----------
.. [1] Austin, P. C. "Balance diagnostics for comparing the distribution
       of baseline covariates between treatment groups in propensity-score
       matched samples." Statistics in Medicine, 28(25), 2009.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from splita._types import RandomizationResult
from splita._validation import (
    check_in_range,
    format_error,
)

_SMD_THRESHOLD = 0.1


class RandomizationValidator:
    """Check covariate balance between control and treatment groups.

    For each covariate, computes the absolute standardised mean
    difference (SMD).  An omnibus chi-squared test checks whether
    the overall imbalance is larger than expected by chance.

    Parameters
    ----------
    alpha : float, default 0.05
        Significance level for the omnibus test.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> ctrl = rng.normal(0, 1, (500, 3))
    >>> trt = rng.normal(0, 1, (500, 3))
    >>> result = RandomizationValidator().validate(ctrl, trt, ["age", "income", "visits"])
    >>> result.balanced
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

    def validate(
        self,
        control_covariates: np.ndarray,
        treatment_covariates: np.ndarray,
        covariate_names: list[str] | None = None,
    ) -> RandomizationResult:
        """Run the balance check and return results.

        Parameters
        ----------
        control_covariates : np.ndarray
            2-D array of shape ``(n_control, p)`` with covariate values.
        treatment_covariates : np.ndarray
            2-D array of shape ``(n_treatment, p)`` with covariate values.
        covariate_names : list of str, optional
            Names for each covariate column.  Defaults to ``x_0``, ``x_1``, ...

        Returns
        -------
        RandomizationResult
            Frozen dataclass with per-covariate SMDs, omnibus p-value, and
            list of imbalanced covariates.

        Raises
        ------
        ValueError
            If arrays have incompatible shapes or fewer than 2 rows.
        """
        ctrl = np.asarray(control_covariates, dtype=float)
        trt = np.asarray(treatment_covariates, dtype=float)

        if ctrl.ndim == 1:
            ctrl = ctrl.reshape(-1, 1)
        if trt.ndim == 1:
            trt = trt.reshape(-1, 1)

        if ctrl.ndim != 2 or trt.ndim != 2:  # pragma: no cover
            raise ValueError(
                format_error(
                    "Covariates must be 1-D or 2-D arrays.",
                    detail=f"control has {ctrl.ndim}-D, treatment has {trt.ndim}-D.",
                )
            )

        if ctrl.shape[1] != trt.shape[1]:
            raise ValueError(
                format_error(
                    "Control and treatment must have the same number of covariates.",
                    detail=f"control has {ctrl.shape[1]} columns, "
                    f"treatment has {trt.shape[1]} columns.",
                )
            )

        if ctrl.shape[0] < 2:
            raise ValueError(
                format_error(
                    "Control must have at least 2 observations.",
                    detail=f"got {ctrl.shape[0]}.",
                )
            )
        if trt.shape[0] < 2:  # pragma: no cover
            raise ValueError(
                format_error(
                    "Treatment must have at least 2 observations.",
                    detail=f"got {trt.shape[0]}.",
                )
            )

        p = ctrl.shape[1]
        if covariate_names is None:
            covariate_names = [f"x_{i}" for i in range(p)]
        elif len(covariate_names) != p:
            raise ValueError(
                format_error(
                    "`covariate_names` must match the number of covariate columns.",
                    detail=f"expected {p} names, got {len(covariate_names)}.",
                )
            )

        # Per-covariate SMD
        smd_list = []
        imbalanced = []
        chi2_stat = 0.0

        for j in range(p):
            c_col = ctrl[:, j]
            t_col = trt[:, j]

            mean_c = float(np.mean(c_col))
            mean_t = float(np.mean(t_col))
            var_c = float(np.var(c_col, ddof=1))
            var_t = float(np.var(t_col, ddof=1))

            pooled_std = np.sqrt((var_c + var_t) / 2.0)
            smd = abs(mean_t - mean_c) / pooled_std if pooled_std > 0 else 0.0

            smd_list.append({"name": covariate_names[j], "smd": float(smd)})

            if smd > _SMD_THRESHOLD:
                imbalanced.append(covariate_names[j])

            # Accumulate for omnibus test (squared t-statistics)
            se = np.sqrt(var_c / ctrl.shape[0] + var_t / trt.shape[0])
            if se > 0:
                t_stat = (mean_t - mean_c) / se
                chi2_stat += t_stat**2

        # Omnibus chi-squared test
        omnibus_pvalue = float(1.0 - chi2.cdf(chi2_stat, df=p))

        max_smd = max(d["smd"] for d in smd_list) if smd_list else 0.0

        balanced = len(imbalanced) == 0 and omnibus_pvalue >= self._alpha

        return RandomizationResult(
            balanced=balanced,
            smd_per_covariate=smd_list,
            max_smd=float(max_smd),
            omnibus_pvalue=omnibus_pvalue,
            imbalanced_covariates=imbalanced,
        )
