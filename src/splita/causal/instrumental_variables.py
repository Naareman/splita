"""Instrumental Variables estimator using Two-Stage Least Squares (2SLS).

Estimates the Local Average Treatment Effect (LATE) for compliers
when treatment assignment is imperfect (non-compliance).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from splita._types import IVResult
from splita._validation import check_array_like, check_in_range, format_error

ArrayLike = list | tuple | np.ndarray


class InstrumentalVariables:
    """Instrumental Variables estimator via Two-Stage Least Squares.

    Estimates the LATE (Local Average Treatment Effect) for compliers
    using an instrumental variable that affects treatment assignment
    but not the outcome directly.

    Parameters
    ----------
    alpha : float, default 0.05
        Significance level for inference.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n = 1000
    >>> z = rng.binomial(1, 0.5, n).astype(float)
    >>> t = (z + rng.normal(0, 0.3, n) > 0.5).astype(float)
    >>> y = 2.0 * t + rng.normal(0, 1, n)
    >>> iv = InstrumentalVariables()
    >>> r = iv.fit(y, t, z)
    >>> abs(r.late - 2.0) < 1.5
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

    def fit(
        self,
        outcome: ArrayLike,
        treatment: ArrayLike,
        instrument: ArrayLike,
        covariates: np.ndarray | None = None,
    ) -> IVResult:
        """Fit the 2SLS model.

        Parameters
        ----------
        outcome : array-like
            Outcome variable (Y).
        treatment : array-like
            Treatment variable (T), may be endogenous.
        instrument : array-like
            Instrumental variable (Z).
        covariates : np.ndarray or None, optional
            Additional exogenous covariates (n x p matrix).

        Returns
        -------
        IVResult
            LATE estimate with inference and instrument diagnostics.

        Raises
        ------
        TypeError
            If inputs cannot be converted to numeric arrays.
        ValueError
            If arrays have mismatched lengths or fewer than 5 elements.
        """
        y = check_array_like(outcome, "outcome", min_length=5)
        t = check_array_like(treatment, "treatment", min_length=5)
        z = check_array_like(instrument, "instrument", min_length=5)

        n = len(y)
        if len(t) != n:
            raise ValueError(
                format_error(
                    "`outcome` and `treatment` must have the same length.",
                    f"outcome has {n} elements, treatment has {len(t)} elements.",
                )
            )
        if len(z) != n:  # pragma: no cover
            raise ValueError(
                format_error(
                    "`outcome` and `instrument` must have the same length.",
                    f"outcome has {n} elements, instrument has {len(z)} elements.",
                )
            )

        # Build first-stage regressor matrix
        if covariates is not None:
            if covariates.ndim == 1:
                covariates = covariates.reshape(-1, 1)
            if covariates.shape[0] != n:  # pragma: no cover
                raise ValueError(
                    format_error(
                        "`covariates` must have the same number of rows as outcome.",
                        f"outcome has {n} rows, covariates has {covariates.shape[0]} rows.",
                    )
                )
            # First stage: T ~ Z + covariates + intercept
            X1 = np.column_stack([np.ones(n), z, covariates])
        else:
            X1 = np.column_stack([np.ones(n), z])

        # Stage 1: regress T on Z (and covariates)
        beta1, _residuals1, _rank1, _ = np.linalg.lstsq(X1, t, rcond=None)
        t_hat = X1 @ beta1

        # First-stage F-statistic for instrument strength
        # Compare full model (with Z) vs restricted model (without Z)
        if covariates is not None:
            X1_restricted = np.column_stack([np.ones(n), covariates])
        else:
            X1_restricted = np.ones((n, 1))

        beta1_r, _, _, _ = np.linalg.lstsq(X1_restricted, t, rcond=None)
        t_hat_restricted = X1_restricted @ beta1_r

        ssr_restricted = float(np.sum((t - t_hat_restricted) ** 2))
        ssr_full = float(np.sum((t - t_hat) ** 2))

        df_num = 1  # one instrument
        df_den = n - X1.shape[1]

        if ssr_full > 0 and df_den > 0:
            first_stage_f = float(((ssr_restricted - ssr_full) / df_num) / (ssr_full / df_den))
        else:
            first_stage_f = 0.0

        weak_instrument = first_stage_f < 10.0

        # Stage 2: regress Y on T_hat (and covariates)
        if covariates is not None:
            X2 = np.column_stack([np.ones(n), t_hat, covariates])
        else:
            X2 = np.column_stack([np.ones(n), t_hat])

        beta2, _, _, _ = np.linalg.lstsq(X2, y, rcond=None)
        late = float(beta2[1])  # coefficient on T_hat

        # Compute SE using actual residuals (not fitted residuals)
        y_pred = X2 @ beta2
        residuals2 = y - y_pred
        sigma2 = float(np.sum(residuals2**2)) / max(n - X2.shape[1], 1)

        # Variance of beta2
        try:
            xtx_inv = np.linalg.inv(X2.T @ X2)
            se = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
        except np.linalg.LinAlgError:  # pragma: no cover
            se = float("inf")

        if se > 0 and se != float("inf"):
            z_stat = late / se
            pvalue = float(2.0 * norm.sf(abs(z_stat)))
            z_crit = float(norm.ppf(1 - self._alpha / 2))
            ci_lower = late - z_crit * se
            ci_upper = late + z_crit * se
        else:  # pragma: no cover
            pvalue = 1.0 if late == 0 else 0.0
            ci_lower = late
            ci_upper = late

        return IVResult(
            late=late,
            se=se,
            pvalue=pvalue,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            significant=pvalue < self._alpha,
            first_stage_f=first_stage_f,
            weak_instrument=weak_instrument,
        )
