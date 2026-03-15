"""Mediation analysis — decompose treatment effects via mediators.

Implements the Baron & Kenny (1986) sequential regression approach
with the Sobel test for the indirect effect (ACME), following the
framework of Imai et al. (2010).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from splita._types import MediationResult
from splita._validation import (
    check_array_like,
    check_in_range,
    check_same_length,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


def _ols_coef(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute OLS coefficients and their standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n, p) with intercept column already included.
    y : np.ndarray
        Response vector (n,).

    Returns
    -------
    coefs : np.ndarray
        Coefficient estimates (p,).
    se : np.ndarray
        Standard errors (p,).
    """
    n, p = X.shape
    # (X'X)^-1 X'y
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    coefs = XtX_inv @ X.T @ y

    # Residual variance
    residuals = y - X @ coefs
    sigma2 = float(np.sum(residuals**2)) / max(n - p, 1)

    # Standard errors
    se = np.sqrt(np.diag(XtX_inv) * sigma2)
    return coefs, se


class MediationAnalysis:
    """Causal mediation analysis using sequential regression.

    Decomposes the total treatment effect into direct and indirect
    (mediated) effects following Baron & Kenny (1986). The indirect
    effect (ACME) is tested using the Sobel test.

    Parameters
    ----------
    alpha : float, default 0.05
        Significance level for the Sobel test.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n = 200
    >>> treatment = rng.binomial(1, 0.5, n).astype(float)
    >>> mediator = 0.5 * treatment + rng.normal(0, 1, n)
    >>> outcome = 0.3 * treatment + 0.6 * mediator + rng.normal(0, 1, n)
    >>> ma = MediationAnalysis()
    >>> ma.fit(outcome, treatment, mediator)  # doctest: +ELLIPSIS
    <splita.causal.mediation.MediationAnalysis object at ...>
    >>> r = ma.result()
    >>> r.indirect_effect > 0
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
        self._result: MediationResult | None = None

    def fit(
        self,
        outcome: ArrayLike,
        treatment: ArrayLike,
        mediator: ArrayLike,
        covariates: ArrayLike | None = None,
    ) -> MediationAnalysis:
        """Fit the mediation model using sequential OLS regressions.

        Step 1: Regress mediator on treatment (and covariates) to get the
        *a* path coefficient.

        Step 2: Regress outcome on treatment + mediator (and covariates)
        to get the direct effect (*c'*) and *b* path coefficient.

        Parameters
        ----------
        outcome : array-like
            Outcome variable (Y).
        treatment : array-like
            Treatment indicator (T), typically binary.
        mediator : array-like
            Mediator variable (M).
        covariates : array-like or None, default None
            Optional covariate matrix (n, k). Each column is a covariate.

        Returns
        -------
        MediationAnalysis
            The fitted estimator (self).

        Raises
        ------
        TypeError
            If inputs cannot be converted to numeric arrays.
        ValueError
            If arrays have fewer than 5 elements or mismatched lengths.
        """
        y = check_array_like(outcome, "outcome", min_length=5)
        t = check_array_like(treatment, "treatment", min_length=5)
        m = check_array_like(mediator, "mediator", min_length=5)

        check_same_length(y, t, "outcome", "treatment")
        check_same_length(y, m, "outcome", "mediator")

        n = len(y)
        ones = np.ones((n, 1))

        # Build covariate block
        if covariates is not None:
            cov = np.asarray(covariates, dtype="float64")
            if cov.ndim == 1:
                cov = cov.reshape(-1, 1)
            if cov.shape[0] != n:
                raise ValueError(
                    format_error(
                        "`covariates` must have the same number of rows as other inputs.",
                        f"expected {n} rows, got {cov.shape[0]}.",
                    )
                )
        else:
            cov = np.empty((n, 0))

        # Step 1: M = a0 + a*T + covariates
        X1 = np.column_stack([ones, t.reshape(-1, 1), cov])
        coefs1, se1 = _ols_coef(X1, m)
        a = float(coefs1[1])  # T -> M coefficient
        se_a = float(se1[1])

        # Step 2: Y = b0 + c'*T + b*M + covariates
        X2 = np.column_stack([ones, t.reshape(-1, 1), m.reshape(-1, 1), cov])
        coefs2, se2 = _ols_coef(X2, y)
        c_prime = float(coefs2[1])  # direct effect (T -> Y controlling M)
        b = float(coefs2[2])  # M -> Y coefficient
        se_b = float(se2[2])

        # Total effect: regress Y on T alone
        X_total = np.column_stack([ones, t.reshape(-1, 1), cov])
        coefs_total, _ = _ols_coef(X_total, y)
        total_effect = float(coefs_total[1])

        # ACME (indirect effect) = a * b
        indirect_effect = a * b

        # Sobel test: SE(a*b) = sqrt(a^2 * se_b^2 + b^2 * se_a^2)
        se_indirect = float(np.sqrt(a**2 * se_b**2 + b**2 * se_a**2))

        if se_indirect > 0:
            z_sobel = indirect_effect / se_indirect
            acme_pvalue = float(2 * norm.sf(abs(z_sobel)))
        else:
            acme_pvalue = 1.0

        # CI for ACME
        z_crit = float(norm.ppf(1 - self._alpha / 2))
        acme_ci = (
            indirect_effect - z_crit * se_indirect,
            indirect_effect + z_crit * se_indirect,
        )

        # Proportion mediated
        proportion_mediated = indirect_effect / total_effect if abs(total_effect) > 1e-10 else 0.0

        self._result = MediationResult(
            total_effect=total_effect,
            direct_effect=c_prime,
            indirect_effect=indirect_effect,
            proportion_mediated=float(proportion_mediated),
            acme_pvalue=acme_pvalue,
            acme_ci=acme_ci,
            a_path=a,
            b_path=b,
            n=n,
        )
        return self

    def result(self) -> MediationResult:
        """Return the mediation analysis result.

        Returns
        -------
        MediationResult
            Decomposition of the treatment effect.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called.
        """
        if self._result is None:
            raise RuntimeError(
                format_error(
                    "MediationAnalysis must be fitted before calling result().",
                    "call .fit() with outcome, treatment, and mediator data first.",
                )
            )
        return self._result
