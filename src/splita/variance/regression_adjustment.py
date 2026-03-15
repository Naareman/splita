"""Lin's Regression Adjustment for A/B tests (Lin, 2013).

OLS regression adjustment that is provably at least as efficient as CUPED.
Uses treatment indicator + demeaned covariates + treatment*covariate
interactions with HC2 robust standard errors.

References
----------
.. [1] Lin, W.  "Agnostic notes on regression adjustments to experimental
       data: Reexamining Freedman's critique."  Annals of Applied Statistics,
       7(1):295-318, 2013.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from splita._types import RegressionAdjustmentResult
from splita._validation import (
    check_array_like,
    check_in_range,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


class RegressionAdjustment:
    """Lin's regression adjustment for estimating treatment effects.

    Fits a fully-interacted OLS model with HC2 robust standard errors:

        Y = intercept + tau*T + beta*X_centered + gamma*(T * X_centered) + eps

    where ``X_centered = X - mean(X)`` over the pooled sample.

    This is provably at least as efficient as CUPED and often strictly
    more efficient when the treatment effect varies with covariates.

    Parameters
    ----------
    alpha : float, default 0.05
        Significance level for confidence intervals and hypothesis tests.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> pre = rng.normal(10, 2, size=200)
    >>> ctrl = pre[:100] + rng.normal(0, 1, 100)
    >>> trt = pre[100:] + 0.5 + rng.normal(0, 1, 100)
    >>> ra = RegressionAdjustment()
    >>> result = ra.fit_transform(ctrl, trt, pre[:100], pre[100:])
    >>> result.variance_reduction > 0.0
    True
    """

    def __init__(self, *, alpha: float = 0.05) -> None:
        check_in_range(
            alpha,
            "alpha",
            0.0,
            1.0,
            hint="typical values are 0.05, 0.01, or 0.10.",
        )
        self.alpha = alpha

    def fit_transform(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
        X_control: ArrayLike,
        X_treatment: ArrayLike,
    ) -> RegressionAdjustmentResult:
        """Estimate the average treatment effect using regression adjustment.

        Parameters
        ----------
        control : array-like
            Post-experiment observations for the control group.
        treatment : array-like
            Post-experiment observations for the treatment group.
        X_control : array-like
            Covariate matrix for the control group.  Can be 1-D (single
            covariate) or 2-D (multiple covariates, shape ``n_control x p``).
        X_treatment : array-like
            Covariate matrix for the treatment group.  Same number of
            columns as *X_control*.

        Returns
        -------
        RegressionAdjustmentResult
            Frozen dataclass with ATE, SE, p-value, CI, and diagnostics.

        Raises
        ------
        ValueError
            If array lengths are mismatched or inputs are invalid.
        """
        # ── validate outcome arrays ─────────────────────────────────
        ctrl = check_array_like(control, "control", min_length=2)
        trt = check_array_like(treatment, "treatment", min_length=2)
        n_ctrl = len(ctrl)
        n_trt = len(trt)

        # ── validate covariate arrays ───────────────────────────────
        Xc = self._validate_covariates(X_control, "X_control", n_ctrl)
        Xt = self._validate_covariates(X_treatment, "X_treatment", n_trt)

        if Xc.shape[1] != Xt.shape[1]:
            raise ValueError(
                format_error(
                    "`X_control` and `X_treatment` must have the same number "
                    f"of covariates, got {Xc.shape[1]} and {Xt.shape[1]}.",
                    hint="ensure both covariate matrices have the same columns.",
                )
            )

        # ── Step 1: pool data ───────────────────────────────────────
        Y = np.concatenate([ctrl, trt])
        T = np.concatenate([np.zeros(n_ctrl), np.ones(n_trt)])
        X = np.vstack([Xc, Xt])
        n = len(Y)

        # ── Step 2: demean covariates ───────────────────────────────
        X_mean = X.mean(axis=0)
        X_centered = X - X_mean

        # ── Step 3: build design matrix ─────────────────────────────
        # [intercept, T, X_centered, T * X_centered]
        intercept = np.ones((n, 1))
        T_col = T.reshape(-1, 1)
        TX = T_col * X_centered  # interaction terms

        Z = np.hstack([intercept, T_col, X_centered, TX])
        # T coefficient is at index 1
        t_idx = 1

        # ── Step 4: OLS ─────────────────────────────────────────────
        # beta = (Z'Z)^{-1} Z'Y
        ZtZ = Z.T @ Z
        ZtY = Z.T @ Y
        try:
            ZtZ_inv = np.linalg.inv(ZtZ)
        except np.linalg.LinAlgError:
            raise ValueError(
                format_error(
                    "Design matrix is singular — can't estimate regression.",
                    detail="covariates may be perfectly collinear.",
                    hint="remove redundant covariates.",
                )
            ) from None

        beta = ZtZ_inv @ ZtY
        ate = float(beta[t_idx])

        # ── Step 5: residuals and R-squared ─────────────────────────
        Y_hat = Z @ beta
        residuals = Y - Y_hat

        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((Y - np.mean(Y)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # ── Step 6: HC2 robust standard errors ──────────────────────
        # h_ii = diag(Z (Z'Z)^{-1} Z')
        # Computed efficiently without forming the full hat matrix
        H_diag = np.sum((Z @ ZtZ_inv) * Z, axis=1)
        # Clamp leverage to avoid division by zero
        H_diag = np.minimum(H_diag, 1.0 - 1e-12)

        # HC2 weights: e_i^2 / (1 - h_ii)
        hc2_weights = residuals**2 / (1.0 - H_diag)

        # Sandwich: (Z'Z)^{-1} Z' diag(w) Z (Z'Z)^{-1}
        # = (Z'Z)^{-1} (sum_i w_i z_i z_i') (Z'Z)^{-1}
        meat = (Z * hc2_weights.reshape(-1, 1)).T @ Z
        sandwich = ZtZ_inv @ meat @ ZtZ_inv

        se = float(np.sqrt(sandwich[t_idx, t_idx]))

        # ── Step 7: inference (t-distribution) ──────────────────────
        df = n - Z.shape[1]  # degrees of freedom
        if df < 1:
            raise ValueError(
                format_error(
                    "Not enough observations for the number of covariates.",
                    detail=f"n={n}, parameters={Z.shape[1]}, df={df}.",
                    hint="reduce the number of covariates or increase sample size.",
                )
            )

        t_stat = ate / se if se > 0 else 0.0
        pvalue = float(2.0 * stats.t.sf(abs(t_stat), df=df))
        t_crit = float(stats.t.ppf(1.0 - self.alpha / 2.0, df=df))
        ci_lower = ate - t_crit * se
        ci_upper = ate + t_crit * se

        # ── variance reduction metric ───────────────────────────────
        # Compare against unadjusted difference-in-means SE
        var_ctrl = float(np.var(ctrl, ddof=1))
        var_trt = float(np.var(trt, ddof=1))
        se_unadjusted = float(np.sqrt(var_ctrl / n_ctrl + var_trt / n_trt))
        variance_reduction = 1.0 - (se**2) / (se_unadjusted**2) if se_unadjusted > 0 else 0.0

        return RegressionAdjustmentResult(
            ate=ate,
            se=se,
            pvalue=pvalue,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            significant=pvalue < self.alpha,
            alpha=self.alpha,
            variance_reduction=variance_reduction,
            r_squared=r_squared,
        )

    # ── private helpers ─────────────────────────────────────────────

    @staticmethod
    def _validate_covariates(
        X: ArrayLike,
        name: str,
        expected_n: int,
    ) -> np.ndarray:
        """Validate and reshape covariate input to 2-D."""
        if isinstance(X, (list, tuple)):
            X = np.asarray(X, dtype="float64")
        elif not isinstance(X, np.ndarray):
            raise TypeError(
                format_error(
                    f"`{name}` must be array-like (list, tuple, or ndarray), "
                    f"got type {type(X).__name__}.",
                )
            )
        else:
            X = np.asarray(X, dtype="float64")

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError(
                format_error(
                    f"`{name}` must be 1-D or 2-D, got {X.ndim}-D array with shape {X.shape}.",
                )
            )

        if X.shape[0] != expected_n:
            raise ValueError(
                format_error(
                    f"`{name}` must have {expected_n} rows to match the "
                    f"outcome array, got {X.shape[0]}.",
                )
            )

        return X
