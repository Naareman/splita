"""MultivariateCUPED — Multivariate extension of CUPED.

When multiple pre-experiment covariates are available, multivariate CUPED
uses all of them simultaneously to achieve greater variance reduction than
scalar CUPED with any single covariate.

The adjustment coefficient is a vector:
    ``theta = Cov(Y, X) @ Var(X)^{-1}``

And the adjusted metric is:
    ``Y_adj = Y - X @ theta + mean(X @ theta)``

References
----------
.. [1] Deng, A., Xu, Y., Kohavi, R. & Walker, T.  "Improving the Sensitivity
       of Online Controlled Experiments by Utilizing Pre-Experiment Data."
       WSDM 2013.
.. [2] Poyarkov, A., Drutsa, A., Khalyavin, A., Gusev, G. & Serdyukov, P.
       "Boosted Decision Tree Regression Adjustment for Variance Reduction
       in Online Controlled Experiments."  KDD 2016.
"""

from __future__ import annotations

import warnings

import numpy as np

from splita._validation import (
    check_array_like,
    check_in_range,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


class MultivariateCUPED:
    """Reduce metric variance using multiple pre-experiment covariates.

    Extends scalar CUPED to the multivariate case.  When covariates are
    correlated with the outcome but capture different aspects of
    pre-experiment behaviour, this achieves greater variance reduction
    than using any single covariate.

    Parameters
    ----------
    alpha : float, default 0.05
        Significance level (used only for informational purposes; the
        class does not perform hypothesis testing itself).

    Attributes
    ----------
    theta_ : np.ndarray
        Fitted adjustment coefficient vector (one element per covariate).
    correlation_ : float
        Multiple R (square root of R-squared) between the outcome and
        the linear combination of covariates.
    variance_reduction_ : float
        Fraction of variance explained (R-squared).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n = 500
    >>> X1 = rng.normal(10, 2, size=2 * n)
    >>> X2 = rng.normal(5, 1, size=2 * n)
    >>> ctrl = X1[:n] + 0.5 * X2[:n] + rng.normal(0, 1, n)
    >>> trt = X1[n:] + 0.5 * X2[n:] + 0.3 + rng.normal(0, 1, n)
    >>> X_ctrl = np.column_stack([X1[:n], X2[:n]])
    >>> X_trt = np.column_stack([X1[n:], X2[n:]])
    >>> mcuped = MultivariateCUPED()
    >>> ctrl_adj, trt_adj = mcuped.fit_transform(ctrl, trt, X_ctrl, X_trt)
    >>> mcuped.variance_reduction_ > 0.3
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

        # Fitted attributes (set by fit)
        self.theta_: np.ndarray
        self.correlation_: float
        self.variance_reduction_: float
        self._x_pool_mean: np.ndarray
        self._is_fitted = False

    # ── fit ──────────────────────────────────────────────────────────

    def fit(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
        X_control: ArrayLike,
        X_treatment: ArrayLike,
    ) -> MultivariateCUPED:
        """Learn the multivariate adjustment coefficients from data.

        Parameters
        ----------
        control : array-like
            Post-experiment observations for the control group.
        treatment : array-like
            Post-experiment observations for the treatment group.
        X_control : array-like
            Pre-experiment covariate matrix for the control group.
            Shape ``(n_control, p)`` where ``p`` is the number of
            covariates.  A 1-D array is treated as a single covariate.
        X_treatment : array-like
            Pre-experiment covariate matrix for the treatment group.
            Shape ``(n_treatment, p)``.

        Returns
        -------
        MultivariateCUPED
            The fitted instance (for method chaining).

        Raises
        ------
        ValueError
            If covariate matrices have mismatched dimensions, zero
            variance, or are singular.
        """
        control_arr = check_array_like(control, "control", min_length=2)
        treatment_arr = check_array_like(treatment, "treatment", min_length=2)

        X_c = self._validate_covariate_matrix(X_control, "X_control", len(control_arr))
        X_t = self._validate_covariate_matrix(X_treatment, "X_treatment", len(treatment_arr))

        if X_c.shape[1] != X_t.shape[1]:
            raise ValueError(
                format_error(
                    "`X_control` and `X_treatment` must have the same number of covariates.",
                    detail=f"X_control has {X_c.shape[1]} columns, "
                    f"X_treatment has {X_t.shape[1]} columns.",
                )
            )

        # Pool data
        y = np.concatenate([control_arr, treatment_arr])
        X = np.vstack([X_c, X_t])

        self._x_pool_mean = np.mean(X, axis=0)

        # Compute theta = Cov(Y, X) @ Var(X)^{-1}
        n = len(y)
        X_centered = X - self._x_pool_mean
        y_centered = y - np.mean(y)

        cov_yx = (X_centered.T @ y_centered) / (n - 1)  # shape (p,)
        cov_xx = (X_centered.T @ X_centered) / (n - 1)  # shape (p, p)

        # Check for singular / near-singular covariance matrix
        cond = np.linalg.cond(cov_xx)
        if cond > 1e12:
            raise ValueError(
                format_error(
                    "Covariate covariance matrix is singular or near-singular.",
                    detail=f"condition number = {cond:.1e}; covariates may be "
                    "perfectly or nearly collinear.",
                    hint="remove redundant covariates or use scalar CUPED.",
                )
            )

        try:
            self.theta_ = np.linalg.solve(cov_xx, cov_yx)
        except np.linalg.LinAlgError:  # pragma: no cover
            raise ValueError(
                format_error(
                    "Covariate covariance matrix is singular.",
                    detail="covariates may be perfectly collinear.",
                    hint="remove redundant covariates or use scalar CUPED.",
                )
            ) from None

        if cond > 1e10:
            warnings.warn(
                f"Covariate covariance matrix is near-singular (condition number "
                f"= {cond:.1e}). Results may be numerically unstable.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Multiple R and variance reduction
        y_pred = X_centered @ self.theta_
        ss_pred = float(np.sum(y_pred**2))
        ss_total = float(np.sum(y_centered**2))

        r_squared = min(ss_pred / ss_total, 1.0) if ss_total > 0 else 0.0

        self.variance_reduction_ = r_squared
        self.correlation_ = float(np.sqrt(r_squared))

        if self.correlation_ < 0.3:
            warnings.warn(
                f"Low multiple correlation ({self.correlation_:.2f}) between metric "
                f"and covariates. Multivariate CUPED will provide minimal "
                f"variance reduction.",
                RuntimeWarning,
                stacklevel=2,
            )

        self._is_fitted = True
        return self

    # ── transform ───────────────────────────────────────────────────

    def transform(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
        X_control: ArrayLike,
        X_treatment: ArrayLike,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the multivariate CUPED adjustment using the fitted theta.

        Parameters
        ----------
        control : array-like
            Post-experiment observations for the control group.
        treatment : array-like
            Post-experiment observations for the treatment group.
        X_control : array-like
            Covariate matrix for the control group.
        X_treatment : array-like
            Covariate matrix for the treatment group.

        Returns
        -------
        tuple of np.ndarray
            ``(control_adjusted, treatment_adjusted)`` -- the adjusted
            observations for each group.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        """
        if not self._is_fitted:
            raise RuntimeError(
                format_error(
                    "MultivariateCUPED must be fitted before calling transform().",
                    detail="theta_ has not been estimated yet.",
                    hint="call fit() or fit_transform() first.",
                )
            )

        control_arr = check_array_like(control, "control", min_length=2)
        treatment_arr = check_array_like(treatment, "treatment", min_length=2)

        X_c = self._validate_covariate_matrix(X_control, "X_control", len(control_arr))
        X_t = self._validate_covariate_matrix(X_treatment, "X_treatment", len(treatment_arr))

        # Y_adj = Y - X @ theta + mean(X @ theta)
        # where mean(X @ theta) uses the pooled X mean from fit
        mean_adjustment = float(self._x_pool_mean @ self.theta_)
        control_adj = control_arr - X_c @ self.theta_ + mean_adjustment
        treatment_adj = treatment_arr - X_t @ self.theta_ + mean_adjustment

        return control_adj, treatment_adj

    # ── fit_transform ───────────────────────────────────────────────

    def fit_transform(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
        X_control: ArrayLike,
        X_treatment: ArrayLike,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit and apply the multivariate CUPED adjustment in one step.

        Convenience method equivalent to calling :meth:`fit` followed
        by :meth:`transform` with the same arguments.

        Parameters
        ----------
        control : array-like
            Post-experiment observations for the control group.
        treatment : array-like
            Post-experiment observations for the treatment group.
        X_control : array-like
            Covariate matrix for the control group.
        X_treatment : array-like
            Covariate matrix for the treatment group.

        Returns
        -------
        tuple of np.ndarray
            ``(control_adjusted, treatment_adjusted)``.
        """
        self.fit(control, treatment, X_control, X_treatment)
        result = self.transform(control, treatment, X_control, X_treatment)

        # Inform user about variance reduction achieved
        from splita._advisory import info

        info(
            f"MultivariateCUPED reduced variance by {self.variance_reduction_:.1%} "
            f"(R = {self.correlation_:.3f}). "
            f"This is equivalent to running the experiment "
            f"{1 / (1 - self.variance_reduction_):.1f}x longer."
        )

        if self.variance_reduction_ < 0.05:
            warnings.warn(
                f"MultivariateCUPED reduced variance by only {self.variance_reduction_:.1%}. "
                f"The covariates are weakly predictive of the outcome. "
                f"Consider using CUPAC with ML features for better reduction, "
                f"or check that the covariates are from the right time period.",
                RuntimeWarning,
                stacklevel=2,
            )

        return result

    # ── private helpers ─────────────────────────────────────────────

    @staticmethod
    def _validate_covariate_matrix(X: ArrayLike, name: str, expected_rows: int) -> np.ndarray:
        """Validate and convert a covariate matrix to 2-D."""
        if not isinstance(X, (list, tuple, np.ndarray)):
            raise TypeError(
                format_error(
                    f"`{name}` must be array-like (list, tuple, or ndarray), "
                    f"got type {type(X).__name__}.",
                )
            )

        arr = np.asarray(X, dtype=np.float64)

        # Promote 1-D to column vector
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        if arr.ndim != 2:
            raise ValueError(
                format_error(
                    f"`{name}` must be a 1-D or 2-D array, got {arr.ndim}-D.",
                    hint="pass a matrix with shape (n_observations, n_covariates).",
                )
            )

        if arr.shape[0] != expected_rows:
            raise ValueError(
                format_error(
                    f"`{name}` must have {expected_rows} rows "
                    f"to match data array length, got {arr.shape[0]}.",
                )
            )

        return arr
