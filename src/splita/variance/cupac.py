"""CUPAC — Control Using Predictions As Covariates.

Extends CUPED by using an ML model to predict the outcome and using those
predictions as the CUPED covariate.  Valuable for new users (no historical
data) or when a simple linear covariate is insufficient.  Used at DoorDash
and Meta.

References
----------
.. [1] Li, L. & Lu, X.  "CUPED on Machine Learning."  2019.
.. [2] Uber Engineering.  "Under the Hood of Uber's Experimentation
       Platform."  2018.
"""

from __future__ import annotations

import warnings

import numpy as np

from splita._validation import (
    check_array_like,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


def _check_sklearn() -> tuple:
    """Lazily import sklearn components, raising a clear error if missing.

    Returns
    -------
    tuple
        ``(Ridge, KFold, clone)`` classes from scikit-learn.

    Raises
    ------
    ImportError
        If scikit-learn is not installed.
    """
    try:
        from sklearn.base import clone
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import KFold
    except ImportError:  # pragma: no cover
        raise ImportError(
            "CUPAC requires scikit-learn. Install it with: pip install splita[ml]"
        ) from None
    return Ridge, KFold, clone


def _validate_features(
    X_control: np.ndarray,
    X_treatment: np.ndarray,
    control_arr: np.ndarray,
    treatment_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate feature matrices: shape, column match, row match, finiteness.

    Returns the validated arrays cast to float64.
    """
    X_control = np.asarray(X_control, dtype="float64")
    X_treatment = np.asarray(X_treatment, dtype="float64")

    if X_control.ndim != 2:
        raise ValueError(
            format_error(
                f"`X_control` must be a 2-D array, got {X_control.ndim}-D "
                f"with shape {X_control.shape}.",
                hint="pass a 2-D feature matrix with shape (n_samples, n_features).",
            )
        )

    if X_treatment.ndim != 2:
        raise ValueError(
            format_error(
                f"`X_treatment` must be a 2-D array, got {X_treatment.ndim}-D "
                f"with shape {X_treatment.shape}.",
                hint="pass a 2-D feature matrix with shape (n_samples, n_features).",
            )
        )

    if X_control.shape[1] != X_treatment.shape[1]:
        raise ValueError(
            format_error(
                "`X_control` and `X_treatment` must have the same number of columns.",
                detail=f"X_control has {X_control.shape[1]} columns, "
                f"X_treatment has {X_treatment.shape[1]} columns.",
                hint="ensure both feature matrices have the same features.",
            )
        )

    if X_control.shape[0] != len(control_arr):
        raise ValueError(
            format_error(
                "`X_control` and `control` must have the same number of rows.",
                detail=f"X_control has {X_control.shape[0]} rows, "
                f"control has {len(control_arr)} elements.",
                hint="each row of X_control should correspond to one control observation.",
            )
        )

    if X_treatment.shape[0] != len(treatment_arr):
        raise ValueError(
            format_error(
                "`X_treatment` and `treatment` must have the same number of rows.",
                detail=f"X_treatment has {X_treatment.shape[0]} rows, "
                f"treatment has {len(treatment_arr)} elements.",
                hint="each row of X_treatment should correspond to one treatment observation.",
            )
        )

    # Check for NaN / Inf values
    if not np.all(np.isfinite(X_control)):
        raise ValueError(
            format_error(
                "`X_control` contains NaN or infinite values.",
                hint="remove or impute non-finite values before passing to CUPAC.",
            )
        )

    if not np.all(np.isfinite(X_treatment)):
        raise ValueError(
            format_error(
                "`X_treatment` contains NaN or infinite values.",
                hint="remove or impute non-finite values before passing to CUPAC.",
            )
        )

    return X_control, X_treatment


class CUPAC:
    """Reduce metric variance using ML-predicted covariates.

    CUPAC (Control Using Predictions As Covariates) trains an ML model to
    predict the outcome from features, then uses the out-of-fold predictions
    as a CUPED-style covariate.  This is especially powerful when:

    - No pre-experiment data exists (new users).
    - The relationship between features and the metric is non-linear.
    - Multiple features are available.

    Parameters
    ----------
    estimator : object or None, default None
        Any sklearn-compatible estimator with ``fit()`` and ``predict()``
        methods.  ``None`` defaults to ``Ridge(alpha=1.0)``.
    cv : int, default 5
        Number of cross-validation folds for out-of-fold predictions.
    random_state : int, np.random.Generator, or None, default None
        Seed for reproducible CV splits.

    Attributes
    ----------
    theta_ : float
        CUPED adjustment coefficient (set after :meth:`fit_transform`).
    correlation_ : float
        Pearson correlation between Y and Y_hat.
    variance_reduction_ : float
        R² = correlation² — fraction of variance explained.
    cv_r2_ : float
        Cross-validated R² of the estimator.

    Notes
    -----
    ``fit_transform()`` uses out-of-fold predictions for the adjustment,
    which avoids overfitting and is the recommended approach for a single
    dataset.

    ``fit()`` + ``transform()`` uses in-sample predictions from the fitted
    estimator.  On the data used in ``fit()``, this is slightly less accurate
    than ``fit_transform()`` because the predictions are in-sample rather
    than out-of-fold.  However, ``fit()`` + ``transform()`` allows applying
    the fitted model to *new* data (e.g., a second experiment).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n = 500
    >>> X_ctrl = rng.normal(0, 1, size=(n, 3))
    >>> X_trt = rng.normal(0, 1, size=(n, 3))
    >>> ctrl = X_ctrl @ [1, 2, 0.5] + rng.normal(0, 1, n)
    >>> trt = X_trt @ [1, 2, 0.5] + 0.5 + rng.normal(0, 1, n)
    >>> cupac = CUPAC(random_state=42)
    >>> ctrl_adj, trt_adj = cupac.fit_transform(ctrl, trt, X_ctrl, X_trt)
    >>> cupac.variance_reduction_ > 0.3
    True
    """

    def __init__(
        self,
        *,
        estimator: object | None = None,
        cv: int = 5,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        if cv < 2:
            raise ValueError(
                format_error(
                    f"`cv` must be >= 2, got {cv}.",
                    detail="cross-validation requires at least 2 folds.",
                    hint="use cv=5 (default) or cv=10 for more stable estimates.",
                )
            )

        if cv > 50:
            raise ValueError(
                format_error(
                    f"`cv` must be <= 50, got {cv}.",
                    detail="very high fold counts are computationally expensive "
                    "and rarely improve estimates.",
                    hint="use cv=5 (default) or cv=10; values of 5-10 are typical.",
                )
            )

        if estimator is not None and not (
            hasattr(estimator, "fit") and hasattr(estimator, "predict")
        ):
            raise ValueError(
                format_error(
                    "`estimator` must have fit() and predict() methods.",
                    detail=f"got {type(estimator).__name__} which is missing "
                    f"{'fit()' if not hasattr(estimator, 'fit') else 'predict()'}.",
                    hint="pass an sklearn-compatible estimator such as "
                    "Ridge, GradientBoostingRegressor, etc.",
                )
            )

        self.estimator = estimator
        self.cv = cv
        self.random_state = random_state
        self._is_fitted = False

        # Fitted attributes (set by fit / fit_transform)
        self.theta_: float
        self.correlation_: float
        self.variance_reduction_: float
        self.cv_r2_: float

    # ── internal helpers ────────────────────────────────────────────

    def _resolve_estimator(self, Ridge):
        """Return the user-supplied estimator or a default Ridge."""
        estimator = self.estimator
        if estimator is None:
            estimator = Ridge(alpha=1.0)
        return estimator

    def _resolve_rng_seed(self):
        """Convert random_state to an integer seed for KFold."""
        rng_seed = self.random_state
        if isinstance(rng_seed, np.random.Generator):
            rng_seed = int(rng_seed.integers(0, 2**31))
        return rng_seed

    @staticmethod
    def _compute_cuped_stats(Y, Y_hat):
        """Compute theta, correlation, variance_reduction from Y and Y_hat.

        Returns (theta, correlation, variance_reduction, y_hat_mean, degenerate).
        If degenerate is True, the model predicts a constant.
        """
        var_y_hat = float(np.var(Y_hat, ddof=1))
        y_hat_mean = float(np.mean(Y_hat))

        if var_y_hat == 0.0:
            return 0.0, 0.0, 0.0, y_hat_mean, True

        cov_y_yhat = float(np.cov(Y, Y_hat, ddof=1)[0, 1])
        theta = cov_y_yhat / var_y_hat

        corr_matrix = np.corrcoef(Y, Y_hat)
        correlation = float(corr_matrix[0, 1])
        variance_reduction = correlation**2

        return theta, correlation, variance_reduction, y_hat_mean, False

    # ── fit ─────────────────────────────────────────────────────────

    def fit(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
        X_control: np.ndarray,
        X_treatment: np.ndarray,
    ) -> CUPAC:
        """Fit the ML model using K-fold CV and compute adjustment parameters.

        Pools control and treatment data, runs K-fold cross-validation to
        compute ``theta_``, ``correlation_``, ``variance_reduction_``, and
        ``cv_r2_``.  Then re-fits the estimator on *all* pooled data so that
        :meth:`transform` can predict on new data.

        Parameters
        ----------
        control : array-like
            Post-experiment observations for the control group.
        treatment : array-like
            Post-experiment observations for the treatment group.
        X_control : np.ndarray
            Feature matrix for the control group, shape ``(n_ctrl, n_features)``.
        X_treatment : np.ndarray
            Feature matrix for the treatment group, shape ``(n_trt, n_features)``.

        Returns
        -------
        self
            The fitted CUPAC instance.

        Notes
        -----
        The fitted estimator uses in-sample predictions when later passed to
        :meth:`transform`.  For the most accurate single-dataset adjustment,
        prefer :meth:`fit_transform`, which uses out-of-fold predictions.
        """
        Ridge, KFold, clone = _check_sklearn()

        # Validate
        control_arr = check_array_like(control, "control", min_length=2)
        treatment_arr = check_array_like(treatment, "treatment", min_length=2)
        X_control, X_treatment = _validate_features(
            X_control,
            X_treatment,
            control_arr,
            treatment_arr,
        )

        # Pool
        Y = np.concatenate([control_arr, treatment_arr])
        X = np.concatenate([X_control, X_treatment], axis=0)
        n_total = len(Y)

        estimator = self._resolve_estimator(Ridge)
        rng_seed = self._resolve_rng_seed()

        # K-fold CV for statistics
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=rng_seed)
        Y_hat = np.empty(n_total, dtype="float64")

        ss_res = 0.0
        ss_tot = 0.0
        y_mean_global = float(np.mean(Y))

        for train_idx, val_idx in kf.split(X):
            fold_model = clone(estimator)
            fold_model.fit(X[train_idx], Y[train_idx])
            preds = fold_model.predict(X[val_idx])
            Y_hat[val_idx] = preds

            ss_res += np.sum((Y[val_idx] - preds) ** 2)
            ss_tot += np.sum((Y[val_idx] - y_mean_global) ** 2)

        self.cv_r2_ = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        if self.cv_r2_ < 0.1:
            warnings.warn(
                f"ML model R² is low ({self.cv_r2_:.2f}) on cross-validation. "
                f"CUPAC will provide minimal variance reduction.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Compute CUPED stats from OOF predictions
        theta, correlation, variance_reduction, _y_hat_mean, _ = self._compute_cuped_stats(Y, Y_hat)
        self.theta_ = theta
        self.correlation_ = correlation
        self.variance_reduction_ = variance_reduction

        # Re-fit on ALL data for use in transform
        self._fitted_estimator_ = clone(estimator)
        self._fitted_estimator_.fit(X, Y)
        self._mean_y_hat_ = float(np.mean(self._fitted_estimator_.predict(X)))

        self._is_fitted = True
        return self

    # ── transform ───────────────────────────────────────────────────

    def transform(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
        X_control: np.ndarray,
        X_treatment: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply CUPAC adjustment using the previously fitted estimator.

        Uses the estimator fitted by :meth:`fit` to predict Y_hat on the
        new feature matrices, then applies the CUPED adjustment:
        ``Y_adj = Y - theta * (Y_hat - mean_Y_hat_from_fit)``.

        Parameters
        ----------
        control : array-like
            Post-experiment observations for the control group.
        treatment : array-like
            Post-experiment observations for the treatment group.
        X_control : np.ndarray
            Feature matrix for the control group, shape ``(n_ctrl, n_features)``.
        X_treatment : np.ndarray
            Feature matrix for the treatment group, shape ``(n_trt, n_features)``.

        Returns
        -------
        tuple of np.ndarray
            ``(control_adjusted, treatment_adjusted)`` — the CUPAC-adjusted
            observations for each group.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if not self._is_fitted:
            raise RuntimeError(
                format_error(
                    "CUPAC has not been fitted yet.",
                    hint="call fit() or fit_transform() before transform().",
                )
            )

        control_arr = check_array_like(control, "control", min_length=2)
        treatment_arr = check_array_like(treatment, "treatment", min_length=2)
        X_control, X_treatment = _validate_features(
            X_control,
            X_treatment,
            control_arr,
            treatment_arr,
        )

        # Predict using the fitted estimator
        Y_hat_ctrl = self._fitted_estimator_.predict(X_control)
        Y_hat_trt = self._fitted_estimator_.predict(X_treatment)

        # Apply CUPED adjustment using theta and mean from fit
        ctrl_adj = control_arr - self.theta_ * (Y_hat_ctrl - self._mean_y_hat_)
        trt_adj = treatment_arr - self.theta_ * (Y_hat_trt - self._mean_y_hat_)

        return ctrl_adj, trt_adj

    # ── fit_transform ───────────────────────────────────────────────

    def fit_transform(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
        X_control: np.ndarray,
        X_treatment: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit the ML model and apply CUPAC variance reduction in one step.

        Generates out-of-fold predictions via K-fold cross-validation,
        then uses those predictions as the covariate in a CUPED-style
        adjustment.  This is **not** equivalent to ``fit()`` + ``transform()``
        — it uses out-of-fold predictions which avoid overfitting and are
        more accurate for single-dataset adjustment.

        Parameters
        ----------
        control : array-like
            Post-experiment observations for the control group.
        treatment : array-like
            Post-experiment observations for the treatment group.
        X_control : np.ndarray
            Feature matrix for the control group, shape ``(n_ctrl, n_features)``.
        X_treatment : np.ndarray
            Feature matrix for the treatment group, shape ``(n_trt, n_features)``.

        Returns
        -------
        tuple of np.ndarray
            ``(control_adjusted, treatment_adjusted)`` — the CUPAC-adjusted
            observations for each group.

        Raises
        ------
        ImportError
            If scikit-learn is not installed.
        ValueError
            If feature matrices have wrong shape or mismatched columns.

        Notes
        -----
        Prefer this method over ``fit()`` + ``transform()`` when adjusting
        the same dataset used for fitting — out-of-fold predictions avoid
        overfitting and give a more honest variance reduction estimate.
        """
        Ridge, KFold, clone = _check_sklearn()

        # Validate outcome arrays
        control_arr = check_array_like(control, "control", min_length=2)
        treatment_arr = check_array_like(treatment, "treatment", min_length=2)

        # Validate feature matrices
        X_control, X_treatment = _validate_features(
            X_control,
            X_treatment,
            control_arr,
            treatment_arr,
        )

        # Pool all data
        Y = np.concatenate([control_arr, treatment_arr])
        X = np.concatenate([X_control, X_treatment], axis=0)
        n_total = len(Y)
        n_ctrl = len(control_arr)

        # Resolve estimator
        estimator = self._resolve_estimator(Ridge)

        # Generate out-of-fold predictions via K-fold CV
        rng_seed = self._resolve_rng_seed()

        kf = KFold(n_splits=self.cv, shuffle=True, random_state=rng_seed)
        Y_hat = np.empty(n_total, dtype="float64")

        ss_res = 0.0
        ss_tot = 0.0
        y_mean_global = float(np.mean(Y))

        for train_idx, val_idx in kf.split(X):
            fold_model = clone(estimator)
            fold_model.fit(X[train_idx], Y[train_idx])
            preds = fold_model.predict(X[val_idx])
            Y_hat[val_idx] = preds

            # Accumulate R² components for this fold
            ss_res += np.sum((Y[val_idx] - preds) ** 2)
            ss_tot += np.sum((Y[val_idx] - y_mean_global) ** 2)

        # Cross-validated R²
        self.cv_r2_ = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        if self.cv_r2_ < 0.1:
            warnings.warn(
                f"ML model R² is low ({self.cv_r2_:.2f}) on cross-validation. "
                f"CUPAC will provide minimal variance reduction.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Compute CUPED stats
        theta, correlation, variance_reduction, y_hat_mean, degenerate = self._compute_cuped_stats(
            Y, Y_hat
        )
        self.theta_ = theta
        self.correlation_ = correlation
        self.variance_reduction_ = variance_reduction

        if degenerate:
            return control_arr.copy(), treatment_arr.copy()

        # Apply CUPED adjustment: Y_adj = Y - theta * (Y_hat - mean(Y_hat))
        Y_adj = Y - self.theta_ * (Y_hat - y_hat_mean)

        # Also fit on all data and store for potential transform() calls
        self._fitted_estimator_ = clone(estimator)
        self._fitted_estimator_.fit(X, Y)
        self._mean_y_hat_ = float(np.mean(self._fitted_estimator_.predict(X)))
        self._is_fitted = True

        # Split back into control and treatment
        control_adj = Y_adj[:n_ctrl]
        treatment_adj = Y_adj[n_ctrl:]

        return control_adj, treatment_adj
