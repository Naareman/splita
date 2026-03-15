"""DoubleML — Double/Debiased Machine Learning (Chernozhukov et al. 2018).

Implements the partially linear model for estimating Average Treatment Effects
with high-dimensional confounders.  Uses cross-fitting to avoid regularisation
bias and produces root-n consistent, asymptotically normal ATE estimates.

References
----------
.. [1] Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E.,
       Hansen, C., Newey, W., & Robins, J. (2018). "Double/debiased
       machine learning for treatment and structural parameters."
       *The Econometrics Journal*, 21(1), C1-C68.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from splita._types import DoubleMLResult
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
            "DoubleML requires scikit-learn. Install it with: pip install splita[ml]"
        ) from None
    return Ridge, KFold, clone


def _validate_X(X: np.ndarray, name: str) -> np.ndarray:
    """Validate a feature matrix: 2-D, finite."""
    X = np.asarray(X, dtype="float64")
    if X.ndim != 2:
        raise ValueError(
            format_error(
                f"`{name}` must be a 2-D array, got {X.ndim}-D with shape {X.shape}.",
                hint="pass a 2-D feature matrix with shape (n_samples, n_features).",
            )
        )
    if not np.all(np.isfinite(X)):
        raise ValueError(
            format_error(
                f"`{name}` contains NaN or infinite values.",
                hint="remove or impute non-finite values before passing to DoubleML.",
            )
        )
    return X


class DoubleML:
    """Double/Debiased Machine Learning for ATE estimation.

    Implements the partially linear model:

        Y = theta * T + g(X) + epsilon
        T = m(X) + eta

    where ``g(X) = E[Y|X]`` and ``m(X) = E[T|X]`` are nuisance functions
    estimated via cross-fitting, and ``theta`` is the causal parameter of
    interest (the ATE).

    Parameters
    ----------
    outcome_model : object or None, default None
        Sklearn-compatible estimator for E[Y|X].  Defaults to
        ``Ridge(alpha=1.0)``.
    propensity_model : object or None, default None
        Sklearn-compatible estimator for E[T|X].  Defaults to
        ``Ridge(alpha=1.0)``.
    cv : int, default 5
        Number of cross-fitting folds.
    random_state : int, np.random.Generator, or None, default None
        Seed for reproducible CV splits.
    alpha : float, default 0.05
        Significance level for the confidence interval and hypothesis test.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n = 1000
    >>> X = rng.normal(0, 1, size=(n, 5))
    >>> T = (X[:, 0] > 0).astype(float) + rng.normal(0, 0.3, n)
    >>> Y = 2.0 * T + X @ [1, 0.5, 0, 0, 0] + rng.normal(0, 1, n)
    >>> result = DoubleML(random_state=42).fit_transform(Y, T, X)
    >>> abs(result.ate - 2.0) < 0.5
    True
    """

    def __init__(
        self,
        *,
        outcome_model: object | None = None,
        propensity_model: object | None = None,
        cv: int = 5,
        random_state: int | np.random.Generator | None = None,
        alpha: float = 0.05,
    ) -> None:
        if cv < 2:
            raise ValueError(
                format_error(
                    f"`cv` must be >= 2, got {cv}.",
                    detail="cross-fitting requires at least 2 folds.",
                    hint="use cv=5 (default) or cv=10 for more stable estimates.",
                )
            )

        if cv > 50:
            raise ValueError(
                format_error(
                    f"`cv` must be <= 50, got {cv}.",
                    detail="very high fold counts are computationally expensive.",
                    hint="use cv=5 (default) or cv=10; values of 5-10 are typical.",
                )
            )

        for name, model in [
            ("outcome_model", outcome_model),
            ("propensity_model", propensity_model),
        ]:
            if model is not None and not (hasattr(model, "fit") and hasattr(model, "predict")):
                raise ValueError(
                    format_error(
                        f"`{name}` must have fit() and predict() methods.",
                        detail=f"got {type(model).__name__} which is missing "
                        f"{'fit()' if not hasattr(model, 'fit') else 'predict()'}.",
                        hint="pass an sklearn-compatible estimator such as "
                        "Ridge, GradientBoostingRegressor, etc.",
                    )
                )

        if not 0 < alpha < 1:
            raise ValueError(
                format_error(
                    f"`alpha` must be in (0, 1), got {alpha}.",
                    hint="typical values are 0.05, 0.01, or 0.10.",
                )
            )

        self.outcome_model = outcome_model
        self.propensity_model = propensity_model
        self.cv = cv
        self.random_state = random_state
        self.alpha = alpha

    # ── public API ───────────────────────────────────────────────────

    def fit_transform(
        self,
        outcome: ArrayLike,
        treatment: ArrayLike,
        X: np.ndarray,
    ) -> DoubleMLResult:
        """Estimate the ATE using Double ML cross-fitting.

        Parameters
        ----------
        outcome : array-like
            Outcome variable Y, shape ``(n,)``.
        treatment : array-like
            Treatment variable T, shape ``(n,)``.
        X : np.ndarray
            Covariate matrix, shape ``(n, p)``.

        Returns
        -------
        DoubleMLResult
            Frozen dataclass with ATE estimate, SE, p-value, CI, and
            model diagnostics.
        """
        Ridge, KFold, clone = _check_sklearn()

        # Validate inputs
        Y = check_array_like(outcome, "outcome", min_length=2)
        T = check_array_like(treatment, "treatment", min_length=2)
        X = _validate_X(X, "X")

        n = len(Y)
        if len(T) != n:
            raise ValueError(
                format_error(
                    "`outcome` and `treatment` must have the same length.",
                    detail=f"outcome has {n} elements, treatment has {len(T)} elements.",
                )
            )
        if X.shape[0] != n:
            raise ValueError(
                format_error(
                    "`X` must have the same number of rows as `outcome`.",
                    detail=f"X has {X.shape[0]} rows, outcome has {n} elements.",
                )
            )

        # Resolve models
        outcome_est = (
            clone(self.outcome_model) if self.outcome_model is not None else Ridge(alpha=1.0)
        )
        propensity_est = (
            clone(self.propensity_model) if self.propensity_model is not None else Ridge(alpha=1.0)
        )

        # Resolve random state for KFold
        rng_seed = self.random_state
        if isinstance(rng_seed, np.random.Generator):
            rng_seed = int(rng_seed.integers(0, 2**31))

        # Step 1-2: Cross-fit both models
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=rng_seed)

        Y_hat = np.empty(n, dtype="float64")
        T_hat = np.empty(n, dtype="float64")

        # Track R-squared components
        ss_res_y, ss_tot_y = 0.0, 0.0
        ss_res_t, ss_tot_t = 0.0, 0.0
        y_mean = float(np.mean(Y))
        t_mean = float(np.mean(T))

        for train_idx, val_idx in kf.split(X):
            # Outcome model: E[Y|X]
            y_model = clone(outcome_est)
            y_model.fit(X[train_idx], Y[train_idx])
            y_pred = y_model.predict(X[val_idx])
            Y_hat[val_idx] = y_pred

            ss_res_y += np.sum((Y[val_idx] - y_pred) ** 2)
            ss_tot_y += np.sum((Y[val_idx] - y_mean) ** 2)

            # Propensity model: E[T|X]
            t_model = clone(propensity_est)
            t_model.fit(X[train_idx], T[train_idx])
            t_pred = t_model.predict(X[val_idx])
            T_hat[val_idx] = t_pred

            ss_res_t += np.sum((T[val_idx] - t_pred) ** 2)
            ss_tot_t += np.sum((T[val_idx] - t_mean) ** 2)

        # Cross-validated R-squared
        outcome_r2 = float(1.0 - ss_res_y / ss_tot_y) if ss_tot_y > 0 else 0.0
        propensity_r2 = float(1.0 - ss_res_t / ss_tot_t) if ss_tot_t > 0 else 0.0

        # Step 3: Residualize
        Y_resid = Y - Y_hat
        T_resid = T - T_hat

        # Step 4: Estimate ATE via partial regression
        denom = float(T_resid @ T_resid)
        theta = 0.0 if denom == 0.0 else float(T_resid @ Y_resid) / denom

        # Step 5-6: SE via influence function
        psi = (Y_resid - theta * T_resid) * T_resid
        mean_T_resid_sq = float(np.mean(T_resid**2))

        if mean_T_resid_sq == 0.0:
            se = float("inf")
        else:
            se = float(np.sqrt(np.mean(psi**2) / n)) / mean_T_resid_sq

        # p-value and CI
        z = theta / se if se > 0 and se != float("inf") else 0.0
        pvalue = float(2 * norm.sf(abs(z)))

        z_crit = float(norm.ppf(1 - self.alpha / 2))
        ci_lower = theta - z_crit * se
        ci_upper = theta + z_crit * se

        # Variance reduction: compare SE to naive difference-in-means SE
        naive_se = float(np.std(Y, ddof=1)) / np.sqrt(n)
        variance_reduction = max(0.0, 1.0 - (se / naive_se) ** 2) if naive_se > 0 else 0.0

        return DoubleMLResult(
            ate=theta,
            se=se,
            pvalue=pvalue,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            significant=pvalue < self.alpha,
            outcome_r2=outcome_r2,
            propensity_r2=propensity_r2,
            variance_reduction=variance_reduction,
        )
