"""DoublyRobustEstimator — Augmented Inverse Propensity Weighting (AIPW).

Combines an outcome model with propensity score weighting for doubly
robust estimation of the average treatment effect.

References
----------
.. [1] Robins, J. M. (1994).  "Correcting for non-compliance in randomized
       trials using structural nested mean models."
.. [2] Bang, H. & Robins, J. M. (2005).  "Doubly robust estimation in
       missing data and causal inference models."
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.stats import norm

from splita._types import DoublyRobustResult
from splita._validation import (
    check_array_like,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


class DoublyRobustEstimator:
    """Doubly robust (AIPW) estimator for the average treatment effect.

    Consistent if *either* the outcome model *or* the propensity model
    is correctly specified (hence "doubly robust"). Uses cross-fitting
    (2-fold) to avoid overfitting bias.

    Parameters
    ----------
    alpha : float, default 0.05
        Significance level for confidence intervals.
    n_folds : int, default 2
        Number of cross-fitting folds.
    random_state : int or None, default None
        Seed for reproducibility of cross-fitting splits.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n = 500
    >>> x = rng.normal(0, 1, (n, 3))
    >>> t = (x[:, 0] + rng.normal(0, 1, n) > 0).astype(float)
    >>> y = 2 * t + x[:, 0] + rng.normal(0, 1, n)
    >>> result = DoublyRobustEstimator(random_state=42).fit(y, t, x)
    >>> abs(result.ate - 2.0) < 1.0
    True
    """

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        n_folds: int = 2,
        random_state: int | None = None,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(
                format_error(
                    f"`alpha` must be in (0, 1), got {alpha}.",
                    hint="typical values are 0.05, 0.01, or 0.10.",
                )
            )
        if n_folds < 2:
            raise ValueError(
                format_error(
                    f"`n_folds` must be >= 2, got {n_folds}.",
                    hint="use at least 2 folds for cross-fitting.",
                )
            )
        self._alpha = alpha
        self._n_folds = n_folds
        self._random_state = random_state

    def fit(
        self,
        outcome: ArrayLike,
        treatment: ArrayLike,
        covariates: np.ndarray | ArrayLike,
    ) -> DoublyRobustResult:
        """Estimate the ATE using doubly robust (AIPW) estimation.

        Parameters
        ----------
        outcome : array-like
            Observed outcomes (Y).
        treatment : array-like
            Treatment assignments (0 or 1).
        covariates : 2-D array-like
            Covariate matrix (X).

        Returns
        -------
        DoublyRobustResult
            ATE estimate with standard error, p-value, and diagnostics.

        Raises
        ------
        ValueError
            If inputs are invalid or sklearn is not available.
        """
        # Lazy sklearn import
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.linear_model import Ridge
            from sklearn.metrics import r2_score, roc_auc_score
            from sklearn.model_selection import KFold
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                format_error(
                    "DoublyRobustEstimator requires scikit-learn.",
                    detail=str(exc),
                    hint="install scikit-learn: pip install scikit-learn.",
                )
            ) from exc

        y = check_array_like(outcome, "outcome", min_length=10)
        t = check_array_like(treatment, "treatment", min_length=10)
        X = np.asarray(covariates, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                format_error(
                    f"`covariates` must be a 2-D array, got {X.ndim}-D array with shape {X.shape}.",
                    hint="pass a matrix with shape (n_samples, n_features).",
                )
            )

        if len(y) != len(t) or len(y) != X.shape[0]:
            raise ValueError(
                format_error(
                    "`outcome`, `treatment`, and `covariates` must have the "
                    "same number of samples.",
                    detail=f"outcome: {len(y)}, treatment: {len(t)}, covariates: {X.shape[0]}.",
                )
            )

        unique_t = np.unique(t)
        if not (len(unique_t) == 2 and set(unique_t) <= {0.0, 1.0}):
            raise ValueError(
                format_error(
                    "`treatment` must contain only 0s and 1s.",
                    detail=f"unique values: {unique_t.tolist()}.",
                    hint="encode treatment as binary (0 = control, 1 = treated).",
                )
            )

        n = len(y)
        mu0_hat = np.zeros(n)
        mu1_hat = np.zeros(n)
        e_hat = np.zeros(n)
        y_pred_all = np.zeros(n)

        kf = KFold(
            n_splits=self._n_folds,
            shuffle=True,
            random_state=self._random_state,
        )

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, _y_test = y[train_idx], y[test_idx]
            t_train, _t_test = t[train_idx], t[test_idx]

            # Outcome model (separate for treated and control)
            outcome_model_0 = Ridge(alpha=1.0)
            outcome_model_1 = Ridge(alpha=1.0)

            mask_0 = t_train == 0
            mask_1 = t_train == 1

            if np.sum(mask_0) < 2 or np.sum(mask_1) < 2:  # pragma: no cover
                warnings.warn(
                    "Too few treated/control observations in fold. Results may be unreliable.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            outcome_model_0.fit(X_train[mask_0], y_train[mask_0])
            outcome_model_1.fit(X_train[mask_1], y_train[mask_1])

            mu0_hat[test_idx] = outcome_model_0.predict(X_test)
            mu1_hat[test_idx] = outcome_model_1.predict(X_test)

            # For R2 on outcome: use whichever model applies
            for idx in test_idx:
                if t[idx] == 0:
                    y_pred_all[idx] = mu0_hat[idx]
                else:
                    y_pred_all[idx] = mu1_hat[idx]

            # Propensity model
            prop_model = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=3,
                random_state=self._random_state,
            )
            prop_model.fit(X_train, t_train)
            e_hat[test_idx] = np.clip(prop_model.predict_proba(X_test)[:, 1], 0.01, 0.99)

        # AIPW scores
        aipw_scores = (
            mu1_hat - mu0_hat + t * (y - mu1_hat) / e_hat - (1 - t) * (y - mu0_hat) / (1 - e_hat)
        )

        ate = float(np.mean(aipw_scores))
        se = float(np.std(aipw_scores, ddof=1) / math.sqrt(n))

        # p-value (two-sided)
        z = ate / se if se > 0 else 0.0
        pvalue = float(2.0 * norm.sf(abs(z)))

        # CI
        z_crit = float(norm.ppf(1.0 - self._alpha / 2.0))
        ci_lower = ate - z_crit * se
        ci_upper = ate + z_crit * se

        # Diagnostics
        outcome_r2 = float(r2_score(y, y_pred_all))
        try:
            propensity_auc = float(roc_auc_score(t, e_hat))
        except ValueError:  # pragma: no cover
            propensity_auc = 0.5  # degenerate case

        return DoublyRobustResult(
            ate=ate,
            se=se,
            pvalue=pvalue,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            outcome_r2=outcome_r2,
            propensity_auc=propensity_auc,
        )
