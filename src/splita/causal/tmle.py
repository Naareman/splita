"""Targeted Maximum Likelihood Estimation (TMLE).

Two-step semiparametric estimator for the average treatment effect
(van der Laan & Rubin, 2006). Uses initial outcome and propensity
models followed by a targeted update step.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from splita._types import TMLEResult
from splita._validation import check_array_like, check_same_length, format_error

ArrayLike = list | tuple | np.ndarray


class TMLE:
    """Targeted Maximum Likelihood Estimation for the ATE.

    Implements the two-step TMLE procedure:
    1. Initial fit of outcome model and propensity score.
    2. Targeted update (fluctuation) using the clever covariate.

    Parameters
    ----------
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Notes
    -----
    Uses scikit-learn ``LogisticRegression`` and ``Ridge`` internally
    (lazy import). Falls back to simple linear models if sklearn is
    unavailable.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n = 300
    >>> X = rng.normal(0, 1, (n, 3))
    >>> A = rng.binomial(1, 0.5, n)
    >>> Y = 2.0 * A + X[:, 0] + rng.normal(0, 1, n)
    >>> tmle = TMLE()
    >>> r = tmle.fit(Y, A, X)
    >>> abs(r.ate - 2.0) < 1.5
    True
    """

    def __init__(self, *, alpha: float = 0.05) -> None:
        if not 0 < alpha < 1:
            raise ValueError(
                format_error(
                    f"`alpha` must be in (0, 1), got {alpha}.",
                    "alpha represents the significance level.",
                    "typical values are 0.05, 0.01, or 0.10.",
                )
            )
        self._alpha = alpha

    def fit(
        self,
        outcome: ArrayLike,
        treatment: ArrayLike,
        covariates: ArrayLike,
    ) -> TMLEResult:
        """Fit TMLE to estimate the average treatment effect.

        Parameters
        ----------
        outcome : array-like, shape (n,)
            Outcome variable.
        treatment : array-like, shape (n,)
            Binary treatment indicator (0 or 1).
        covariates : array-like, shape (n, p)
            Covariates / confounders.

        Returns
        -------
        TMLEResult
            Frozen dataclass with ATE estimate and diagnostics.

        Raises
        ------
        TypeError
            If inputs cannot be converted to numeric arrays.
        ValueError
            If arrays are too short, mismatched, or treatment is not binary.
        """
        Y = check_array_like(outcome, "outcome", min_length=10)
        A = check_array_like(treatment, "treatment", min_length=10)
        check_same_length(Y, A, "outcome", "treatment")

        X = np.asarray(covariates, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n = len(Y)
        if X.shape[0] != n:
            raise ValueError(
                format_error(
                    "`covariates` must have the same number of rows as `outcome`.",
                    f"outcome has {n} rows, covariates has {X.shape[0]}.",
                )
            )

        unique_a = np.unique(A)
        if not np.all(np.isin(unique_a, [0, 1])):
            raise ValueError(
                format_error(
                    "`treatment` must be binary (0 or 1).",
                    f"got unique values: {unique_a.tolist()}.",
                    "encode your treatment indicator as 0/1.",
                )
            )

        # Step 1: Initial estimates
        # Outcome model: E[Y | A, X]
        Q1, Q0 = self._fit_outcome_model(Y, A, X)

        # Propensity score: P(A=1 | X)
        g = self._fit_propensity(A, X)
        g = np.clip(g, 0.025, 0.975)

        # Initial ATE estimate
        initial_estimate = float(np.mean(Q1 - Q0))

        # Step 2: Targeted update
        # Clever covariate
        H1 = 1.0 / g
        H0 = -1.0 / (1.0 - g)
        H_A = A * H1 + (1.0 - A) * H0

        # Initial prediction for observed (A, X)
        Q_A = A * Q1 + (1.0 - A) * Q0

        # Fluctuation parameter (epsilon) via univariate regression
        resid = Y - Q_A
        epsilon = float(np.sum(H_A * resid) / (np.sum(H_A**2) + 1e-12))

        # Updated predictions
        Q1_star = Q1 + epsilon * H1
        Q0_star = Q0 + epsilon * H0

        # Targeted ATE
        targeted_estimate = float(np.mean(Q1_star - Q0_star))
        ate = targeted_estimate

        # Influence function for SE
        D = H_A * (Y - Q_A - epsilon * H_A) + (Q1_star - Q0_star) - ate
        se = float(np.std(D, ddof=1) / np.sqrt(n))

        if se > 0:
            z = ate / se
            pvalue = float(2.0 * norm.sf(abs(z)))
        else:
            pvalue = 1.0 if ate == 0 else 0.0

        z_crit = float(norm.ppf(1 - self._alpha / 2))
        ci_lower = ate - z_crit * se
        ci_upper = ate + z_crit * se

        return TMLEResult(
            ate=ate,
            se=se,
            pvalue=pvalue,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            initial_estimate=initial_estimate,
            targeted_estimate=targeted_estimate,
        )

    @staticmethod
    def _fit_outcome_model(
        Y: np.ndarray, A: np.ndarray, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit E[Y | A, X] and predict Q(1,X), Q(0,X).

        Uses sklearn Ridge if available, falls back to OLS.
        """
        n, _p = X.shape
        X_with_a = np.column_stack([X, A])

        try:
            from sklearn.linear_model import Ridge

            model = Ridge(alpha=1.0)
            model.fit(X_with_a, Y)
            X1 = np.column_stack([X, np.ones(n)])
            X0 = np.column_stack([X, np.zeros(n)])
            Q1 = model.predict(X1)
            Q0 = model.predict(X0)
        except ImportError:
            # OLS fallback
            X_bias = np.column_stack([np.ones(n), X_with_a])
            beta = np.linalg.lstsq(X_bias, Y, rcond=None)[0]
            X1_bias = np.column_stack([np.ones(n), X, np.ones(n)])
            X0_bias = np.column_stack([np.ones(n), X, np.zeros(n)])
            Q1 = X1_bias @ beta
            Q0 = X0_bias @ beta

        return Q1, Q0

    @staticmethod
    def _fit_propensity(A: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Fit P(A=1 | X) via logistic regression."""
        try:
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(max_iter=1000, C=1.0)
            model.fit(X, A)
            g = model.predict_proba(X)[:, 1]
        except ImportError:
            # Fallback: simple logistic via gradient descent
            n, p = X.shape
            X_bias = np.column_stack([np.ones(n), X])
            beta = np.zeros(p + 1)
            lr = 0.1
            for _ in range(200):
                logits = np.clip(X_bias @ beta, -30, 30)
                prob = 1.0 / (1.0 + np.exp(-logits))
                grad = X_bias.T @ (prob - A) / n
                beta -= lr * grad
            logits = np.clip(X_bias @ beta, -30, 30)
            g = 1.0 / (1.0 + np.exp(-logits))

        return g
