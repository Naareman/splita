"""Effect transportability estimator.

Transports a treatment effect estimated in an experimental population
to a target population using inverse odds weighting (Rosenman et al. 2025).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from splita._types import TransportResult
from splita._validation import check_array_like, check_same_length, format_error

ArrayLike = list | tuple | np.ndarray


class EffectTransport:
    """Transport treatment effects across populations.

    Uses inverse odds of participation weights to re-weight experimental
    estimates so they apply to a target population with different covariate
    distributions.

    Parameters
    ----------
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n_exp = 200
    >>> exp_treat = rng.binomial(1, 0.5, n_exp)
    >>> exp_cov = rng.normal(0, 1, (n_exp, 2))
    >>> exp_outcome = 2.0 * exp_treat + exp_cov[:, 0] + rng.normal(0, 1, n_exp)
    >>> target_cov = rng.normal(0.5, 1, (500, 2))
    >>> et = EffectTransport()
    >>> r = et.transport(exp_outcome, exp_treat, exp_cov, target_cov)
    >>> isinstance(r.transported_ate, float)
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

    def transport(
        self,
        experiment_outcome: ArrayLike,
        experiment_treatment: ArrayLike,
        experiment_covariates: ArrayLike,
        target_covariates: ArrayLike,
    ) -> TransportResult:
        """Transport experimental results to a target population.

        Parameters
        ----------
        experiment_outcome : array-like, shape (n_exp,)
            Outcome variable in the experiment.
        experiment_treatment : array-like, shape (n_exp,)
            Binary treatment indicator (0 or 1) in the experiment.
        experiment_covariates : array-like, shape (n_exp, p)
            Covariates in the experiment.
        target_covariates : array-like, shape (n_target, p)
            Covariates in the target population.

        Returns
        -------
        TransportResult
            Frozen dataclass with the transported ATE and diagnostics.

        Raises
        ------
        TypeError
            If inputs cannot be converted to numeric arrays.
        ValueError
            If arrays are too short or have incompatible shapes.
        """
        y = check_array_like(experiment_outcome, "experiment_outcome", min_length=4)
        a = check_array_like(experiment_treatment, "experiment_treatment", min_length=4)
        check_same_length(y, a, "experiment_outcome", "experiment_treatment")

        X_exp = np.asarray(experiment_covariates, dtype=float)
        X_tgt = np.asarray(target_covariates, dtype=float)

        if X_exp.ndim == 1:
            X_exp = X_exp.reshape(-1, 1)
        if X_tgt.ndim == 1:
            X_tgt = X_tgt.reshape(-1, 1)

        n_exp = len(y)
        n_tgt = X_tgt.shape[0]

        if X_exp.shape[0] != n_exp:
            raise ValueError(
                format_error(
                    "`experiment_covariates` must have the same number of rows "
                    "as `experiment_outcome`.",
                    f"outcome has {n_exp} rows, covariates has {X_exp.shape[0]}.",
                )
            )
        if X_exp.shape[1] != X_tgt.shape[1]:
            raise ValueError(
                format_error(
                    "Experiment and target covariates must have the same number of columns.",
                    f"experiment has {X_exp.shape[1]}, target has {X_tgt.shape[1]}.",
                )
            )

        # Estimate participation probabilities via logistic regression
        # S=1 for experiment, S=0 for target
        X_combined = np.vstack([X_exp, X_tgt])
        s = np.concatenate([np.ones(n_exp), np.zeros(n_tgt)])

        # Standardise covariates
        mu = np.mean(X_combined, axis=0)
        sd = np.std(X_combined, axis=0) + 1e-10
        X_std = (X_combined - mu) / sd

        # Logistic regression via IRLS (simplified)
        p_hat = self._logistic_fit_predict(X_std, s, n_exp)

        # Inverse odds weights for experiment units
        # w_i = (1 - p_hat_i) / p_hat_i -- re-weights experiment to look like target
        p_exp = np.clip(p_hat[:n_exp], 0.01, 0.99)
        weights = (1.0 - p_exp) / p_exp

        # Normalise weights
        treated_mask = a == 1
        control_mask = a == 0

        if np.sum(treated_mask) < 1 or np.sum(control_mask) < 1:
            raise ValueError(
                format_error(
                    "Both treated and control groups must have at least 1 unit.",
                    f"got {int(np.sum(treated_mask))} treated, "
                    f"{int(np.sum(control_mask))} control.",
                )
            )

        w_t = weights[treated_mask]
        w_c = weights[control_mask]
        y_t = y[treated_mask]
        y_c = y[control_mask]

        # Weighted means
        mean_t = float(np.sum(w_t * y_t) / np.sum(w_t))
        mean_c = float(np.sum(w_c * y_c) / np.sum(w_c))
        transported_ate = mean_t - mean_c

        # Standard error via weighted sandwich estimator
        w_t_norm = w_t / np.sum(w_t)
        w_c_norm = w_c / np.sum(w_c)
        var_t = float(np.sum(w_t_norm**2 * (y_t - mean_t) ** 2))
        var_c = float(np.sum(w_c_norm**2 * (y_c - mean_c) ** 2))
        se = float(np.sqrt(var_t + var_c))

        z_crit = float(norm.ppf(1 - self._alpha / 2))
        ci_lower = transported_ate - z_crit * se
        ci_upper = transported_ate + z_crit * se

        # Weight diagnostics
        eff_n = float(np.sum(weights) ** 2 / np.sum(weights**2))
        weight_diagnostics = {
            "max_weight": float(np.max(weights)),
            "mean_weight": float(np.mean(weights)),
            "effective_n": eff_n,
        }

        return TransportResult(
            transported_ate=transported_ate,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            weight_diagnostics=weight_diagnostics,
        )

    @staticmethod
    def _logistic_fit_predict(X: np.ndarray, y: np.ndarray, n_exp: int) -> np.ndarray:
        """Simple logistic regression via gradient descent."""
        n, p = X.shape
        X_bias = np.column_stack([np.ones(n), X])
        beta = np.zeros(p + 1)

        lr = 0.1
        for _ in range(200):
            logits = X_bias @ beta
            logits = np.clip(logits, -30, 30)
            prob = 1.0 / (1.0 + np.exp(-logits))
            grad = X_bias.T @ (prob - y) / n
            beta -= lr * grad

        logits = X_bias @ beta
        logits = np.clip(logits, -30, 30)
        return 1.0 / (1.0 + np.exp(-logits))
