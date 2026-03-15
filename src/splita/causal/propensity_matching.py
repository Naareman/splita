"""Propensity Score Matching for observational causal inference.

Uses logistic regression to estimate propensity scores, then
nearest-neighbor matching with optional caliper to estimate ATT.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from splita._types import PSMResult
from splita._validation import check_array_like, check_in_range, check_positive, format_error

ArrayLike = list | tuple | np.ndarray


def _logistic_regression(X: np.ndarray, y: np.ndarray, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Fit logistic regression via IRLS (no sklearn dependency).

    Parameters
    ----------
    X : np.ndarray
        Feature matrix with intercept column (n x p).
    y : np.ndarray
        Binary outcome vector.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    np.ndarray
        Coefficient vector.
    """
    n, p = X.shape
    beta = np.zeros(p)

    for _ in range(max_iter):
        z = X @ beta
        # Clip to prevent overflow
        z = np.clip(z, -30, 30)
        prob = 1.0 / (1.0 + np.exp(-z))
        prob = np.clip(prob, 1e-10, 1 - 1e-10)

        W = prob * (1 - prob)
        gradient = X.T @ (y - prob)
        hessian = X.T @ (X * W[:, np.newaxis])

        try:
            delta = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break

        beta_new = beta + delta
        if np.max(np.abs(delta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    return beta


def _standardised_mean_diff(treated: np.ndarray, control: np.ndarray) -> float:
    """Compute standardised mean difference (SMD).

    Parameters
    ----------
    treated : np.ndarray
        Values for treated group.
    control : np.ndarray
        Values for control group.

    Returns
    -------
    float
        Absolute standardised mean difference.
    """
    mean_diff = float(np.mean(treated) - np.mean(control))
    var_t = float(np.var(treated, ddof=1)) if len(treated) > 1 else 0.0
    var_c = float(np.var(control, ddof=1)) if len(control) > 1 else 0.0
    pooled = math.sqrt((var_t + var_c) / 2.0) if (var_t + var_c) > 0 else 1.0
    return abs(mean_diff / pooled)


class PropensityScoreMatching:
    """Propensity Score Matching for causal inference.

    Estimates the Average Treatment effect on the Treated (ATT)
    by matching treated units to similar control units based on
    estimated propensity scores.

    Parameters
    ----------
    n_neighbors : int, default 1
        Number of nearest neighbors to match.
    caliper : float or None, default None
        Maximum distance in propensity score units for a valid match.
        If None, all matches are accepted.
    random_state : int, np.random.Generator, or None, default None
        Random state (unused, reserved for future tie-breaking).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n = 500
    >>> x = rng.normal(0, 1, (n, 2))
    >>> t = (x[:, 0] + rng.normal(0, 0.5, n) > 0).astype(float)
    >>> y = 2.0 * t + x[:, 0] + rng.normal(0, 1, n)
    >>> psm = PropensityScoreMatching()
    >>> r = psm.fit(y, t, x)
    >>> abs(r.att - 2.0) < 2.0
    True
    """

    def __init__(
        self,
        *,
        n_neighbors: int = 1,
        caliper: float | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        if not isinstance(n_neighbors, int) or n_neighbors < 1:
            raise ValueError(
                format_error(
                    "`n_neighbors` must be a positive integer, got {}.".format(n_neighbors),
                    hint="use n_neighbors=1 for 1:1 matching.",
                )
            )
        if caliper is not None:
            check_positive(caliper, "caliper")
        self._n_neighbors = n_neighbors
        self._caliper = caliper
        self._random_state = random_state

    def fit(
        self,
        outcome: ArrayLike,
        treatment: ArrayLike,
        covariates: np.ndarray,
    ) -> PSMResult:
        """Fit propensity score model and estimate ATT.

        Parameters
        ----------
        outcome : array-like
            Outcome variable.
        treatment : array-like
            Binary treatment indicator (0 or 1).
        covariates : np.ndarray
            Covariate matrix (n x p).

        Returns
        -------
        PSMResult
            ATT estimate with balance diagnostics.

        Raises
        ------
        TypeError
            If inputs cannot be converted to numeric arrays.
        ValueError
            If arrays have mismatched lengths or insufficient data.
        """
        y = check_array_like(outcome, "outcome", min_length=4)
        t = check_array_like(treatment, "treatment", min_length=4)
        n = len(y)

        if len(t) != n:
            raise ValueError(
                format_error(
                    "`outcome` and `treatment` must have the same length.",
                    f"outcome has {n} elements, treatment has {len(t)} elements.",
                )
            )

        if not isinstance(covariates, np.ndarray):
            raise TypeError(
                format_error(
                    "`covariates` must be a numpy ndarray.",
                    f"got type {type(covariates).__name__}.",
                )
            )
        if covariates.ndim == 1:
            covariates = covariates.reshape(-1, 1)
        if covariates.shape[0] != n:
            raise ValueError(
                format_error(
                    "`covariates` must have the same number of rows as outcome.",
                    f"outcome has {n} rows, covariates has {covariates.shape[0]} rows.",
                )
            )

        # Binary treatment check
        unique_t = np.unique(t)
        if not np.all(np.isin(unique_t, [0.0, 1.0])):
            raise ValueError(
                format_error(
                    "`treatment` must be binary (0 or 1).",
                    f"found unique values: {unique_t.tolist()}.",
                )
            )

        treated_idx = np.where(t == 1.0)[0]
        control_idx = np.where(t == 0.0)[0]

        if len(treated_idx) < 2 or len(control_idx) < 2:
            raise ValueError(
                format_error(
                    "Both treatment and control groups must have at least 2 units.",
                    f"treated: {len(treated_idx)}, control: {len(control_idx)}.",
                )
            )

        # Estimate propensity scores via logistic regression
        X_lr = np.column_stack([np.ones(n), covariates])
        beta = _logistic_regression(X_lr, t)
        z = X_lr @ beta
        z = np.clip(z, -30, 30)
        ps = 1.0 / (1.0 + np.exp(-z))

        # Balance before matching
        n_covariates = covariates.shape[1]
        balance_before: dict[str, float] = {}
        for j in range(n_covariates):
            smd = _standardised_mean_diff(
                covariates[treated_idx, j], covariates[control_idx, j]
            )
            balance_before[f"X{j}"] = smd

        # Nearest-neighbor matching for each treated unit
        ps_treated = ps[treated_idx]
        ps_control = ps[control_idx]

        matched_treated: list[int] = []
        matched_control: list[int] = []

        for i, ps_t in enumerate(ps_treated):
            distances = np.abs(ps_control - ps_t)
            nn_indices = np.argsort(distances)[: self._n_neighbors]

            if self._caliper is not None:
                nn_indices = [idx for idx in nn_indices if distances[idx] <= self._caliper]

            if len(nn_indices) > 0:
                matched_treated.append(treated_idx[i])
                for nn_idx in nn_indices:
                    matched_control.append(control_idx[nn_idx])

        n_matched = len(matched_treated)
        n_unmatched = len(treated_idx) - n_matched

        if n_matched == 0:
            raise ValueError(
                format_error(
                    "No matches found within caliper.",
                    f"caliper={self._caliper} is too restrictive.",
                    hint="increase caliper or set caliper=None.",
                )
            )

        # ATT from matched pairs
        y_treated_matched = y[matched_treated]
        y_control_matched = np.array(
            [float(np.mean(y[matched_control[i * self._n_neighbors : (i + 1) * self._n_neighbors]]))
             for i in range(n_matched)]
        ) if self._n_neighbors > 1 else y[matched_control]

        # Truncate to same length
        min_len = min(len(y_treated_matched), len(y_control_matched))
        y_treated_matched = y_treated_matched[:min_len]
        y_control_matched = y_control_matched[:min_len]

        att = float(np.mean(y_treated_matched - y_control_matched))

        # SE of the ATT
        diffs = y_treated_matched - y_control_matched
        se = float(np.std(diffs, ddof=1)) / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0

        if se > 0:
            z_stat = att / se
            pvalue = float(2.0 * norm.sf(abs(z_stat)))
            z_crit = float(norm.ppf(1 - 0.05 / 2))
            ci_lower = att - z_crit * se
            ci_upper = att + z_crit * se
        else:
            pvalue = 1.0 if att == 0 else 0.0
            ci_lower = att
            ci_upper = att

        # Balance after matching
        balance_after: dict[str, float] = {}
        for j in range(n_covariates):
            if len(matched_treated) > 0 and len(matched_control) > 0:
                smd = _standardised_mean_diff(
                    covariates[matched_treated, j],
                    covariates[matched_control[:len(matched_treated)], j]
                    if len(matched_control) >= len(matched_treated)
                    else covariates[matched_control, j],
                )
            else:
                smd = 0.0
            balance_after[f"X{j}"] = smd

        return PSMResult(
            att=att,
            se=se,
            pvalue=pvalue,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            significant=pvalue < 0.05,
            n_matched=n_matched,
            n_unmatched=n_unmatched,
            balance_before=balance_before,
            balance_after=balance_after,
        )
