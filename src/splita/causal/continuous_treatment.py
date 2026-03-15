"""Continuous treatment effect — dose-response curve estimation.

Estimates the causal dose-response curve for continuous treatment
variables (e.g. price, frequency, dosage) using kernel-weighted
local linear regression (Hirano & Imbens 2004).
"""

from __future__ import annotations

import numpy as np

from splita._types import DoseResponseResult
from splita._validation import (
    check_array_like,
    check_same_length,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


def _gaussian_kernel(u: np.ndarray) -> np.ndarray:
    """Gaussian kernel function."""
    return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)


def _local_linear_regression(
    x: np.ndarray,
    y: np.ndarray,
    x_eval: float,
    bandwidth: float,
) -> tuple[float, float]:
    """Fit a local linear regression at a single evaluation point.

    Parameters
    ----------
    x : np.ndarray
        Predictor values.
    y : np.ndarray
        Response values.
    x_eval : float
        Point at which to evaluate the regression.
    bandwidth : float
        Kernel bandwidth.

    Returns
    -------
    intercept : float
        Local intercept (fitted value at x_eval).
    slope : float
        Local slope at x_eval.
    """
    u = (x - x_eval) / bandwidth
    w = _gaussian_kernel(u)

    # Weighted least squares: [1, x - x_eval] @ beta = y
    dx = x - x_eval
    sw = np.sum(w)
    if sw < 1e-10:
        return float(np.mean(y)), 0.0  # pragma: no cover

    swx = np.sum(w * dx)
    swx2 = np.sum(w * dx**2)
    swy = np.sum(w * y)
    swxy = np.sum(w * dx * y)

    det = sw * swx2 - swx**2
    if abs(det) < 1e-10:
        return float(swy / sw), 0.0  # pragma: no cover

    intercept = (swx2 * swy - swx * swxy) / det
    slope = (sw * swxy - swx * swy) / det
    return float(intercept), float(slope)


class ContinuousTreatmentEffect:
    """Dose-response curve estimation for continuous treatments.

    Uses kernel-weighted local linear regression to estimate the
    relationship between a continuous treatment variable and the
    outcome, following Hirano & Imbens (2004).

    Parameters
    ----------
    n_grid : int, default 50
        Number of evaluation points along the dose range.
    bandwidth : float or None, default None
        Kernel bandwidth. If None, uses Silverman's rule of thumb.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> dose = rng.uniform(0, 10, 200)
    >>> outcome = 2 * dose - 0.1 * dose**2 + rng.normal(0, 1, 200)
    >>> cte = ContinuousTreatmentEffect()
    >>> cte.fit(outcome, dose)  # doctest: +ELLIPSIS
    <splita.causal.continuous_treatment.ContinuousTreatmentEffect object at ...>
    >>> r = cte.result()
    >>> len(r.dose_response_curve) > 0
    True
    """

    def __init__(
        self,
        *,
        n_grid: int = 50,
        bandwidth: float | None = None,
    ) -> None:
        if n_grid < 3:
            raise ValueError(
                format_error(
                    f"`n_grid` must be >= 3, got {n_grid}.",
                    "need at least 3 evaluation points for a dose-response curve.",
                )
            )
        self._n_grid = int(n_grid)
        self._bandwidth = bandwidth
        self._result: DoseResponseResult | None = None

    def fit(
        self,
        outcome: ArrayLike,
        treatment_dose: ArrayLike,
        covariates: ArrayLike | None = None,
    ) -> ContinuousTreatmentEffect:
        """Fit the dose-response model.

        Parameters
        ----------
        outcome : array-like
            Observed outcomes.
        treatment_dose : array-like
            Continuous treatment doses.
        covariates : array-like or None, default None
            Optional covariates. If provided, the outcome is first
            residualised on the covariates before fitting the
            dose-response curve.

        Returns
        -------
        ContinuousTreatmentEffect
            The fitted estimator (self).

        Raises
        ------
        TypeError
            If inputs cannot be converted to numeric arrays.
        ValueError
            If arrays have fewer than 10 elements or mismatched lengths.
        """
        y = check_array_like(outcome, "outcome", min_length=10)
        d = check_array_like(treatment_dose, "treatment_dose", min_length=10)
        check_same_length(y, d, "outcome", "treatment_dose")

        n = len(y)

        # Residualise outcome on covariates if provided
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
            # OLS residualisation: y_resid = y - X(X'X)^{-1}X'y
            X_cov = np.column_stack([np.ones(n), cov])
            try:
                beta = np.linalg.lstsq(X_cov, y, rcond=None)[0]
                y = y - X_cov @ beta
            except np.linalg.LinAlgError:  # pragma: no cover
                pass  # Fall back to raw y

        # Bandwidth selection: Silverman's rule of thumb
        if self._bandwidth is not None:
            bw = self._bandwidth
        else:
            std_d = float(np.std(d))
            iqr_d = float(np.percentile(d, 75) - np.percentile(d, 25))
            if iqr_d > 0:
                bw = 0.9 * min(std_d, iqr_d / 1.34) * n ** (-0.2)
            elif std_d > 0:  # pragma: no cover
                bw = 0.9 * std_d * n ** (-0.2)
            else:
                bw = 1.0  # pragma: no cover

        if bw <= 0:  # pragma: no cover
            bw = 1.0

        # Evaluation grid
        d_min = float(np.min(d))
        d_max = float(np.max(d))
        # Trim edges slightly to avoid boundary effects
        margin = 0.05 * (d_max - d_min)
        grid = np.linspace(d_min + margin, d_max - margin, self._n_grid)

        # Fit local linear regression at each grid point
        curve: list[tuple[float, float]] = []
        fitted_values = np.empty(self._n_grid)

        for i, x_eval in enumerate(grid):
            intercept, _slope = _local_linear_regression(d, y, x_eval, bw)
            curve.append((float(x_eval), intercept))
            fitted_values[i] = intercept

        # Optimal dose: dose that maximises the fitted effect
        best_idx = int(np.argmax(fitted_values))
        optimal_dose = float(grid[best_idx])

        # Slope at the mean dose
        mean_dose = float(np.mean(d))
        _, slope_at_mean = _local_linear_regression(d, y, mean_dose, bw)

        # R-squared (using linear fit as baseline)
        X_lin = np.column_stack([np.ones(n), d])
        beta_lin = np.linalg.lstsq(X_lin, y, rcond=None)[0]
        y_pred_lin = X_lin @ beta_lin
        ss_res = float(np.sum((y - y_pred_lin) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        self._result = DoseResponseResult(
            dose_response_curve=curve,
            optimal_dose=optimal_dose,
            slope_at_mean=float(slope_at_mean),
            r_squared=float(r_squared),
            n=n,
        )
        return self

    def result(self) -> DoseResponseResult:
        """Return the dose-response result.

        Returns
        -------
        DoseResponseResult
            The estimated dose-response curve and summary statistics.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called.
        """
        if self._result is None:
            raise RuntimeError(
                format_error(
                    "ContinuousTreatmentEffect must be fitted before calling result().",
                    "call .fit() with outcome and treatment_dose data first.",
                )
            )
        return self._result
