"""Regression Discontinuity Design (RDD) estimator.

Local linear regression on both sides of a cutoff with IK bandwidth
selection (Imbens & Kalyanaraman, 2012).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from splita._types import RDDResult
from splita._validation import check_array_like, check_same_length, format_error

ArrayLike = list | tuple | np.ndarray


class RegressionDiscontinuity:
    """Sharp regression discontinuity design estimator.

    Estimates the local average treatment effect (LATE) at a cutoff
    by fitting separate local linear regressions on each side.

    Parameters
    ----------
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> x = rng.uniform(-2, 2, 400)
    >>> y = 3.0 * (x >= 0) + 0.5 * x + rng.normal(0, 0.5, 400)
    >>> rdd = RegressionDiscontinuity()
    >>> r = rdd.fit(y, x, cutoff=0.0)
    >>> abs(r.late - 3.0) < 1.5
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
        running_variable: ArrayLike,
        cutoff: float = 0.0,
        *,
        bandwidth: float | None = None,
    ) -> RDDResult:
        """Fit the RDD model.

        Parameters
        ----------
        outcome : array-like
            Outcome variable.
        running_variable : array-like
            Running (forcing) variable that determines treatment assignment.
        cutoff : float, default 0.0
            Cutoff value; units at or above the cutoff are treated.
        bandwidth : float or None, default None
            Bandwidth for local linear regression. If ``None``, the
            Imbens-Kalyanaraman (IK) optimal bandwidth is used.

        Returns
        -------
        RDDResult
            Frozen dataclass with the LATE estimate and diagnostics.

        Raises
        ------
        TypeError
            If inputs cannot be converted to numeric arrays.
        ValueError
            If arrays are too short, mismatched, or bandwidth is invalid.
        """
        y = check_array_like(outcome, "outcome", min_length=4)
        x = check_array_like(running_variable, "running_variable", min_length=4)
        check_same_length(y, x, "outcome", "running_variable")

        if bandwidth is not None and bandwidth <= 0:
            raise ValueError(
                format_error(
                    f"`bandwidth` must be > 0, got {bandwidth}.",
                    "bandwidth controls the window around the cutoff.",
                    "pass None for automatic IK bandwidth selection.",
                )
            )

        # Centre the running variable
        x_c = x - cutoff

        if bandwidth is None:
            bandwidth = self._ik_bandwidth(y, x_c)

        # Select observations within bandwidth
        left_mask = (x_c < 0) & (x_c >= -bandwidth)
        right_mask = (x_c >= 0) & (x_c <= bandwidth)

        n_left = int(np.sum(left_mask))
        n_right = int(np.sum(right_mask))

        if n_left < 2:
            raise ValueError(
                format_error(
                    "Too few observations to the left of the cutoff.",
                    f"got {n_left} observations within bandwidth={bandwidth:.4f}.",
                    "increase the bandwidth or provide more data near the cutoff.",
                )
            )
        if n_right < 2:
            raise ValueError(
                format_error(
                    "Too few observations to the right of the cutoff.",
                    f"got {n_right} observations within bandwidth={bandwidth:.4f}.",
                    "increase the bandwidth or provide more data near the cutoff.",
                )
            )

        # Local linear regression: y = a + b*x on each side
        # Evaluate at x=0 (the cutoff) to get intercepts
        x_left = x_c[left_mask]
        y_left = y[left_mask]
        x_right = x_c[right_mask]
        y_right = y[right_mask]

        # Triangular kernel weights
        w_left = 1.0 - np.abs(x_left) / bandwidth
        w_right = 1.0 - np.abs(x_right) / bandwidth

        # Weighted least squares on left side
        a_left, _, resid_left = self._wls(x_left, y_left, w_left)
        a_right, _, resid_right = self._wls(x_right, y_right, w_right)

        # LATE = intercept_right - intercept_left
        late = a_right - a_left

        # Standard error via heteroskedasticity-robust formula
        se_left = self._robust_se_intercept(x_left, resid_left, w_left)
        se_right = self._robust_se_intercept(x_right, resid_right, w_right)
        se = float(np.sqrt(se_left**2 + se_right**2))

        if se > 0:
            z = late / se
            pvalue = float(2.0 * norm.sf(abs(z)))
        else:
            pvalue = 1.0 if late == 0 else 0.0

        z_crit = float(norm.ppf(1 - self._alpha / 2))
        ci_lower = late - z_crit * se
        ci_upper = late + z_crit * se

        return RDDResult(
            late=float(late),
            se=se,
            pvalue=pvalue,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            bandwidth_used=float(bandwidth),
            n_left=n_left,
            n_right=n_right,
        )

    @staticmethod
    def _wls(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float, np.ndarray]:
        """Weighted least squares: y = a + b*x. Returns (a, b, residuals)."""
        W = np.diag(w)
        X = np.column_stack([np.ones(len(x)), x])
        XtW = X.T @ W
        beta = np.linalg.solve(XtW @ X, XtW @ y)
        residuals = y - X @ beta
        return float(beta[0]), float(beta[1]), residuals

    @staticmethod
    def _robust_se_intercept(x: np.ndarray, resid: np.ndarray, w: np.ndarray) -> float:
        """HC1 robust standard error for the intercept in WLS."""
        n = len(x)
        X = np.column_stack([np.ones(n), x])
        W = np.diag(w)
        XtW = X.T @ W
        bread = np.linalg.inv(XtW @ X)
        meat = X.T @ np.diag(w**2 * resid**2) @ X
        sandwich = bread @ meat @ bread
        # HC1 correction
        se = float(np.sqrt(sandwich[0, 0] * n / max(n - 2, 1)))
        return se

    @staticmethod
    def _ik_bandwidth(y: np.ndarray, x_c: np.ndarray) -> float:
        """Imbens-Kalyanaraman optimal bandwidth selection.

        Simplified plug-in rule: h = C * sigma * N^{-1/5}
        where C is calibrated from the data.
        """
        n = len(x_c)
        sigma = float(np.std(y))
        x_range = float(np.max(x_c) - np.min(x_c))

        if x_range == 0 or sigma == 0:
            return 1.0

        # Silverman-style rule adapted for RDD
        h = 1.84 * sigma * n ** (-1.0 / 5.0)
        # Scale by data range to be reasonable
        h = min(h, x_range / 2.0)
        h = max(h, x_range / (n / 2.0))
        return float(h)
