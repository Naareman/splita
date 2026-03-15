"""Robust mean estimators for A/B tests with heavy-tailed data.

Provides Huber M-estimator, Median of Means, and Catoni estimator
for treatment effect estimation that is robust to outliers.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from splita._types import RobustMeanResult
from splita._validation import check_array_like, check_in_range, check_one_of, format_error

ArrayLike = list | tuple | np.ndarray


def _huber_location(data: np.ndarray, c: float = 1.345, max_iter: int = 50, tol: float = 1e-6) -> float:
    """Compute Huber M-estimate of location via IRLS.

    Parameters
    ----------
    data : np.ndarray
        Input data.
    c : float
        Huber tuning constant (default 1.345 gives 95% efficiency at normal).
    max_iter : int
        Maximum IRLS iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    float
        Huber M-estimate of location.
    """
    mu = float(np.median(data))
    mad = float(np.median(np.abs(data - np.median(data))))
    if mad == 0:
        mad = float(np.std(data))
    if mad == 0:
        return mu

    for _ in range(max_iter):
        residuals = (data - mu) / mad
        weights = np.where(np.abs(residuals) <= c, 1.0, c / np.abs(residuals))
        mu_new = float(np.sum(weights * data) / np.sum(weights))
        if abs(mu_new - mu) < tol:
            mu = mu_new
            break
        mu = mu_new

    return mu


def _median_of_means(data: np.ndarray, k: int | None = None) -> float:
    """Compute median of means estimator.

    Parameters
    ----------
    data : np.ndarray
        Input data.
    k : int, optional
        Number of subgroups. If None, uses ceil(log(n)).

    Returns
    -------
    float
        Median of subgroup means.
    """
    n = len(data)
    if k is None:
        k = max(2, int(math.ceil(math.log(n)))) if n > 1 else 1
    k = min(k, n)

    # Split into k roughly equal subgroups
    indices = np.arange(n)
    subgroups = np.array_split(indices, k)
    means = [float(np.mean(data[sg])) for sg in subgroups if len(sg) > 0]

    return float(np.median(means))


def _catoni_location(data: np.ndarray, alpha: float = 0.05) -> float:
    """Compute Catoni M-estimate of location.

    Uses the influence function psi(x) = sign(x) * log(1 + |x| + x^2/2)
    which provides sub-Gaussian concentration for heavy-tailed data.

    Parameters
    ----------
    data : np.ndarray
        Input data.
    alpha : float
        Confidence level parameter.

    Returns
    -------
    float
        Catoni M-estimate of location.
    """
    n = len(data)
    mu = float(np.median(data))
    scale = float(np.std(data))
    if scale == 0:
        return mu

    # Bandwidth parameter
    v = scale ** 2
    s = math.sqrt(2.0 * math.log(1.0 / alpha) / n) if n > 0 else 1.0

    for _ in range(50):
        centered = (data - mu) / (scale + 1e-10)
        # Catoni influence function
        psi = np.sign(centered) * np.log(1.0 + np.abs(centered) + centered ** 2 / 2.0)
        gradient = float(np.mean(psi))
        mu_new = mu + scale * gradient * 0.5
        if abs(mu_new - mu) < 1e-8:
            mu = mu_new
            break
        mu = mu_new

    return mu


class RobustMeanEstimator:
    """Robust mean estimator for A/B test treatment effects.

    Provides treatment effect estimation that is resistant to outliers
    using Huber M-estimation, Median of Means, or Catoni estimation.

    Parameters
    ----------
    method : ``"huber"``, ``"median_of_means"``, or ``"catoni"``, default ``"huber"``
        Robust estimation method.
    alpha : float, default 0.05
        Significance level for inference.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> control = rng.normal(10, 1, 200)
    >>> treatment = rng.normal(12, 1, 200)
    >>> est = RobustMeanEstimator(method="huber")
    >>> r = est.fit_transform(control, treatment)
    >>> abs(r.ate - 2.0) < 1.0
    True
    """

    def __init__(
        self,
        *,
        method: str = "huber",
        alpha: float = 0.05,
    ) -> None:
        check_one_of(method, "method", ["huber", "median_of_means", "catoni"])
        check_in_range(
            alpha,
            "alpha",
            0.0,
            1.0,
            hint="typical values are 0.05, 0.01, or 0.10",
        )
        self._method = method
        self._alpha = alpha

    def fit_transform(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
    ) -> RobustMeanResult:
        """Estimate robust treatment effect.

        Parameters
        ----------
        control : array-like
            Control group observations.
        treatment : array-like
            Treatment group observations.

        Returns
        -------
        RobustMeanResult
            Robust treatment effect estimate with inference.

        Raises
        ------
        TypeError
            If inputs cannot be converted to numeric arrays.
        ValueError
            If arrays have fewer than 2 elements.
        """
        ctrl = check_array_like(control, "control", min_length=2)
        trt = check_array_like(treatment, "treatment", min_length=2)

        if self._method == "huber":
            mu_ctrl = _huber_location(ctrl)
            mu_trt = _huber_location(trt)
        elif self._method == "median_of_means":
            mu_ctrl = _median_of_means(ctrl)
            mu_trt = _median_of_means(trt)
        else:  # catoni
            mu_ctrl = _catoni_location(ctrl, alpha=self._alpha)
            mu_trt = _catoni_location(trt, alpha=self._alpha)

        ate = mu_trt - mu_ctrl

        # Standard error via bootstrap-like variance estimate
        # Use standard SE formula with robust location estimates
        se_ctrl = float(np.std(ctrl, ddof=1)) / math.sqrt(len(ctrl))
        se_trt = float(np.std(trt, ddof=1)) / math.sqrt(len(trt))
        se = math.sqrt(se_ctrl ** 2 + se_trt ** 2)

        if se > 0:
            z = ate / se
            pvalue = float(2.0 * norm.sf(abs(z)))
            z_crit = float(norm.ppf(1 - self._alpha / 2))
            ci_lower = ate - z_crit * se
            ci_upper = ate + z_crit * se
        else:
            pvalue = 1.0 if ate == 0 else 0.0
            ci_lower = ate
            ci_upper = ate

        return RobustMeanResult(
            ate=ate,
            se=se,
            pvalue=pvalue,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            significant=pvalue < self._alpha,
            method=self._method,
        )
