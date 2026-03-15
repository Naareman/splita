"""Pre-experiment validation — metric sensitivity and variance estimation.

Tools for evaluating whether a metric is suitable for experimentation
before running an actual test.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import norm
from scipy.stats import skew as scipy_skew

from splita._types import MetricSensitivityResult, VarianceEstimateResult
from splita._utils import ensure_rng
from splita._validation import (
    check_array_like,
    check_in_range,
    check_is_integer,
    check_positive,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


class MetricSensitivity:
    """Estimate whether a metric can detect a given effect size.

    Simulates experiments at the specified MDE using the variance
    estimated from historical data, and reports achievable power.

    Parameters
    ----------
    n_simulations : int, default 500
        Number of Monte Carlo simulations.
    alpha : float, default 0.05
        Significance level.
    random_state : int, Generator, or None, default None
        Seed for reproducibility.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> data = rng.normal(10.0, 2.0, size=1000)
    >>> ms = MetricSensitivity(n_simulations=200, random_state=0)
    >>> result = ms.run(data, mde=0.5)
    >>> 0.0 <= result.estimated_power <= 1.0
    True
    """

    def __init__(
        self,
        *,
        n_simulations: int = 500,
        alpha: float = 0.05,
        random_state: int | np.random.Generator | None = None,
    ):
        check_is_integer(n_simulations, "n_simulations", min_value=10)
        check_in_range(
            alpha,
            "alpha",
            0.0,
            1.0,
            hint="typical values are 0.05, 0.01, or 0.10",
        )
        self._n_simulations = int(n_simulations)
        self._alpha = alpha
        self._rng = ensure_rng(random_state)

    def run(
        self,
        historical_data: ArrayLike,
        mde: float,
    ) -> MetricSensitivityResult:
        """Run the metric sensitivity analysis.

        Parameters
        ----------
        historical_data : array-like
            Historical metric observations used to estimate variance.
        mde : float
            Minimum detectable effect (absolute).

        Returns
        -------
        MetricSensitivityResult
            Frozen dataclass with sensitivity analysis results.

        Raises
        ------
        ValueError
            If data is too short or MDE is not positive.
        """
        data = check_array_like(historical_data, "historical_data", min_length=10)
        check_positive(mde, "mde")

        estimated_std = float(np.std(data, ddof=1))
        n_data = len(data)

        if estimated_std == 0.0:
            return MetricSensitivityResult(
                estimated_std=0.0,
                estimated_power=1.0,
                recommended_n=2,
                is_sensitive=True,
                mde=mde,
                n_simulations=self._n_simulations,
                alpha=self._alpha,
            )

        # Simulate experiments at n = len(data) per group
        n_per_group = n_data
        rejections = 0

        for _ in range(self._n_simulations):
            ctrl = self._rng.normal(0, estimated_std, size=n_per_group)
            trt = self._rng.normal(mde, estimated_std, size=n_per_group)

            se = math.sqrt(np.var(ctrl, ddof=1) / n_per_group + np.var(trt, ddof=1) / n_per_group)
            if se > 0:
                z = (np.mean(trt) - np.mean(ctrl)) / se
                pval = 2.0 * norm.sf(abs(z))
                if pval < self._alpha:
                    rejections += 1

        estimated_power = rejections / self._n_simulations

        # Recommended n for 80% power (analytic approximation)
        z_alpha = float(norm.ppf(1 - self._alpha / 2))
        z_beta = float(norm.ppf(0.80))
        recommended_n = max(
            2,
            math.ceil(2 * ((z_alpha + z_beta) * estimated_std / mde) ** 2),
        )

        # Consider sensitive if recommended_n <= 1_000_000
        is_sensitive = recommended_n <= 1_000_000

        return MetricSensitivityResult(
            estimated_std=estimated_std,
            estimated_power=estimated_power,
            recommended_n=recommended_n,
            is_sensitive=is_sensitive,
            mde=mde,
            n_simulations=self._n_simulations,
            alpha=self._alpha,
        )


class VarianceEstimator:
    """Estimate distributional properties of a metric.

    Computes mean, standard deviation, skewness, kurtosis, and
    percentiles, then flags potential issues and makes recommendations
    for experiment design.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> data = rng.normal(10.0, 2.0, size=500)
    >>> ve = VarianceEstimator()
    >>> result = ve.fit(data).result()
    >>> result.is_heavy_tailed
    False
    """

    def __init__(self) -> None:
        self._fitted = False
        self._data: np.ndarray | None = None

    def fit(self, data: ArrayLike) -> VarianceEstimator:
        """Fit the variance estimator on observed data.

        Parameters
        ----------
        data : array-like
            Metric observations.

        Returns
        -------
        VarianceEstimator
            The fitted estimator (self).

        Raises
        ------
        ValueError
            If data has fewer than 4 elements.
        """
        self._data = check_array_like(data, "data", min_length=4)
        self._fitted = True
        return self

    def result(self) -> VarianceEstimateResult:
        """Return the variance estimation result.

        Returns
        -------
        VarianceEstimateResult
            Frozen dataclass with distributional properties and
            recommendations.

        Raises
        ------
        RuntimeError
            If the estimator has not been fitted.
        """
        if not self._fitted or self._data is None:
            raise RuntimeError(
                format_error(
                    "VarianceEstimator must be fitted before calling result().",
                    "call .fit() first.",
                )
            )

        data = self._data
        mean = float(np.mean(data))
        std = float(np.std(data, ddof=1))
        skewness = float(scipy_skew(data))
        kurt = float(scipy_kurtosis(data, fisher=True))

        percentiles = {
            "p5": float(np.percentile(data, 5)),
            "p25": float(np.percentile(data, 25)),
            "p50": float(np.percentile(data, 50)),
            "p75": float(np.percentile(data, 75)),
            "p95": float(np.percentile(data, 95)),
        }

        is_heavy_tailed = kurt > 5.0
        is_skewed = abs(skewness) > 2.0

        recommendations: list[str] = []
        if is_heavy_tailed:
            recommendations.append(
                "Heavy-tailed distribution detected (kurtosis > 5). "
                "Consider outlier handling (winsorization or trimming) "
                "before running the experiment."
            )
        if is_skewed:
            recommendations.append(
                "High skewness detected (|skew| > 2). "
                "Consider using Mann-Whitney test or bootstrap instead "
                "of a standard t-test."
            )
        if not recommendations:
            recommendations.append("Distribution appears well-behaved for standard A/B testing.")

        return VarianceEstimateResult(
            mean=mean,
            std=std,
            skewness=skewness,
            kurtosis=kurt,
            percentiles=percentiles,
            is_heavy_tailed=is_heavy_tailed,
            is_skewed=is_skewed,
            recommendations=recommendations,
        )
