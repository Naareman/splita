"""ClusterBootstrap -- Cluster-level bootstrap for ratio metrics.

Resamples at the cluster (user) level to correctly account for
within-cluster correlation.  Suitable for ratio metrics where
each user contributes multiple observations.

References
----------
.. [1] Cameron, A. C., Gelbach, J. B. & Miller, D. L. "Bootstrap-Based
       Improvements for Inference with Clustered Errors." Review of
       Economics and Statistics, 90(3), 2008.
"""

from __future__ import annotations

import numpy as np

from splita._types import ClusterBootstrapResult
from splita._validation import (
    check_array_like,
    check_in_range,
    check_is_integer,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


class ClusterBootstrap:
    """Cluster-level bootstrap inference for A/B tests.

    Resamples entire clusters (e.g., users) rather than individual
    observations to preserve within-cluster correlation structure.

    Parameters
    ----------
    n_bootstrap : int, default 5000
        Number of bootstrap resamples.
    alpha : float, default 0.05
        Significance level for the confidence interval.
    random_state : int or None, default None
        Seed for reproducibility.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n_users = 100
    >>> ctrl = rng.normal(10, 2, n_users * 5)
    >>> trt = rng.normal(11, 2, n_users * 5)
    >>> ctrl_clusters = np.repeat(np.arange(n_users), 5)
    >>> trt_clusters = np.repeat(np.arange(n_users), 5)
    >>> result = ClusterBootstrap(random_state=42).run(ctrl, trt, ctrl_clusters, trt_clusters)
    >>> result.significant
    True
    """

    def __init__(
        self,
        *,
        n_bootstrap: int = 5000,
        alpha: float = 0.05,
        random_state: int | None = None,
    ) -> None:
        check_is_integer(n_bootstrap, "n_bootstrap", min_value=100)
        check_in_range(
            alpha,
            "alpha",
            0.0,
            1.0,
            hint="typical values are 0.05, 0.01, or 0.10",
        )
        self._n_bootstrap = int(n_bootstrap)
        self._alpha = alpha
        self._rng = np.random.default_rng(random_state)

    @staticmethod
    def _validate_clusters(clusters: ArrayLike, name: str, expected_len: int) -> np.ndarray:
        """Validate and convert cluster labels."""
        if not isinstance(clusters, (list, tuple, np.ndarray)):
            raise TypeError(
                format_error(
                    f"`{name}` must be array-like (list, tuple, or ndarray), "
                    f"got type {type(clusters).__name__}.",
                )
            )
        arr = np.asarray(clusters)
        if arr.ndim != 1:
            raise ValueError(
                format_error(
                    f"`{name}` must be a 1-D array, got {arr.ndim}-D.",
                    hint="pass a flat list or 1-D array of cluster IDs.",
                )
            )
        if len(arr) != expected_len:
            raise ValueError(
                format_error(
                    f"`{name}` must have the same length as its data array.",
                    detail=f"expected {expected_len} elements, got {len(arr)}.",
                )
            )
        return arr

    @staticmethod
    def _cluster_means(values: np.ndarray, clusters: np.ndarray) -> np.ndarray:
        """Compute mean value per cluster."""
        unique = np.unique(clusters)
        means = np.empty(len(unique))
        for i, c in enumerate(unique):
            means[i] = values[clusters == c].mean()
        return means

    def run(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
        control_clusters: ArrayLike,
        treatment_clusters: ArrayLike,
    ) -> ClusterBootstrapResult:
        """Run cluster bootstrap and return results.

        Parameters
        ----------
        control : array-like
            Control group observations.
        treatment : array-like
            Treatment group observations.
        control_clusters : array-like
            Cluster IDs for each control observation.
        treatment_clusters : array-like
            Cluster IDs for each treatment observation.

        Returns
        -------
        ClusterBootstrapResult
            Frozen dataclass with ATE, SE, CI, and p-value.
        """
        ctrl = check_array_like(control, "control", min_length=2)
        trt = check_array_like(treatment, "treatment", min_length=2)
        ctrl_cl = self._validate_clusters(control_clusters, "control_clusters", len(ctrl))
        trt_cl = self._validate_clusters(treatment_clusters, "treatment_clusters", len(trt))

        # Compute cluster-level means
        ctrl_means = self._cluster_means(ctrl, ctrl_cl)
        trt_means = self._cluster_means(trt, trt_cl)

        n_ctrl_clusters = len(ctrl_means)
        n_trt_clusters = len(trt_means)

        if n_ctrl_clusters < 2:
            raise ValueError(
                format_error(
                    "Control must have at least 2 clusters.",
                    detail=f"got {n_ctrl_clusters} cluster(s).",
                )
            )
        if n_trt_clusters < 2:
            raise ValueError(
                format_error(
                    "Treatment must have at least 2 clusters.",
                    detail=f"got {n_trt_clusters} cluster(s).",
                )
            )

        observed_ate = float(trt_means.mean() - ctrl_means.mean())

        # Bootstrap
        boot_ates = np.empty(self._n_bootstrap)
        for b in range(self._n_bootstrap):
            idx_c = self._rng.choice(n_ctrl_clusters, size=n_ctrl_clusters, replace=True)
            idx_t = self._rng.choice(n_trt_clusters, size=n_trt_clusters, replace=True)
            boot_ates[b] = trt_means[idx_t].mean() - ctrl_means[idx_c].mean()

        se = float(np.std(boot_ates, ddof=1))
        # Percentile CI
        ci_lower = float(np.percentile(boot_ates, 100 * self._alpha / 2))
        ci_upper = float(np.percentile(boot_ates, 100 * (1 - self._alpha / 2)))

        # Two-sided p-value: fraction of bootstrap ATEs on opposite side of 0
        if observed_ate >= 0:
            pvalue = float(2 * np.mean(boot_ates <= 0))
        else:
            pvalue = float(2 * np.mean(boot_ates >= 0))
        pvalue = min(pvalue, 1.0)

        return ClusterBootstrapResult(
            ate=observed_ate,
            se=se,
            pvalue=pvalue,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            significant=pvalue < self._alpha,
            n_clusters=n_ctrl_clusters + n_trt_clusters,
        )
