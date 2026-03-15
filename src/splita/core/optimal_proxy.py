"""OptimalProxyMetrics — learn optimal weighted proxy for a north star metric.

Finds the linear combination of candidate metrics that maximizes
correlation with the north star metric.

References
----------
.. [1] Jeunen, O. (2024).  "Powerful A/B-Testing Metrics and Where to
       Find Them."
"""

from __future__ import annotations

import numpy as np

from splita._types import ProxyResult
from splita._validation import (
    check_array_like,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


class OptimalProxyMetrics:
    """Learn an optimal weighted combination of candidate metrics.

    Finds weights that maximize the Pearson correlation between the
    composite proxy metric (weighted sum of candidates) and a north
    star metric. This is equivalent to ordinary least squares regression.

    Attributes
    ----------
    weights_ : np.ndarray
        Learned weights for each candidate metric (set after :meth:`fit`).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n = 200
    >>> ns = rng.normal(10, 2, n)
    >>> candidates = np.column_stack([
    ...     ns + rng.normal(0, 1, n),
    ...     rng.normal(5, 1, n),
    ...     ns * 0.5 + rng.normal(0, 0.5, n),
    ... ])
    >>> proxy = OptimalProxyMetrics()
    >>> proxy.fit(candidates, ns)
    >>> result = proxy.result(candidates, ns)
    >>> result.correlation_with_north_star > 0.5
    True
    """

    def __init__(self) -> None:
        self.weights_: np.ndarray
        self._is_fitted = False

    def fit(
        self,
        candidate_metrics: np.ndarray | ArrayLike,
        north_star: ArrayLike,
    ) -> OptimalProxyMetrics:
        """Learn optimal weights via OLS regression.

        Parameters
        ----------
        candidate_metrics : 2-D array-like, shape (n_samples, n_metrics)
            Matrix of candidate metric values.
        north_star : array-like, shape (n_samples,)
            The north star metric to maximize correlation with.

        Returns
        -------
        OptimalProxyMetrics
            The fitted instance (for method chaining).

        Raises
        ------
        ValueError
            If inputs have incompatible shapes or degenerate data.
        """
        X = np.asarray(candidate_metrics, dtype=float)
        y = check_array_like(north_star, "north_star", min_length=2)

        if X.ndim != 2:
            raise ValueError(
                format_error(
                    "`candidate_metrics` must be a 2-D array, got "
                    f"{X.ndim}-D array with shape {X.shape}.",
                    hint="pass a matrix with shape (n_samples, n_metrics).",
                )
            )

        if X.shape[0] != len(y):
            raise ValueError(
                format_error(
                    "`candidate_metrics` and `north_star` must have the same "
                    f"number of rows.",
                    detail=f"candidate_metrics has {X.shape[0]} rows, "
                    f"north_star has {len(y)} elements.",
                )
            )

        if X.shape[1] < 1:
            raise ValueError(
                format_error(
                    "`candidate_metrics` must have at least 1 column.",
                    hint="pass at least one candidate metric.",
                )
            )

        if X.shape[0] <= X.shape[1]:
            raise ValueError(
                format_error(
                    f"`candidate_metrics` must have more rows ({X.shape[0]}) "
                    f"than columns ({X.shape[1]}) for OLS.",
                    hint="provide more samples than candidate metrics.",
                )
            )

        # OLS: w = (X^T X)^{-1} X^T y
        try:
            self.weights_ = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                format_error(
                    "OLS failed — candidate metrics may be collinear.",
                    detail=str(exc),
                    hint="remove redundant candidate metrics.",
                )
            ) from exc

        self._is_fitted = True
        return self

    def transform(self, candidate_metrics: np.ndarray | ArrayLike) -> np.ndarray:
        """Apply learned weights to produce composite proxy values.

        Parameters
        ----------
        candidate_metrics : 2-D array-like, shape (n_samples, n_metrics)
            Matrix of candidate metric values.

        Returns
        -------
        np.ndarray
            Composite proxy metric values.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        """
        if not self._is_fitted:
            raise RuntimeError(
                format_error(
                    "OptimalProxyMetrics must be fitted before calling transform().",
                    detail="weights_ have not been estimated yet.",
                    hint="call fit() first.",
                )
            )

        X = np.asarray(candidate_metrics, dtype=float)
        if X.ndim != 2:
            raise ValueError(
                format_error(
                    "`candidate_metrics` must be a 2-D array, got "
                    f"{X.ndim}-D array with shape {X.shape}.",
                )
            )

        if X.shape[1] != len(self.weights_):
            raise ValueError(
                format_error(
                    f"`candidate_metrics` must have {len(self.weights_)} columns "
                    f"(matching fit), got {X.shape[1]}.",
                )
            )

        return X @ self.weights_

    def result(
        self,
        candidate_metrics: np.ndarray | ArrayLike,
        north_star: ArrayLike,
    ) -> ProxyResult:
        """Compute the proxy and return a full result.

        Parameters
        ----------
        candidate_metrics : 2-D array-like
            Matrix of candidate metric values.
        north_star : array-like
            The north star metric.

        Returns
        -------
        ProxyResult
            Weights, correlation, and proxy values.
        """
        proxy_values = self.transform(candidate_metrics)
        y = check_array_like(north_star, "north_star", min_length=2)

        corr = float(np.corrcoef(proxy_values, y)[0, 1])

        return ProxyResult(
            weights=self.weights_.tolist(),
            correlation_with_north_star=corr,
            optimal_proxy_values=proxy_values.tolist(),
        )
