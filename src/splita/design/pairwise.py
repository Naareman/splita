"""Pairwise matching design for variance reduction.

Pairs similar units using Mahalanobis distance, then randomly assigns
one of each pair to treatment and one to control.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from splita._types import PairwiseDesignResult
from splita._utils import ensure_rng
from splita._validation import format_error

ArrayLike = list | tuple | np.ndarray


class PairwiseDesign:
    """Design an experiment using pairwise matching.

    Pairs similar units based on Mahalanobis distance over covariates,
    then randomly assigns one member of each pair to treatment. This
    reduces variance at the design phase without requiring CUPED-style
    post-hoc adjustment.

    Parameters
    ----------
    random_state : int, Generator, or None, default None
        Seed for reproducibility of treatment assignment.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> X = rng.normal(size=(20, 3))
    >>> result = PairwiseDesign(random_state=42).assign(X)
    >>> result.n_pairs
    10
    """

    def __init__(
        self,
        *,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self._rng = ensure_rng(random_state)

    def assign(self, X: np.ndarray) -> PairwiseDesignResult:
        """Assign units to treatment/control via pairwise matching.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_units, n_features).  Each row is a unit.

        Returns
        -------
        PairwiseDesignResult
            Treatment assignments, pairs, and balance diagnostics.

        Raises
        ------
        ValueError
            If fewer than 2 units are provided.
        """
        X_arr = np.asarray(X, dtype="float64")
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.ndim != 2:
            raise ValueError(
                format_error(
                    "`X` must be a 1-D or 2-D array.",
                    f"got {X_arr.ndim}-D array with shape {X_arr.shape}.",
                )
            )

        n = X_arr.shape[0]
        if n < 2:
            raise ValueError(
                format_error(
                    "`X` must have at least 2 rows for pairing.",
                    f"got {n} row(s).",
                    "provide at least 2 units.",
                )
            )

        # Compute Mahalanobis distance matrix
        # Use covariance with regularisation for numerical stability
        cov = np.cov(X_arr, rowvar=False)
        if cov.ndim == 0:
            # Single feature
            cov = np.array([[float(cov)]])

        # Regularise
        cov += np.eye(cov.shape[0]) * 1e-6

        try:
            VI = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            # Fallback to euclidean if covariance is singular
            VI = np.eye(cov.shape[0])

        dist_matrix = cdist(X_arr, X_arr, metric="mahalanobis", VI=VI)

        # Greedy matching: find pairs with smallest distance
        # Set diagonal to inf to avoid self-matching
        np.fill_diagonal(dist_matrix, np.inf)

        matched = set()
        pairs: list[tuple[int, int]] = []

        # Get all pairwise distances sorted
        n_pairs_target = n // 2
        flat_indices = np.argsort(dist_matrix, axis=None)

        for flat_idx in flat_indices:
            if len(pairs) >= n_pairs_target:
                break
            i, j = divmod(int(flat_idx), n)
            if i in matched or j in matched:
                continue
            if i == j:
                continue
            pairs.append((min(i, j), max(i, j)))
            matched.add(i)
            matched.add(j)

        # Assign treatment within each pair
        assignments = np.full(n, -1, dtype=int)  # -1 for unmatched
        final_pairs: list[tuple[int, int]] = []

        for i, j in pairs:
            if self._rng.random() < 0.5:
                assignments[i] = 0  # control
                assignments[j] = 1  # treatment
                final_pairs.append((i, j))
            else:
                assignments[i] = 1  # treatment
                assignments[j] = 0  # control
                final_pairs.append((j, i))

        # Handle odd unit: assign to control
        unmatched = [idx for idx in range(n) if assignments[idx] == -1]
        for idx in unmatched:
            assignments[idx] = 0

        # Compute balance score: max standardised mean difference
        balance_score = self._compute_balance(X_arr, assignments)

        return PairwiseDesignResult(
            assignments=assignments.tolist(),
            pairs=final_pairs,
            balance_score=balance_score,
            n_pairs=len(final_pairs),
        )

    @staticmethod
    def _compute_balance(X: np.ndarray, assignments: np.ndarray) -> float:
        """Compute the max standardised mean difference across features.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        assignments : np.ndarray
            Treatment assignments (0/1).

        Returns
        -------
        float
            Maximum absolute standardised mean difference.
        """
        ctrl_mask = assignments == 0
        trt_mask = assignments == 1

        if not np.any(ctrl_mask) or not np.any(trt_mask):
            return 0.0

        X_ctrl = X[ctrl_mask]
        X_trt = X[trt_mask]

        max_smd = 0.0
        for col in range(X.shape[1]):
            mean_diff = abs(float(np.mean(X_trt[:, col]) - np.mean(X_ctrl[:, col])))
            pooled_std = float(
                np.sqrt((np.var(X_ctrl[:, col], ddof=1) + np.var(X_trt[:, col], ddof=1)) / 2)
            )
            smd = mean_diff / pooled_std if pooled_std > 1e-12 else 0.0
            max_smd = max(max_smd, smd)

        return max_smd
