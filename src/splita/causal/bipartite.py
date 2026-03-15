"""Bipartite experiment analysis (Harshaw et al. 2023, Vinted 2024).

When randomization units differ from outcome units: e.g., buyers are
randomized but seller outcomes are also of interest.  Uses an exposure
mapping to determine which sellers are "exposed" to treatment based on
the fraction of their buying partners who are treated.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from splita._types import BipartiteResult
from splita._validation import (
    check_array_like,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


class BipartiteExperiment:
    """Analyse a bipartite experiment with cross-side exposure mapping.

    In a bipartite experiment, buyers are randomized into treatment or
    control, but outcomes are measured on both buyers and sellers.  The
    seller-side treatment effect is estimated via an exposure mapping:
    a seller is classified as "exposed" if the fraction of their buying
    partners assigned to treatment exceeds a threshold.

    Parameters
    ----------
    exposure_threshold : float, default 0.5
        Fraction of a seller's buyers that must be treated for the
        seller to be classified as "exposed".
    alpha : float, default 0.05
        Significance level.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n_buyers, n_sellers = 50, 20
    >>> buyer_outcomes = rng.normal(10, 1, n_buyers)
    >>> seller_outcomes = rng.normal(8, 1, n_sellers)
    >>> buyer_treatments = rng.binomial(1, 0.5, n_buyers).astype(float)
    >>> graph = (rng.random((n_buyers, n_sellers)) > 0.7).astype(float)
    >>> result = BipartiteExperiment().fit(
    ...     buyer_outcomes, seller_outcomes, buyer_treatments, graph
    ... )
    >>> isinstance(result.n_exposed_sellers, int)
    True
    """

    def __init__(
        self,
        *,
        exposure_threshold: float = 0.5,
        alpha: float = 0.05,
    ) -> None:
        if not 0.0 < exposure_threshold <= 1.0:
            raise ValueError(
                format_error(
                    "`exposure_threshold` must be in (0, 1], got {}.".format(
                        exposure_threshold
                    ),
                    "threshold determines when a seller is considered exposed.",
                    "typical values are 0.3, 0.5, or 0.7.",
                )
            )
        if not 0.0 < alpha < 1.0:
            raise ValueError(
                format_error(
                    "`alpha` must be in (0, 1), got {}.".format(alpha),
                    "alpha controls the significance level.",
                    "typical values are 0.05, 0.01, or 0.10.",
                )
            )
        self._exposure_threshold = exposure_threshold
        self._alpha = alpha

    def fit(
        self,
        buyer_outcomes: ArrayLike,
        seller_outcomes: ArrayLike,
        buyer_treatments: ArrayLike,
        transaction_graph: np.ndarray,
    ) -> BipartiteResult:
        """Estimate buyer-side and seller-side treatment effects.

        Parameters
        ----------
        buyer_outcomes : array-like
            Outcome values for each buyer.
        seller_outcomes : array-like
            Outcome values for each seller.
        buyer_treatments : array-like
            Binary treatment assignments (0/1) for each buyer.
        transaction_graph : np.ndarray
            Binary matrix of shape ``(n_buyers, n_sellers)`` where
            ``transaction_graph[i, j] = 1`` if buyer *i* transacted
            with seller *j*.

        Returns
        -------
        BipartiteResult
            Buyer-side and seller-side treatment effect estimates.

        Raises
        ------
        ValueError
            If inputs are invalid or have incompatible shapes.
        TypeError
            If *transaction_graph* is not a 2-D array.
        """
        y_b = check_array_like(buyer_outcomes, "buyer_outcomes", min_length=2)
        y_s = check_array_like(seller_outcomes, "seller_outcomes", min_length=2)
        t_b = check_array_like(buyer_treatments, "buyer_treatments", min_length=2)

        n_buyers = len(y_b)
        n_sellers = len(y_s)

        if len(t_b) != n_buyers:
            raise ValueError(
                format_error(
                    "`buyer_treatments` must have the same length as `buyer_outcomes`.",
                    f"buyer_outcomes has {n_buyers} elements, "
                    f"buyer_treatments has {len(t_b)}.",
                )
            )

        unique_t = np.unique(t_b)
        if not np.all(np.isin(unique_t, [0.0, 1.0])):
            raise ValueError(
                format_error(
                    "`buyer_treatments` must contain only 0 and 1.",
                    f"found unique values: {unique_t.tolist()}.",
                    "encode treatment as 0 (control) and 1 (treatment).",
                )
            )

        n_trt = int(np.sum(t_b == 1.0))
        n_ctrl = int(np.sum(t_b == 0.0))
        if n_trt == 0 or n_ctrl == 0:
            raise ValueError(
                format_error(
                    "Need both treatment and control buyers.",
                    f"got {n_trt} treated and {n_ctrl} control buyers.",
                )
            )

        # Validate transaction graph
        if not isinstance(transaction_graph, np.ndarray):
            raise TypeError(
                format_error(
                    "`transaction_graph` must be a numpy ndarray.",
                    f"got type {type(transaction_graph).__name__}.",
                )
            )

        if transaction_graph.ndim != 2:
            raise ValueError(
                format_error(
                    "`transaction_graph` must be a 2-D array.",
                    f"got {transaction_graph.ndim}-D array with shape "
                    f"{transaction_graph.shape}.",
                )
            )

        if transaction_graph.shape != (n_buyers, n_sellers):
            raise ValueError(
                format_error(
                    "`transaction_graph` must have shape (n_buyers, n_sellers).",
                    f"expected ({n_buyers}, {n_sellers}), "
                    f"got {transaction_graph.shape}.",
                    "rows = buyers, columns = sellers.",
                )
            )

        G = transaction_graph.astype(float)

        # --- Buyer-side effect (standard difference in means) ---
        buyer_trt_mean = float(np.mean(y_b[t_b == 1.0]))
        buyer_ctrl_mean = float(np.mean(y_b[t_b == 0.0]))
        buyer_side_effect = buyer_trt_mean - buyer_ctrl_mean

        # --- Seller-side effect via exposure mapping ---
        # For each seller, compute fraction of their buyers that are treated
        buyer_counts = G.sum(axis=0)  # total buyers per seller
        treated_buyer_counts = (G.T @ t_b)  # treated buyers per seller

        # Avoid division by zero for sellers with no buyers
        seller_exposure = np.zeros(n_sellers)
        has_buyers = buyer_counts > 0
        seller_exposure[has_buyers] = (
            treated_buyer_counts[has_buyers] / buyer_counts[has_buyers]
        )

        # Classify sellers as exposed or unexposed
        exposed_mask = seller_exposure >= self._exposure_threshold
        unexposed_mask = (~exposed_mask) & has_buyers

        n_exposed = int(np.sum(exposed_mask))
        n_unexposed = int(np.sum(unexposed_mask))

        if n_exposed == 0 or n_unexposed == 0:
            # Can't estimate seller-side effect without both groups
            return BipartiteResult(
                buyer_side_effect=buyer_side_effect,
                seller_side_effect=0.0,
                cross_side_pvalue=1.0,
                n_exposed_sellers=n_exposed,
            )

        # Seller-side effect: exposed vs unexposed sellers
        exposed_mean = float(np.mean(y_s[exposed_mask]))
        unexposed_mean = float(np.mean(y_s[unexposed_mask]))
        seller_side_effect = exposed_mean - unexposed_mean

        # SE for seller-side effect (Welch approximation)
        se_exposed = float(np.std(y_s[exposed_mask], ddof=1) / np.sqrt(n_exposed))
        se_unexposed = float(
            np.std(y_s[unexposed_mask], ddof=1) / np.sqrt(n_unexposed)
        )
        se_seller = float(np.sqrt(se_exposed**2 + se_unexposed**2))

        if se_seller > 0:
            z = seller_side_effect / se_seller
            cross_side_pvalue = float(2 * norm.sf(abs(z)))
        else:
            cross_side_pvalue = 1.0 if seller_side_effect == 0 else 0.0

        return BipartiteResult(
            buyer_side_effect=buyer_side_effect,
            seller_side_effect=seller_side_effect,
            cross_side_pvalue=cross_side_pvalue,
            n_exposed_sellers=n_exposed,
        )
