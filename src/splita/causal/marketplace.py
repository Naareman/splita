"""Marketplace experiment analysis (Bajari et al. Management Science 2023).

Framework for buyer vs seller side randomization in two-sided marketplaces.
Computes bias estimates based on market balance and recommends the better
randomization side.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.stats import norm

from splita._types import MarketplaceResult
from splita._validation import (
    check_array_like,
    check_same_length,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


class MarketplaceExperiment:
    """Analyse a marketplace experiment with buyer or seller randomization.

    In two-sided marketplaces, randomizing on one side creates interference
    on the other side.  This class estimates the bias from that interference
    and recommends the better randomization side.

    Parameters
    ----------
    alpha : float, default 0.05
        Significance level.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n = 200
    >>> outcomes = rng.normal(10, 1, n) + np.repeat([0, 2], 100)
    >>> treatments = np.repeat([0, 1], 100)
    >>> clusters = np.tile(np.arange(20), 10)
    >>> result = MarketplaceExperiment().analyze(
    ...     outcomes, treatments, side="buyer", clusters=clusters
    ... )
    >>> isinstance(result.ate, float)
    True
    """

    def __init__(self, *, alpha: float = 0.05) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(
                format_error(
                    f"`alpha` must be in (0, 1), got {alpha}.",
                    "alpha controls the significance level.",
                    "typical values are 0.05, 0.01, or 0.10.",
                )
            )
        self._alpha = alpha

    def analyze(
        self,
        outcomes: ArrayLike,
        treatments: ArrayLike,
        *,
        side: Literal["buyer", "seller"],
        clusters: ArrayLike,
    ) -> MarketplaceResult:
        """Analyse a marketplace experiment.

        Parameters
        ----------
        outcomes : array-like
            Outcome values for all units.
        treatments : array-like
            Binary treatment assignments (0 or 1).
        side : ``'buyer'`` or ``'seller'``
            Which side of the marketplace was randomized.
        clusters : array-like
            Cluster labels (e.g., geographic region, product category).

        Returns
        -------
        MarketplaceResult
            ATE estimate with bias and design effect diagnostics.

        Raises
        ------
        ValueError
            If inputs are invalid or insufficient.
        """
        if side not in ("buyer", "seller"):
            raise ValueError(
                format_error(
                    f"`side` must be 'buyer' or 'seller', got {side!r}.",
                    "specify which side of the marketplace was randomized.",
                )
            )

        y = check_array_like(outcomes, "outcomes", min_length=4)
        t = check_array_like(treatments, "treatments", min_length=4)
        check_same_length(y, t, "outcomes", "treatments")

        unique_t = np.unique(t)
        if not np.all(np.isin(unique_t, [0.0, 1.0])):
            raise ValueError(
                format_error(
                    "`treatments` must contain only 0 and 1.",
                    f"found unique values: {unique_t.tolist()}.",
                    "encode treatment as 0 (control) and 1 (treatment).",
                )
            )

        c = np.asarray(clusters)
        if c.ndim != 1:  # pragma: no cover
            raise ValueError(
                format_error(
                    "`clusters` must be a 1-D array.",
                    f"got {c.ndim}-D array with shape {c.shape}.",
                )
            )
        check_same_length(y, c, "outcomes", "clusters")

        unique_clusters = np.unique(c)
        if len(unique_clusters) < 2:
            raise ValueError(
                format_error(
                    "Need at least 2 clusters for marketplace analysis.",
                    f"got {len(unique_clusters)} cluster(s).",
                )
            )

        # Compute cluster-level treatment effects
        cluster_effects = []
        cluster_sizes = []
        cluster_treatment_fracs = []

        for clust in unique_clusters:
            mask = c == clust
            y_c = y[mask]
            t_c = t[mask]
            n_c = len(y_c)
            cluster_sizes.append(n_c)

            trt_mask = t_c == 1.0
            ctrl_mask = t_c == 0.0
            n_trt = int(np.sum(trt_mask))
            n_ctrl = int(np.sum(ctrl_mask))
            cluster_treatment_fracs.append(n_trt / n_c if n_c > 0 else 0.0)

            if n_trt > 0 and n_ctrl > 0:
                effect = float(np.mean(y_c[trt_mask]) - np.mean(y_c[ctrl_mask]))
                cluster_effects.append(effect)

        if len(cluster_effects) < 2:  # pragma: no cover
            raise ValueError(
                format_error(
                    "Need at least 2 clusters with both treatment and control.",
                    f"only {len(cluster_effects)} cluster(s) have both arms.",
                )
            )

        cluster_effects_arr = np.array(cluster_effects)
        ate = float(np.mean(cluster_effects_arr))

        # Cluster-robust standard error
        se = float(np.std(cluster_effects_arr, ddof=1) / np.sqrt(len(cluster_effects_arr)))

        # p-value
        if se > 0:
            z = ate / se
            pvalue = float(2 * norm.sf(abs(z)))
        else:
            pvalue = 1.0 if ate == 0 else 0.0  # pragma: no cover

        # Bias estimate: based on treatment fraction imbalance across clusters
        # Higher variance in treatment fractions = more marketplace bias
        frac_arr = np.array(cluster_treatment_fracs)
        treatment_frac_var = float(np.var(frac_arr))
        overall_std = float(np.std(y))
        estimated_bias = treatment_frac_var * overall_std

        # Design effect: DEFF = 1 + (m_bar - 1) * ICC
        avg_cluster_size = float(np.mean(cluster_sizes))
        icc = self._compute_icc(y, c)
        design_effect = 1.0 + (avg_cluster_size - 1.0) * icc

        # Recommend the other side if bias is large relative to SE
        if estimated_bias > se * 0.5:
            recommended_side = "seller" if side == "buyer" else "buyer"  # pragma: no cover
        else:
            recommended_side = side

        return MarketplaceResult(
            ate=ate,
            se=se,
            pvalue=pvalue,
            estimated_bias=estimated_bias,
            design_effect=design_effect,
            recommended_side=recommended_side,
        )

    @staticmethod
    def _compute_icc(values: np.ndarray, clusters: np.ndarray) -> float:
        """Compute intraclass correlation coefficient.

        Parameters
        ----------
        values : np.ndarray
            Outcome values.
        clusters : np.ndarray
            Cluster labels.

        Returns
        -------
        float
            ICC clipped to [0, 1].
        """
        unique = np.unique(clusters)
        k = len(unique)
        if k < 2:
            return 0.0  # pragma: no cover

        grand_mean = float(np.mean(values))
        cluster_sizes = []
        ssw = 0.0

        for clust in unique:
            mask = clusters == clust
            group = values[mask]
            n_c = len(group)
            cluster_sizes.append(n_c)
            m_c = float(np.mean(group))
            ssw += float(np.sum((group - m_c) ** 2))

        n_total = len(values)
        ssb = sum(
            n * (float(np.mean(values[clusters == clust])) - grand_mean) ** 2
            for n, clust in zip(cluster_sizes, unique, strict=True)
        )

        df_b = k - 1
        df_w = n_total - k
        msb = ssb / df_b if df_b > 0 else 0.0
        msw = ssw / df_w if df_w > 0 else 0.0

        n_avg = n_total / k
        denom = msb + (n_avg - 1) * msw
        if denom <= 0:  # pragma: no cover
            return 0.0

        icc = (msb - msw) / denom
        return max(0.0, icc)
