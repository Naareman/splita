"""Interference experiment analysis (Basse & Feller 2018).

For marketplace/social experiments where users interact across treatment
and control.  Uses Horvitz-Thompson estimation at the cluster level and
accounts for within-cluster correlation.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from splita._types import InterferenceResult
from splita._validation import (
    check_array_like,
    check_in_range,
    check_same_length,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


class InterferenceExperiment:
    """Analyse an experiment with potential interference between units.

    Clusters units and uses Horvitz-Thompson estimation to account for
    within-cluster spillover effects.  The design effect quantifies
    how much the cluster-robust SE exceeds the naive unit-level SE.

    Parameters
    ----------
    outcomes : array-like
        Outcome values for all units.
    treatments : array-like
        Binary treatment assignments (0 or 1) for all units.
    clusters : array-like
        Cluster labels for each unit.
    alpha : float, default 0.05
        Significance level.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n = 200
    >>> clusters = np.repeat(np.arange(20), 10)
    >>> treatments = np.tile([0] * 5 + [1] * 5, 20)
    >>> outcomes = rng.normal(10, 1, n) + treatments * 2.0
    >>> result = InterferenceExperiment(
    ...     outcomes, treatments, clusters
    ... ).run()
    >>> result.n_clusters
    20
    """

    def __init__(
        self,
        outcomes: ArrayLike,
        treatments: ArrayLike,
        clusters: ArrayLike,
        *,
        alpha: float = 0.05,
    ) -> None:
        check_in_range(
            alpha,
            "alpha",
            0.0,
            1.0,
            hint="typical values are 0.05, 0.01, or 0.10",
        )

        self._outcomes = check_array_like(outcomes, "outcomes", min_length=2)
        self._treatments = check_array_like(treatments, "treatments", min_length=2)

        # Validate binary treatments
        unique_t = np.unique(self._treatments)
        if not np.all(np.isin(unique_t, [0.0, 1.0])):
            raise ValueError(
                format_error(
                    "`treatments` must contain only 0 and 1.",
                    f"found unique values: {unique_t.tolist()}.",
                    "encode treatment as 0 (control) and 1 (treatment).",
                )
            )

        # Cluster labels — allow non-numeric
        self._clusters = np.asarray(clusters)
        if self._clusters.ndim != 1:
            raise ValueError(
                format_error(
                    "`clusters` must be a 1-D array.",
                    f"got {self._clusters.ndim}-D array "
                    f"with shape {self._clusters.shape}.",
                )
            )

        check_same_length(self._outcomes, self._treatments, "outcomes", "treatments")
        check_same_length(self._outcomes, self._clusters, "outcomes", "clusters")

        unique_clusters = np.unique(self._clusters)
        if len(unique_clusters) < 2:
            raise ValueError(
                format_error(
                    "Need at least 2 clusters for interference analysis.",
                    f"got {len(unique_clusters)} cluster(s).",
                    "add more clusters to enable cluster-robust inference.",
                )
            )

        self._alpha = alpha

    @staticmethod
    def _compute_icc(values: np.ndarray, clusters: np.ndarray) -> float:
        """Compute the intraclass correlation coefficient (ICC).

        Uses one-way random effects ANOVA: ICC = (MSB - MSW) / (MSB + (k-1)*MSW)
        where k is the average cluster size.

        Parameters
        ----------
        values : np.ndarray
            Outcome values.
        clusters : np.ndarray
            Cluster labels.

        Returns
        -------
        float
            ICC, clipped to [0, 1].
        """
        unique = np.unique(clusters)
        k = len(unique)
        if k < 2:
            return 0.0

        grand_mean = float(np.mean(values))
        cluster_sizes = []
        cluster_means = []
        ssw = 0.0

        for c in unique:
            mask = clusters == c
            group = values[mask]
            n_c = len(group)
            cluster_sizes.append(n_c)
            m_c = float(np.mean(group))
            cluster_means.append(m_c)
            ssw += float(np.sum((group - m_c) ** 2))

        n_total = len(values)
        ssb = sum(
            n * (m - grand_mean) ** 2
            for n, m in zip(cluster_sizes, cluster_means, strict=True)
        )

        df_b = k - 1
        df_w = n_total - k

        msb = ssb / df_b if df_b > 0 else 0.0
        msw = ssw / df_w if df_w > 0 else 0.0

        n_avg = n_total / k
        denom = msb + (n_avg - 1) * msw
        if denom <= 0:
            return 0.0

        icc = (msb - msw) / denom
        return max(0.0, icc)

    def run(self) -> InterferenceResult:
        """Run the interference-aware analysis.

        Returns
        -------
        InterferenceResult
            Horvitz-Thompson estimate with cluster-robust inference.

        Raises
        ------
        ValueError
            If both treatment groups are not represented across clusters.
        """
        outcomes = self._outcomes
        treatments = self._treatments
        clusters = self._clusters
        unique_clusters = np.unique(clusters)
        n_clusters = len(unique_clusters)

        # Compute cluster-level Horvitz-Thompson estimates
        cluster_effects = []
        cluster_sizes = []

        for c in unique_clusters:
            mask = clusters == c
            y_c = outcomes[mask]
            t_c = treatments[mask]
            n_c = len(y_c)
            cluster_sizes.append(n_c)

            trt_mask = t_c == 1.0
            ctrl_mask = t_c == 0.0
            n_trt = int(np.sum(trt_mask))
            n_ctrl = int(np.sum(ctrl_mask))

            if n_trt == 0 or n_ctrl == 0:
                # Skip clusters with only one arm
                continue

            # Horvitz-Thompson: weight by inverse probability within cluster
            ht_trt = float(np.mean(y_c[trt_mask]))
            ht_ctrl = float(np.mean(y_c[ctrl_mask]))
            cluster_effects.append(ht_trt - ht_ctrl)

        if len(cluster_effects) < 2:
            raise ValueError(
                format_error(
                    "Need at least 2 clusters with both treatment and control units.",
                    f"only {len(cluster_effects)} cluster(s) have both arms.",
                    "ensure each cluster contains units from both treatment groups.",
                )
            )

        cluster_effects = np.array(cluster_effects)
        n_valid = len(cluster_effects)

        # ATE: average of cluster-level effects
        ate = float(np.mean(cluster_effects))

        # Cluster-robust SE
        se_cluster = float(np.std(cluster_effects, ddof=1) / np.sqrt(n_valid))

        # Design effect: DEFF = 1 + (m_bar - 1) * ICC
        # where m_bar is the average cluster size and ICC is the intraclass
        # correlation of outcomes within clusters.
        avg_cluster_size = float(np.mean(cluster_sizes))

        # Compute ICC from outcomes within clusters
        icc = self._compute_icc(outcomes, clusters)
        design_effect = 1.0 + (avg_cluster_size - 1.0) * icc

        # Inference using cluster-robust SE
        if se_cluster > 0:
            z_stat = ate / se_cluster
            pvalue = float(2 * norm.sf(abs(z_stat)))
            z_crit = float(norm.ppf(1 - self._alpha / 2))
            ci_lower = ate - z_crit * se_cluster
            ci_upper = ate + z_crit * se_cluster
        else:
            pvalue = 1.0 if ate == 0 else 0.0
            ci_lower = ate
            ci_upper = ate

        return InterferenceResult(
            ate=ate,
            se=se_cluster,
            pvalue=pvalue,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            significant=pvalue < self._alpha,
            n_clusters=n_clusters,
            design_effect=design_effect,
        )
