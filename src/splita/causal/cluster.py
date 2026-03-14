"""Cluster-randomized experiment analysis.

Handles experiments where randomization occurs at the cluster level
(e.g., cities, stores, schools) but observations are at the unit level.
Uses cluster-robust standard errors by collapsing to cluster means.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import t as t_dist

from splita._types import ClusterResult
from splita._validation import (
    check_array_like,
    check_in_range,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


class ClusterExperiment:
    """Analyse a cluster-randomized experiment.

    Randomization occurs at the cluster level; individual observations are
    nested within clusters.  The analysis collapses data to cluster-level
    means and performs a two-sample t-test on the cluster means, yielding
    cluster-robust inference.

    Parameters
    ----------
    control : array-like
        Observations from the control group (unit level).
    treatment : array-like
        Observations from the treatment group (unit level).
    control_clusters : array-like
        Cluster labels for each control observation.
    treatment_clusters : array-like
        Cluster labels for each treatment observation.
    alpha : float, default 0.05
        Significance level.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> ctrl = rng.normal(10, 2, 100)
    >>> trt = rng.normal(11, 2, 100)
    >>> ctrl_clusters = np.repeat(np.arange(10), 10)
    >>> trt_clusters = np.repeat(np.arange(10), 10)
    >>> result = ClusterExperiment(
    ...     ctrl, trt,
    ...     control_clusters=ctrl_clusters,
    ...     treatment_clusters=trt_clusters,
    ... ).run()
    >>> result.n_clusters_control
    10
    """

    def __init__(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
        *,
        control_clusters: ArrayLike,
        treatment_clusters: ArrayLike,
        alpha: float = 0.05,
    ) -> None:
        check_in_range(
            alpha,
            "alpha",
            0.0,
            1.0,
            hint="typical values are 0.05, 0.01, or 0.10",
        )

        self._control = check_array_like(control, "control", min_length=2)
        self._treatment = check_array_like(treatment, "treatment", min_length=2)

        # Cluster labels — allow non-numeric by converting to string first
        ctrl_cl = np.asarray(control_clusters)
        trt_cl = np.asarray(treatment_clusters)

        if ctrl_cl.ndim != 1:
            raise ValueError(
                format_error(
                    "`control_clusters` must be a 1-D array.",
                    f"got {ctrl_cl.ndim}-D array with shape {ctrl_cl.shape}.",
                )
            )
        if trt_cl.ndim != 1:
            raise ValueError(
                format_error(
                    "`treatment_clusters` must be a 1-D array.",
                    f"got {trt_cl.ndim}-D array with shape {trt_cl.shape}.",
                )
            )

        if len(ctrl_cl) != len(self._control):
            raise ValueError(
                format_error(
                    "`control` and `control_clusters` must have the same length.",
                    f"control has {len(self._control)} elements, "
                    f"control_clusters has {len(ctrl_cl)} elements.",
                )
            )
        if len(trt_cl) != len(self._treatment):
            raise ValueError(
                format_error(
                    "`treatment` and `treatment_clusters` must have the same length.",
                    f"treatment has {len(self._treatment)} elements, "
                    f"treatment_clusters has {len(trt_cl)} elements.",
                )
            )

        self._control_clusters = ctrl_cl
        self._treatment_clusters = trt_cl
        self._alpha = alpha

    @staticmethod
    def _cluster_means(values: np.ndarray, clusters: np.ndarray) -> np.ndarray:
        """Compute per-cluster means."""
        unique = np.unique(clusters)
        means = np.array([float(np.mean(values[clusters == c])) for c in unique])
        return means

    @staticmethod
    def _compute_icc(values: np.ndarray, clusters: np.ndarray) -> float:
        """Compute the intraclass correlation coefficient (ICC).

        Uses one-way random effects ANOVA: ICC = (MSB - MSW) / (MSB + (k-1)*MSW)
        where k is the average cluster size.
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

    def run(self) -> ClusterResult:
        """Run the cluster-randomized analysis.

        Returns
        -------
        ClusterResult
            Cluster-robust inference results.

        Raises
        ------
        ValueError
            If there are fewer than 2 clusters in either group.
        """
        ctrl_means = self._cluster_means(self._control, self._control_clusters)
        trt_means = self._cluster_means(self._treatment, self._treatment_clusters)

        n_c = len(ctrl_means)
        n_t = len(trt_means)

        if n_c < 2:
            raise ValueError(
                format_error(
                    "Need at least 2 control clusters for inference.",
                    f"got {n_c} control cluster(s).",
                    "add more clusters or use a unit-level test.",
                )
            )
        if n_t < 2:
            raise ValueError(
                format_error(
                    "Need at least 2 treatment clusters for inference.",
                    f"got {n_t} treatment cluster(s).",
                    "add more clusters or use a unit-level test.",
                )
            )

        mean_ctrl = float(np.mean(ctrl_means))
        mean_trt = float(np.mean(trt_means))
        lift = mean_trt - mean_ctrl

        # Welch's t-test on cluster means
        s_ctrl = float(np.std(ctrl_means, ddof=1))
        s_trt = float(np.std(trt_means, ddof=1))
        se = float(np.sqrt(s_ctrl**2 / n_c + s_trt**2 / n_t))

        if se > 0:
            t_stat = lift / se
            # Welch-Satterthwaite df
            num = (s_ctrl**2 / n_c + s_trt**2 / n_t) ** 2
            denom = (s_ctrl**2 / n_c) ** 2 / (n_c - 1) + (s_trt**2 / n_t) ** 2 / (
                n_t - 1
            )
            df = num / denom if denom > 0 else float(n_c + n_t - 2)
            pvalue = float(2 * t_dist.sf(abs(t_stat), df))
            t_crit = float(t_dist.ppf(1 - self._alpha / 2, df))
            ci_lower = lift - t_crit * se
            ci_upper = lift + t_crit * se
        else:
            pvalue = 1.0 if lift == 0 else 0.0
            ci_lower = lift
            ci_upper = lift

        # ICC on pooled data
        all_values = np.concatenate([self._control, self._treatment])
        all_clusters = np.concatenate(
            [
                np.array(["ctrl_" + str(c) for c in self._control_clusters]),
                np.array(["trt_" + str(c) for c in self._treatment_clusters]),
            ]
        )
        icc = self._compute_icc(all_values, all_clusters)

        return ClusterResult(
            lift=lift,
            pvalue=pvalue,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            significant=pvalue < self._alpha,
            n_clusters_control=n_c,
            n_clusters_treatment=n_t,
            icc=icc,
        )
