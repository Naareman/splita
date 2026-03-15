"""Geo experiment analysis via Bayesian synthetic control.

Estimates incremental effects at the geographic level for marketing
incrementality measurement, inspired by Google's GeoexperimentsResearch.
"""

from __future__ import annotations

import numpy as np

from splita._types import GeoResult
from splita._validation import format_error

ArrayLike = list | tuple | np.ndarray


class GeoExperiment:
    """Bayesian synthetic control for geo-level experiments.

    Constructs a synthetic control from untreated regions to estimate
    the incremental effect of a marketing intervention in treated regions.

    Parameters
    ----------
    alpha : float, default 0.05
        Significance level for confidence intervals.
    n_bootstrap : int, default 1000
        Number of bootstrap resamples for confidence intervals.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n_t, n_c, T_pre, T_post = 3, 5, 20, 10
    >>> trt_pre = rng.normal(100, 10, (n_t, T_pre))
    >>> trt_post = rng.normal(110, 10, (n_t, T_post))
    >>> ctrl_pre = rng.normal(100, 10, (n_c, T_pre))
    >>> ctrl_post = rng.normal(100, 10, (n_c, T_post))
    >>> geo = GeoExperiment()
    >>> r = geo.fit(trt_pre, trt_post, ctrl_pre, ctrl_post)
    >>> r.n_treated_regions
    3
    """

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        n_bootstrap: int = 1000,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError(
                format_error(
                    f"`alpha` must be in (0, 1), got {alpha}.",
                    "alpha represents the significance level.",
                    "typical values are 0.05, 0.01, or 0.10.",
                )
            )
        if n_bootstrap < 100:
            raise ValueError(
                format_error(
                    f"`n_bootstrap` must be >= 100, got {n_bootstrap}.",
                    "too few bootstrap samples yield unreliable intervals.",
                    "use at least 1000 for stable results.",
                )
            )
        self._alpha = alpha
        self._n_bootstrap = n_bootstrap

    def fit(
        self,
        treated_regions_pre: ArrayLike,
        treated_regions_post: ArrayLike,
        control_regions_pre: ArrayLike,
        control_regions_post: ArrayLike,
    ) -> GeoResult:
        """Fit the geo experiment model.

        Parameters
        ----------
        treated_regions_pre : array-like, shape (n_treated, T_pre)
            Pre-period outcomes for treated regions.
        treated_regions_post : array-like, shape (n_treated, T_post)
            Post-period outcomes for treated regions.
        control_regions_pre : array-like, shape (n_control, T_pre)
            Pre-period outcomes for control regions.
        control_regions_post : array-like, shape (n_control, T_post)
            Post-period outcomes for control regions.

        Returns
        -------
        GeoResult
            Frozen dataclass with incremental effect and diagnostics.

        Raises
        ------
        ValueError
            If inputs have incompatible shapes or too few regions.
        """
        trt_pre = np.asarray(treated_regions_pre, dtype=float)
        trt_post = np.asarray(treated_regions_post, dtype=float)
        ctrl_pre = np.asarray(control_regions_pre, dtype=float)
        ctrl_post = np.asarray(control_regions_post, dtype=float)

        # Ensure 2-D
        if trt_pre.ndim == 1:
            trt_pre = trt_pre.reshape(1, -1)
        if trt_post.ndim == 1:
            trt_post = trt_post.reshape(1, -1)
        if ctrl_pre.ndim == 1:
            ctrl_pre = ctrl_pre.reshape(1, -1)
        if ctrl_post.ndim == 1:
            ctrl_post = ctrl_post.reshape(1, -1)

        n_treated = trt_pre.shape[0]
        n_control = ctrl_pre.shape[0]

        if n_treated < 1:
            raise ValueError(
                format_error(
                    "Must have at least 1 treated region.",
                    f"got {n_treated} treated regions.",
                )
            )
        if n_control < 1:
            raise ValueError(
                format_error(
                    "Must have at least 1 control region.",
                    f"got {n_control} control regions.",
                )
            )

        t_pre = trt_pre.shape[1]
        if ctrl_pre.shape[1] != t_pre:
            raise ValueError(
                format_error(
                    "Pre-period length must match between treated and control.",
                    f"treated has {t_pre}, control has {ctrl_pre.shape[1]}.",
                )
            )

        if t_pre < 2:
            raise ValueError(
                format_error(
                    "Pre-period must have at least 2 time points.",
                    f"got {t_pre}.",
                )
            )

        # Aggregate to region-level time series (mean across regions)
        trt_pre_agg = np.mean(trt_pre, axis=0)  # (T_pre,)
        trt_post_agg = np.mean(trt_post, axis=0)  # (T_post,)
        ctrl_pre_agg = np.mean(ctrl_pre, axis=0)  # (T_pre,)
        ctrl_post_agg = np.mean(ctrl_post, axis=0)  # (T_post,)

        # Fit synthetic control weights: trt_pre ~ w * ctrl_pre
        # Simple OLS on aggregated series
        weights = self._fit_synthetic_weights(trt_pre_agg, ctrl_pre_agg)

        # Predicted counterfactual post-period
        synth_post = weights * ctrl_post_agg

        # Pre-period RMSE
        synth_pre = weights * ctrl_pre_agg
        pre_rmse = float(np.sqrt(np.mean((trt_pre_agg - synth_pre) ** 2)))

        # Incremental effect
        incremental_effect = float(np.sum(trt_post_agg - synth_post))

        # Bootstrap CI using region-level resampling
        rng = np.random.default_rng(42)
        boot_effects = []
        for _ in range(self._n_bootstrap):
            # Resample treated regions
            t_idx = rng.integers(0, n_treated, size=n_treated)
            c_idx = rng.integers(0, n_control, size=n_control)

            b_trt_pre = np.mean(trt_pre[t_idx], axis=0)
            b_trt_post = np.mean(trt_post[t_idx], axis=0)
            b_ctrl_pre = np.mean(ctrl_pre[c_idx], axis=0)
            b_ctrl_post = np.mean(ctrl_post[c_idx], axis=0)

            w = self._fit_synthetic_weights(b_trt_pre, b_ctrl_pre)
            b_synth_post = w * b_ctrl_post
            boot_effects.append(float(np.sum(b_trt_post - b_synth_post)))

        boot_effects = np.array(boot_effects)
        ci_lower = float(np.percentile(boot_effects, 100 * self._alpha / 2))
        ci_upper = float(np.percentile(boot_effects, 100 * (1 - self._alpha / 2)))

        return GeoResult(
            incremental_effect=incremental_effect,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            pre_rmse=pre_rmse,
            n_treated_regions=n_treated,
            n_control_regions=n_control,
        )

    @staticmethod
    def _fit_synthetic_weights(target: np.ndarray, donor: np.ndarray) -> float:
        """Fit a scalar weight: target ~ w * donor via OLS."""
        denom = float(np.dot(donor, donor))
        if denom < 1e-12:
            return 1.0
        return float(np.dot(target, donor) / denom)
