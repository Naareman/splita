"""AdaptiveWinsorizer — Grid-search for optimal winsorization thresholds.

Instead of using fixed percentile thresholds (e.g. 1% / 99%), the adaptive
winsorizer searches over a grid of (lower, upper) percentile pairs and
selects the one that minimises the variance of the treatment effect
estimator.  This is especially useful for heavy-tailed distributions where
the optimal capping level is not known a priori.
"""

from __future__ import annotations

import numpy as np

from splita._validation import (
    check_array_like,
    check_is_integer,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


class AdaptiveWinsorizer:
    """Find optimal winsorization thresholds by grid search.

    Searches over a grid of ``(lower_pct, upper_pct)`` pairs on the
    pooled data (control + treatment combined).  For each pair, both
    groups are winsorized and the variance of the treatment effect
    estimator ``Var(mean_trt - mean_ctrl)`` is computed.  The pair
    that minimises this variance is selected.

    Parameters
    ----------
    n_grid : int, default 50
        Number of points in each dimension of the grid.  Total
        candidates evaluated is ``n_grid * n_grid``.
    lower_range : tuple of float, default (0.001, 0.10)
        Range of lower percentile thresholds to search (inclusive).
        Must satisfy ``0 <= lower_range[0] < lower_range[1] < 0.5``.
    upper_range : tuple of float, default (0.90, 0.999)
        Range of upper percentile thresholds to search (inclusive).
        Must satisfy ``0.5 < upper_range[0] < upper_range[1] <= 1.0``.

    Attributes
    ----------
    optimal_lower_ : float
        Best lower percentile from the grid search.
    optimal_upper_ : float
        Best upper percentile from the grid search.
    variance_reduction_ : float
        Fractional variance reduction achieved vs. no winsorization:
        ``1 - Var_best / Var_original``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> ctrl = np.concatenate([rng.normal(10, 2, 490), rng.normal(100, 50, 10)])
    >>> trt = np.concatenate([rng.normal(10.5, 2, 490), rng.normal(100, 50, 10)])
    >>> winz = AdaptiveWinsorizer()
    >>> ctrl_w, trt_w = winz.fit_transform(ctrl, trt)
    >>> winz.variance_reduction_ > 0.0
    True
    """

    def __init__(
        self,
        *,
        n_grid: int = 50,
        lower_range: tuple[float, float] = (0.001, 0.10),
        upper_range: tuple[float, float] = (0.90, 0.999),
    ) -> None:
        # ── validate n_grid ─────────────────────────────────────────
        check_is_integer(n_grid, "n_grid", min_value=2)

        # ── validate lower_range ────────────────────────────────────
        if (
            not isinstance(lower_range, tuple)
            or len(lower_range) != 2
            or lower_range[0] < 0.0
            or lower_range[1] >= 0.5
            or lower_range[0] >= lower_range[1]
        ):
            raise ValueError(
                format_error(
                    "`lower_range` must be a 2-tuple (a, b) with 0 <= a < b < 0.5.",
                    detail=f"got {lower_range!r}.",
                    hint="e.g. (0.001, 0.10).",
                )
            )

        # ── validate upper_range ────────────────────────────────────
        if (
            not isinstance(upper_range, tuple)
            or len(upper_range) != 2
            or upper_range[0] <= 0.5
            or upper_range[1] > 1.0
            or upper_range[0] >= upper_range[1]
        ):
            raise ValueError(
                format_error(
                    "`upper_range` must be a 2-tuple (a, b) with 0.5 < a < b <= 1.0.",
                    detail=f"got {upper_range!r}.",
                    hint="e.g. (0.90, 0.999).",
                )
            )

        self.n_grid = int(n_grid)
        self.lower_range = lower_range
        self.upper_range = upper_range

        # Fitted attributes (set by fit)
        self.optimal_lower_: float
        self.optimal_upper_: float
        self.variance_reduction_: float
        self._lower_threshold: float
        self._upper_threshold: float
        self._is_fitted = False

    # ── fit ──────────────────────────────────────────────────────────

    def fit(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
    ) -> AdaptiveWinsorizer:
        """Search for optimal winsorization thresholds.

        Thresholds are computed on the **combined** data to avoid
        introducing bias between groups.

        Parameters
        ----------
        control : array-like
            Observations for the control group.
        treatment : array-like
            Observations for the treatment group.

        Returns
        -------
        AdaptiveWinsorizer
            The fitted instance (for method chaining).

        Raises
        ------
        ValueError
            If arrays are too short.
        """
        control_arr = check_array_like(control, "control", min_length=2)
        treatment_arr = check_array_like(treatment, "treatment", min_length=2)

        combined = np.concatenate([control_arr, treatment_arr])
        n_c = len(control_arr)
        n_t = len(treatment_arr)

        # Build grids
        lower_grid = np.linspace(self.lower_range[0], self.lower_range[1], self.n_grid)
        upper_grid = np.linspace(self.upper_range[0], self.upper_range[1], self.n_grid)

        # Precompute percentiles for all grid points
        all_pcts = np.concatenate([lower_grid * 100, upper_grid * 100])
        all_thresholds = np.percentile(combined, all_pcts)
        lower_thresholds = all_thresholds[: self.n_grid]
        upper_thresholds = all_thresholds[self.n_grid :]

        # Variance of the original (unwinsorized) treatment effect estimator
        var_original = float(
            np.var(control_arr, ddof=1) / n_c + np.var(treatment_arr, ddof=1) / n_t
        )

        best_var = np.inf
        best_lower_pct = lower_grid[0]
        best_upper_pct = upper_grid[-1]
        best_lower_thresh = lower_thresholds[0]
        best_upper_thresh = upper_thresholds[-1]

        for lp, lt in zip(lower_grid, lower_thresholds, strict=True):
            for up, ut in zip(upper_grid, upper_thresholds, strict=True):
                if lt >= ut:
                    continue  # invalid: lower threshold must be below upper

                ctrl_w = np.clip(control_arr, lt, ut)
                trt_w = np.clip(treatment_arr, lt, ut)

                var_effect = float(np.var(ctrl_w, ddof=1) / n_c + np.var(trt_w, ddof=1) / n_t)

                if var_effect < best_var:
                    best_var = var_effect
                    best_lower_pct = lp
                    best_upper_pct = up
                    best_lower_thresh = lt
                    best_upper_thresh = ut

        self.optimal_lower_ = float(best_lower_pct)
        self.optimal_upper_ = float(best_upper_pct)
        self._lower_threshold = float(best_lower_thresh)
        self._upper_threshold = float(best_upper_thresh)

        if var_original > 0:
            self.variance_reduction_ = max(0.0, 1.0 - best_var / var_original)
        else:
            self.variance_reduction_ = 0.0

        self._is_fitted = True
        return self

    # ── transform ───────────────────────────────────────────────────

    def transform(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the optimal winsorization thresholds.

        Parameters
        ----------
        control : array-like
            Observations for the control group.
        treatment : array-like
            Observations for the treatment group.

        Returns
        -------
        tuple of np.ndarray
            ``(control_winsorized, treatment_winsorized)``.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        """
        if not self._is_fitted:
            raise RuntimeError(
                format_error(
                    "AdaptiveWinsorizer must be fitted before calling transform().",
                    detail="optimal thresholds have not been computed yet.",
                    hint="call fit() or fit_transform() first.",
                )
            )

        control_arr = check_array_like(control, "control", min_length=2)
        treatment_arr = check_array_like(treatment, "treatment", min_length=2)

        ctrl_w = np.clip(control_arr, self._lower_threshold, self._upper_threshold)
        trt_w = np.clip(treatment_arr, self._lower_threshold, self._upper_threshold)

        return ctrl_w, trt_w

    # ── fit_transform ───────────────────────────────────────────────

    def fit_transform(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit thresholds and apply them in one step.

        Convenience method equivalent to calling :meth:`fit` followed
        by :meth:`transform` with the same arguments.

        Parameters
        ----------
        control : array-like
            Observations for the control group.
        treatment : array-like
            Observations for the treatment group.

        Returns
        -------
        tuple of np.ndarray
            ``(control_winsorized, treatment_winsorized)``.
        """
        self.fit(control, treatment)
        return self.transform(control, treatment)
