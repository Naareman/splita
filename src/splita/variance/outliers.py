"""OutlierHandler — Outlier capping for A/B test metrics.

Reduces the influence of extreme values on test statistics by winsorizing,
trimming, or applying IQR-based capping.  Thresholds are always computed
from the *pooled* data (control + treatment combined) to avoid introducing
bias between groups.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from splita._validation import (
    check_array_like,
    check_positive,
    format_error,
)

ArrayLike = list | tuple | np.ndarray

_VALID_METHODS = ("winsorize", "trim", "iqr", "clustering")


class OutlierHandler:
    """Cap or remove outliers in A/B test metric data.

    Extreme values inflate variance and reduce the power of statistical
    tests.  ``OutlierHandler`` fits thresholds on the **pooled** data
    (both groups combined) and then applies them symmetrically to each
    group, preserving the unbiased comparison.

    Parameters
    ----------
    method : {'winsorize', 'trim', 'iqr', 'clustering'}, default 'winsorize'
        Outlier handling strategy:

        - ``'winsorize'``: cap values at the lower/upper percentile
          thresholds (array length is preserved).
        - ``'trim'``: remove values outside the thresholds (arrays may
          shrink).
        - ``'iqr'``: cap values using IQR-based fences
          (Q1 - k*IQR, Q3 + k*IQR).
        - ``'clustering'``: use DBSCAN to identify outlier clusters.
          Requires scikit-learn.  Outlier values (label=-1) are
          winsorized to the nearest non-outlier percentile.
    lower : float or None, default 0.01
        Lower percentile cap (e.g. 0.01 = 1st percentile).  ``None``
        means no lower capping.  Must be in [0, 0.5).
    upper : float or None, default 0.99
        Upper percentile cap (e.g. 0.99 = 99th percentile).  ``None``
        means no upper capping.  Must be in (0.5, 1].
    side : {'both', 'upper', 'lower'} or None, default None
        Convenience parameter that overrides ``lower``/``upper``:

        - ``'upper'``: only cap the upper tail (``lower`` set to None).
        - ``'lower'``: only cap the lower tail (``upper`` set to None).
        - ``'both'`` or ``None``: use ``lower`` and ``upper`` as-is.
    iqr_multiplier : float, default 1.5
        Multiplier for the IQR method (Tukey's rule).  Use 3.0 for less
        aggressive capping.
    eps : float, default 0.5
        DBSCAN neighbourhood radius (only used when ``method='clustering'``).
    min_cluster_samples : int, default 5
        Minimum number of points to form a dense region in DBSCAN
        (only used when ``method='clustering'``).

    Attributes
    ----------
    lower_threshold_ : float or None
        Fitted lower threshold (set after :meth:`fit`).
    upper_threshold_ : float or None
        Fitted upper threshold (set after :meth:`fit`).
    n_total_ : int
        Total number of observations seen during :meth:`fit`.
    n_capped_ : int
        Number of values capped or removed during :meth:`transform`.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> ctrl = rng.normal(10, 2, size=500)
    >>> trt = rng.normal(10.5, 2, size=500)
    >>> ctrl[0] = 100  # extreme outlier
    >>> handler = OutlierHandler(method='winsorize')
    >>> ctrl_c, trt_c = handler.fit_transform(ctrl, trt)
    >>> ctrl_c[0] < 100
    True
    """

    def __init__(
        self,
        *,
        method: Literal["winsorize", "trim", "iqr", "clustering"] = "winsorize",
        lower: float | None = 0.01,
        upper: float | None = 0.99,
        side: Literal["both", "upper", "lower"] | None = None,
        iqr_multiplier: float = 1.5,
        eps: float = 0.5,
        min_cluster_samples: int = 5,
    ) -> None:
        # ── validate method ──────────────────────────────────────────
        if method not in _VALID_METHODS:
            raise ValueError(
                format_error(
                    f"`method` must be one of {_VALID_METHODS}, got {method!r}.",
                    hint="use 'winsorize' (default), 'trim', or 'iqr'.",
                )
            )

        # ── validate percentile bounds ───────────────────────────────
        if lower is not None and (lower < 0.0 or lower >= 0.5):
            raise ValueError(
                format_error(
                    f"`lower` must be in [0, 0.5), got {lower}.",
                    detail="lower percentile must be below the median.",
                    hint="typical values are 0.01 or 0.05.",
                )
            )
        if upper is not None and (upper <= 0.5 or upper > 1.0):
            raise ValueError(
                format_error(
                    f"`upper` must be in (0.5, 1], got {upper}.",
                    detail="upper percentile must be above the median.",
                    hint="typical values are 0.95 or 0.99.",
                )
            )
        if (  # pragma: no cover
            lower is not None and upper is not None and lower >= upper
        ):
            raise ValueError(
                format_error(
                    f"`lower` must be less than `upper`, "
                    f"got lower={lower}, upper={upper}.",
                    detail="the lower percentile must be strictly "
                    "below the upper percentile.",
                    hint="e.g. lower=0.01, upper=0.99.",
                )
            )

        # ── validate iqr_multiplier ──────────────────────────────────
        check_positive(iqr_multiplier, "iqr_multiplier")

        # ── validate clustering params ───────────────────────────────
        if method == "clustering":
            check_positive(eps, "eps", hint="DBSCAN eps must be > 0.")
            if not isinstance(min_cluster_samples, int) or min_cluster_samples < 1:
                raise ValueError(
                    format_error(
                        f"`min_cluster_samples` must be a positive integer, "
                        f"got {min_cluster_samples}.",
                        hint="typical values are 3 to 10.",
                    )
                )

        # ── apply side convenience ───────────────────────────────────
        if side == "upper":
            lower = None
        elif side == "lower":
            upper = None
        elif side is not None and side != "both":
            raise ValueError(
                format_error(
                    f"`side` must be 'both', 'upper', 'lower', or None, got {side!r}.",
                )
            )

        self.method = method
        self.lower = lower
        self.upper = upper
        self.side = side
        self.iqr_multiplier = iqr_multiplier
        self.eps = eps
        self.min_cluster_samples = min_cluster_samples

        # Fitted attributes (set by fit)
        self.lower_threshold_: float | None = None
        self.upper_threshold_: float | None = None
        self.n_total_: int = 0
        self.n_capped_: int = 0
        self._is_fitted = False

    # ── fit ───────────────────────────────────────────────────────────

    def fit(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
    ) -> OutlierHandler:
        """Learn outlier thresholds from pooled data.

        Thresholds are computed on the **combined** control + treatment
        data to avoid introducing bias between groups.

        Parameters
        ----------
        control : array-like
            Observations for the control group.
        treatment : array-like
            Observations for the treatment group.

        Returns
        -------
        OutlierHandler
            The fitted instance (for method chaining).

        Raises
        ------
        ValueError
            If arrays are too short.
        """
        control_arr = check_array_like(control, "control", min_length=2)
        treatment_arr = check_array_like(treatment, "treatment", min_length=2)

        combined = np.concatenate([control_arr, treatment_arr])
        self.n_total_ = len(combined)

        if self.method in ("winsorize", "trim"):
            self.lower_threshold_ = (
                float(np.percentile(combined, self.lower * 100))
                if self.lower is not None
                else None
            )
            self.upper_threshold_ = (
                float(np.percentile(combined, self.upper * 100))
                if self.upper is not None
                else None
            )
        elif self.method == "iqr":
            q1 = float(np.percentile(combined, 25))
            q3 = float(np.percentile(combined, 75))
            iqr = q3 - q1
            k = self.iqr_multiplier
            # IQR method respects the lower/upper=None settings
            self.lower_threshold_ = q1 - k * iqr if self.lower is not None else None
            self.upper_threshold_ = q3 + k * iqr if self.upper is not None else None
        elif self.method == "clustering":
            self._fit_clustering(combined)

        self._is_fitted = True
        return self

    # ── transform ─────────────────────────────────────────────────────

    def transform(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply fitted thresholds to control and treatment data.

        Parameters
        ----------
        control : array-like
            Observations for the control group.
        treatment : array-like
            Observations for the treatment group.

        Returns
        -------
        tuple of np.ndarray
            ``(control_capped, treatment_capped)`` — the transformed
            observations for each group.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        """
        if not self._is_fitted:
            raise RuntimeError(
                format_error(
                    "OutlierHandler must be fitted before calling transform().",
                    detail="thresholds have not been computed yet.",
                    hint="call fit() or fit_transform() first.",
                )
            )

        control_arr = check_array_like(control, "control", min_length=2)
        treatment_arr = check_array_like(treatment, "treatment", min_length=2)

        if self.method in ("winsorize", "iqr", "clustering"):
            ctrl_out, n_ctrl = self._cap(control_arr)
            trt_out, n_trt = self._cap(treatment_arr)
        else:
            # trim
            ctrl_out, n_ctrl = self._trim(control_arr)
            trt_out, n_trt = self._trim(treatment_arr)

        self.n_capped_ = n_ctrl + n_trt
        return ctrl_out, trt_out

    # ── fit_transform ─────────────────────────────────────────────────

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
            ``(control_capped, treatment_capped)``.
        """
        self.fit(control, treatment)
        return self.transform(control, treatment)

    # ── private helpers ───────────────────────────────────────────────

    def _cap(self, arr: np.ndarray) -> tuple[np.ndarray, int]:
        """Cap (winsorize) values at thresholds. Returns (capped_array, n_affected)."""
        result = arr.copy()
        n_affected = 0

        if self.lower_threshold_ is not None:
            mask = result < self.lower_threshold_
            n_affected += int(mask.sum())
            result[mask] = self.lower_threshold_

        if self.upper_threshold_ is not None:
            mask = result > self.upper_threshold_
            n_affected += int(mask.sum())
            result[mask] = self.upper_threshold_

        return result, n_affected

    def _trim(self, arr: np.ndarray) -> tuple[np.ndarray, int]:
        """Remove values outside thresholds. Returns (trimmed_array, n_removed)."""
        mask = np.ones(len(arr), dtype=bool)

        if self.lower_threshold_ is not None:
            mask &= arr >= self.lower_threshold_
        if self.upper_threshold_ is not None:
            mask &= arr <= self.upper_threshold_

        n_removed = int((~mask).sum())
        return arr[mask], n_removed

    def _fit_clustering(self, combined: np.ndarray) -> None:
        """Fit DBSCAN on pooled data and set thresholds from non-outlier range.

        Points labelled -1 by DBSCAN are outliers.  Thresholds are set to
        the min/max of the non-outlier points so that :meth:`_cap` winsorizes
        outlier values to the nearest non-outlier boundary.
        """
        try:
            from sklearn.cluster import DBSCAN
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                format_error(
                    "method='clustering' requires scikit-learn.",
                    detail="DBSCAN is used for outlier detection.",
                    hint="install scikit-learn: pip install scikit-learn.",
                )
            ) from exc

        X = combined.reshape(-1, 1)
        db = DBSCAN(eps=self.eps, min_samples=self.min_cluster_samples)
        labels = db.fit_predict(X)

        non_outlier_mask = labels != -1

        if not np.any(non_outlier_mask):
            # All points are outliers — fall back to no capping
            self.lower_threshold_ = None
            self.upper_threshold_ = None
            return

        non_outlier_values = combined[non_outlier_mask]
        self.lower_threshold_ = float(np.min(non_outlier_values))
        self.upper_threshold_ = float(np.max(non_outlier_values))
