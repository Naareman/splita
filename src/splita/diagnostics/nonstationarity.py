"""Non-stationarity detection for treatment effects.

Detects if the treatment effect changes over time using sliding window
analysis and CUSUM-like change-point detection.
"""

from __future__ import annotations

import numpy as np

from splita._types import NonStationaryResult
from splita._validation import (
    check_array_like,
    check_in_range,
    check_is_integer,
    check_same_length,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


class NonStationaryDetector:
    """Detect non-stationarity in treatment effects over time.

    Uses a sliding window to compute per-window treatment effects, then
    applies CUSUM-like change-point detection and trend analysis.

    Parameters
    ----------
    window_size : int, default 7
        Number of time periods per window.
    threshold : float, default 0.05
        Significance threshold for change-point detection.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n = 100
    >>> timestamps = np.arange(n)
    >>> control = rng.normal(10, 1, n)
    >>> treatment = rng.normal(10.5, 1, n)
    >>> detector = NonStationaryDetector(window_size=10)
    >>> detector = detector.fit(control, treatment, timestamps)
    >>> result = detector.result()
    >>> result.effect_trend in ('stable', 'increasing', 'decreasing', 'volatile')
    True
    """

    def __init__(
        self,
        *,
        window_size: int = 7,
        threshold: float = 0.05,
    ) -> None:
        check_is_integer(window_size, "window_size", min_value=2)
        check_in_range(
            threshold,
            "threshold",
            0.0,
            1.0,
            hint="typical values are 0.01 or 0.05",
        )

        self._window_size = int(window_size)
        self._threshold = threshold
        self._fitted = False
        self._result: NonStationaryResult | None = None

    def fit(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
        timestamps: ArrayLike,
    ) -> NonStationaryDetector:
        """Fit the detector on control and treatment time-series data.

        Parameters
        ----------
        control : array-like
            Control group outcomes, ordered by time.
        treatment : array-like
            Treatment group outcomes, ordered by time.
        timestamps : array-like
            Timestamps or time indices for each observation.

        Returns
        -------
        NonStationaryDetector
            The fitted detector (self).

        Raises
        ------
        ValueError
            If arrays are mismatched or too short for the window size.
        """
        ctrl = check_array_like(control, "control", min_length=2)
        trt = check_array_like(treatment, "treatment", min_length=2)
        ts = check_array_like(timestamps, "timestamps", min_length=2)

        check_same_length(ctrl, trt, "control", "treatment")
        check_same_length(ctrl, ts, "control", "timestamps")

        n = len(ctrl)
        if n < self._window_size:
            raise ValueError(
                format_error(
                    f"Need at least {self._window_size} observations for "
                    f"window_size={self._window_size}.",
                    f"got {n} observations.",
                    "reduce window_size or provide more data.",
                )
            )

        # Sort by timestamp
        sort_idx = np.argsort(ts)
        ctrl = ctrl[sort_idx]
        trt = trt[sort_idx]

        # Compute per-window effects
        window_effects = []
        step = max(1, self._window_size // 2)  # 50% overlap
        for start in range(0, n - self._window_size + 1, step):
            end = start + self._window_size
            effect = float(np.mean(trt[start:end]) - np.mean(ctrl[start:end]))
            window_effects.append(
                {
                    "start": int(start),
                    "end": int(end),
                    "effect": effect,
                }
            )

        if len(window_effects) < 2:
            # Not enough windows for analysis
            self._result = NonStationaryResult(
                is_stationary=True,
                change_points=[],
                effect_trend="stable",
                window_effects=window_effects,
            )
            self._fitted = True
            return self

        effects = np.array([w["effect"] for w in window_effects])

        # CUSUM change-point detection
        change_points = self._cusum_detect(effects)

        # Trend detection via linear regression
        effect_trend = self._detect_trend(effects, change_points)

        # Stationarity: no change points and stable/non-volatile trend
        is_stationary = len(change_points) == 0 and effect_trend == "stable"

        self._result = NonStationaryResult(
            is_stationary=is_stationary,
            change_points=change_points,
            effect_trend=effect_trend,
            window_effects=window_effects,
        )
        self._fitted = True
        return self

    def _cusum_detect(self, effects: np.ndarray) -> list[int]:
        """Detect change points using CUSUM algorithm.

        Parameters
        ----------
        effects : np.ndarray
            Per-window effect estimates.

        Returns
        -------
        list[int]
            Indices in the effects array where change points were detected.
        """
        n = len(effects)
        if n < 3:
            return []

        mean_effect = float(np.mean(effects))
        std_effect = float(np.std(effects, ddof=1))

        if std_effect < 1e-12:
            return []

        # Standardised residuals
        residuals = (effects - mean_effect) / std_effect

        # CUSUM statistic
        cusum = np.cumsum(residuals)

        # Critical value based on threshold
        # Using a simple heuristic: flag if CUSUM exceeds a boundary
        # The boundary is based on the expected max of a random walk
        h = np.sqrt(n) * (-np.log(self._threshold / 2))

        change_points = []
        # Look for points where CUSUM crosses the boundary
        for i in range(1, n - 1):
            if (
                abs(cusum[i]) > h
                and abs(cusum[i]) >= abs(cusum[i - 1])
                and (i + 1 >= n or abs(cusum[i]) >= abs(cusum[i + 1]))
                and (not change_points or i - change_points[-1] >= 2)
            ):
                change_points.append(int(i))

        return change_points

    def _detect_trend(self, effects: np.ndarray, change_points: list[int]) -> str:
        """Classify the effect trend.

        Parameters
        ----------
        effects : np.ndarray
            Per-window effect estimates.
        change_points : list[int]
            Detected change points.

        Returns
        -------
        str
            One of ``'stable'``, ``'increasing'``, ``'decreasing'``,
            ``'volatile'``.
        """
        n = len(effects)
        if n < 2:
            return "stable"

        # Coefficient of variation
        mean_eff = float(np.mean(effects))
        std_eff = float(np.std(effects, ddof=1))

        cv = std_eff / abs(mean_eff) if abs(mean_eff) > 1e-12 else std_eff

        # Volatile: high variation and multiple change points
        if cv > 1.0 and len(change_points) >= 2:
            return "volatile"

        # Simple linear trend
        x = np.arange(n, dtype=float)
        x_centered = x - np.mean(x)
        slope = float(np.sum(x_centered * effects) / np.sum(x_centered**2))

        # Use correlation coefficient (r) to assess trend strength
        y_centered = effects - np.mean(effects)
        ss_x = float(np.sum(x_centered**2))
        ss_y = float(np.sum(y_centered**2))

        if ss_x > 1e-12 and ss_y > 1e-12:
            r = float(np.sum(x_centered * y_centered) / np.sqrt(ss_x * ss_y))
        else:
            r = 0.0

        # Strong monotonic trend: |r| > 0.5
        if abs(r) > 0.5:
            return "increasing" if slope > 0 else "decreasing"

        if len(change_points) >= 2:
            return "volatile"

        # Check for high variability relative to mean (volatile even without
        # formal change points)
        if cv > 1.5:
            return "volatile"

        return "stable"

    def result(self) -> NonStationaryResult:
        """Return the non-stationarity detection result.

        Returns
        -------
        NonStationaryResult
            Frozen dataclass with stationarity diagnosis.

        Raises
        ------
        RuntimeError
            If the detector has not been fitted.
        """
        if not self._fitted or self._result is None:
            raise RuntimeError(
                format_error(
                    "NonStationaryDetector must be fitted before calling result().",
                    "call .fit() first.",
                )
            )
        return self._result
