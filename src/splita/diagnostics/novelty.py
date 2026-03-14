"""NoveltyCurve — detect novelty/primacy effects via rolling-window analysis."""

from __future__ import annotations

import warnings

import numpy as np

from splita._types import NoveltyCurveResult
from splita._validation import (
    check_array_like,
    check_in_range,
    check_is_integer,
    format_error,
)
from splita.core.experiment import Experiment

ArrayLike = list | tuple | np.ndarray


class NoveltyCurve:
    """Detect novelty or primacy effects by running tests on rolling windows.

    Groups data by timestamp windows and runs :class:`~splita.Experiment`
    on each window.  If the treatment effect decreases significantly over
    time, this indicates a novelty effect.

    Parameters
    ----------
    window_size : int, default 7
        Number of unique timestamp values per window.
    metric : {'auto', 'conversion', 'continuous'}, default 'auto'
        Metric type passed to :class:`~splita.Experiment`.
    alpha : float, default 0.05
        Significance level for per-window tests.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> timestamps = np.repeat(np.arange(21), 100)
    >>> control = rng.binomial(1, 0.10, size=2100)
    >>> treatment = rng.binomial(1, 0.10, size=2100)
    >>> curve = NoveltyCurve(window_size=7).fit(control, treatment, timestamps)
    >>> result = curve.result()
    >>> result.trend_direction in ("stable", "decreasing", "increasing")
    True
    """

    def __init__(
        self,
        *,
        window_size: int = 7,
        metric: str = "auto",
        alpha: float = 0.05,
    ) -> None:
        check_is_integer(window_size, "window_size", min_value=2)
        check_in_range(
            alpha,
            "alpha",
            0.0,
            1.0,
            hint="typical values are 0.05, 0.01, or 0.10",
        )
        self._window_size = int(window_size)
        self._metric = metric
        self._alpha = alpha
        self._result: NoveltyCurveResult | None = None

    def fit(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
        timestamps: ArrayLike,
    ) -> NoveltyCurve:
        """Run rolling-window analysis.

        Parameters
        ----------
        control : array-like
            Control group observations.
        treatment : array-like
            Treatment group observations.
        timestamps : array-like
            Timestamp for each observation.  Must have the same length as
            ``control`` and ``treatment`` combined — i.e.
            ``len(timestamps) == len(control) + len(treatment)``, where the
            first ``len(control)`` entries correspond to control and the
            remaining to treatment.  Alternatively, pass two equal-length
            arrays if each observation has its own timestamp.

        Returns
        -------
        NoveltyCurve
            ``self``, for method chaining.

        Raises
        ------
        ValueError
            If inputs are invalid or there are too few windows.
        """
        ctrl = check_array_like(control, "control", min_length=2)
        trt = check_array_like(treatment, "treatment", min_length=2)
        ts = check_array_like(timestamps, "timestamps", min_length=2)

        # timestamps should match control + treatment combined length
        total_n = len(ctrl) + len(trt)
        if len(ts) != total_n:
            raise ValueError(
                format_error(
                    f"`timestamps` must have length {total_n} "
                    f"(len(control) + len(treatment)), got {len(ts)}.",
                    "each observation needs a corresponding timestamp.",
                    "concatenate control and treatment timestamps.",
                )
            )

        # Split timestamps into control and treatment
        ts_ctrl = ts[: len(ctrl)]
        ts_trt = ts[len(ctrl) :]

        # Get unique sorted timestamps
        unique_ts = np.sort(np.unique(ts))
        n_unique = len(unique_ts)

        if n_unique < self._window_size:
            raise ValueError(
                format_error(
                    f"Need at least {self._window_size} unique timestamps "
                    f"for window_size={self._window_size}, got {n_unique}.",
                    "not enough time periods for rolling-window analysis.",
                    "reduce window_size or collect more data.",
                )
            )

        n_windows = n_unique - self._window_size + 1

        if n_windows < 2:
            warnings.warn(
                f"Only {n_windows} window(s) available. "
                "Novelty detection requires at least 2 windows. "
                "Results may be unreliable.",
                RuntimeWarning,
                stacklevel=2,
            )

        windows: list[dict] = []

        for i in range(n_windows):
            window_ts = unique_ts[i : i + self._window_size]
            ts_min, ts_max = window_ts[0], window_ts[-1]

            mask_ctrl = (ts_ctrl >= ts_min) & (ts_ctrl <= ts_max)
            mask_trt = (ts_trt >= ts_min) & (ts_trt <= ts_max)

            win_ctrl = ctrl[mask_ctrl]
            win_trt = trt[mask_trt]

            if len(win_ctrl) < 2 or len(win_trt) < 2:
                continue

            try:
                exp = Experiment(
                    win_ctrl,
                    win_trt,
                    metric=self._metric,
                    alpha=self._alpha,
                )
                res = exp.run()
                windows.append(
                    {
                        "window_start": float(ts_min),
                        "lift": res.lift,
                        "pvalue": res.pvalue,
                        "ci_lower": res.ci_lower,
                        "ci_upper": res.ci_upper,
                    }
                )
            except (ValueError, RuntimeError):
                # Skip windows where the experiment can't run
                continue

        # Determine trend
        trend_direction, has_novelty = self._detect_trend(windows)

        self._result = NoveltyCurveResult(
            windows=windows,
            has_novelty_effect=has_novelty,
            trend_direction=trend_direction,
        )
        return self

    def result(self) -> NoveltyCurveResult:
        """Return the novelty curve result.

        Returns
        -------
        NoveltyCurveResult
            Frozen dataclass with window-by-window results and novelty
            detection summary.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if self._result is None:
            raise RuntimeError(
                format_error(
                    "Must call `.fit()` before `.result()`.",
                    "no analysis has been run yet.",
                    "call `.fit(control, treatment, timestamps)` first.",
                )
            )
        return self._result

    @staticmethod
    def _detect_trend(
        windows: list[dict],
    ) -> tuple[str, bool]:
        """Detect trend direction and novelty effect.

        Returns
        -------
        tuple[str, bool]
            ``(trend_direction, has_novelty_effect)``
        """
        if len(windows) < 2:
            return "stable", False

        lifts = [w["lift"] for w in windows]
        first_lift = lifts[0]
        last_lift = lifts[-1]

        # Simple linear trend via numpy polyfit
        x = np.arange(len(lifts), dtype=float)
        slope = float(np.polyfit(x, lifts, 1)[0])

        # Determine direction based on slope relative to mean absolute lift
        mean_abs_lift = float(np.mean(np.abs(lifts)))
        threshold = mean_abs_lift * 0.1 if mean_abs_lift > 0 else 1e-10

        if slope < -threshold:
            trend_direction = "decreasing"
        elif slope > threshold:
            trend_direction = "increasing"
        else:
            trend_direction = "stable"

        # Novelty detection: last window lift < first window lift * 0.5
        has_novelty = False
        if first_lift != 0:
            has_novelty = abs(last_lift) < abs(first_lift) * 0.5
        elif trend_direction == "decreasing":
            has_novelty = True

        return trend_direction, has_novelty
