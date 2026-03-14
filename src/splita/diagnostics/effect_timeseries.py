"""EffectTimeSeries — track treatment effect stability over time."""

from __future__ import annotations

import numpy as np

from splita._types import EffectTimeSeriesResult
from splita._validation import (
    check_array_like,
    check_in_range,
    format_error,
)
from splita.core.experiment import Experiment

ArrayLike = list | tuple | np.ndarray


class EffectTimeSeries:
    """Track cumulative treatment effect over time.

    Groups data by unique timestamp values and computes the cumulative
    effect at each time point (all data up to that point) using
    :class:`~splita.Experiment`.

    Parameters
    ----------
    alpha : float, default 0.05
        Significance level for the per-timepoint tests.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> timestamps = np.repeat(np.arange(10), 200)
    >>> control = rng.binomial(1, 0.10, size=2000)
    >>> treatment = rng.binomial(1, 0.15, size=2000)
    >>> ts = EffectTimeSeries().fit(control, treatment, timestamps)
    >>> result = ts.result()
    >>> len(result.time_points) > 0
    True
    """

    def __init__(self, *, alpha: float = 0.05) -> None:
        check_in_range(
            alpha,
            "alpha",
            0.0,
            1.0,
            hint="typical values are 0.05, 0.01, or 0.10",
        )
        self._alpha = alpha
        self._result: EffectTimeSeriesResult | None = None

    def fit(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
        timestamps: ArrayLike,
    ) -> EffectTimeSeries:
        """Compute cumulative effect at each time point.

        Parameters
        ----------
        control : array-like
            Control group observations.
        treatment : array-like
            Treatment group observations.
        timestamps : array-like
            Timestamp for each observation.  Must have length
            ``len(control) + len(treatment)``.  The first ``len(control)``
            entries correspond to control, the rest to treatment.

        Returns
        -------
        EffectTimeSeries
            ``self``, for method chaining.

        Raises
        ------
        ValueError
            If inputs are invalid.
        """
        ctrl = check_array_like(control, "control", min_length=2)
        trt = check_array_like(treatment, "treatment", min_length=2)
        ts = check_array_like(timestamps, "timestamps", min_length=2)

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

        ts_ctrl = ts[: len(ctrl)]
        ts_trt = ts[len(ctrl) :]

        unique_ts = np.sort(np.unique(ts))

        time_points: list[dict] = []

        for t in unique_ts:
            mask_ctrl = ts_ctrl <= t
            mask_trt = ts_trt <= t

            cum_ctrl = ctrl[mask_ctrl]
            cum_trt = trt[mask_trt]

            if len(cum_ctrl) < 2 or len(cum_trt) < 2:
                continue

            try:
                exp = Experiment(
                    cum_ctrl,
                    cum_trt,
                    alpha=self._alpha,
                )
                res = exp.run()
                time_points.append(
                    {
                        "timestamp": float(t),
                        "cumulative_lift": res.lift,
                        "pvalue": res.pvalue,
                        "ci_lower": res.ci_lower,
                        "ci_upper": res.ci_upper,
                        "n_control": len(cum_ctrl),
                        "n_treatment": len(cum_trt),
                    }
                )
            except (ValueError, RuntimeError):  # pragma: no cover
                continue

        # Determine stability
        is_stable = self._check_stability(time_points)

        final_lift = time_points[-1]["cumulative_lift"] if time_points else 0.0
        final_pvalue = time_points[-1]["pvalue"] if time_points else 1.0

        self._result = EffectTimeSeriesResult(
            time_points=time_points,
            is_stable=is_stable,
            final_lift=final_lift,
            final_pvalue=final_pvalue,
        )
        return self

    def result(self) -> EffectTimeSeriesResult:
        """Return the effect time series result.

        Returns
        -------
        EffectTimeSeriesResult
            Frozen dataclass with per-timepoint cumulative results.

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
    def _check_stability(time_points: list[dict]) -> bool:
        """Check if the effect is stable over time.

        Stable means the lift in the last third of time points does not
        deviate significantly from the lift in the middle third.

        Returns
        -------
        bool
            ``True`` if the effect appears stable.
        """
        if len(time_points) < 3:
            return True

        lifts = [tp["cumulative_lift"] for tp in time_points]
        n = len(lifts)

        # Use the second half vs the full series final value
        # Stability = coefficient of variation of the last half of lifts is small
        half = n // 2
        recent_lifts = lifts[half:]

        if len(recent_lifts) < 2:  # pragma: no cover
            return True

        recent_arr = np.array(recent_lifts)
        mean_lift = float(np.mean(recent_arr))
        std_lift = float(np.std(recent_arr))

        if abs(mean_lift) < 1e-10:
            # Near-zero lift — check absolute variation
            return std_lift < 0.01

        cv = abs(std_lift / mean_lift)
        return cv < 0.5
