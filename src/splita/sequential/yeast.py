"""YEAST — Yet another Easy Always-valid Sequential Test.

A novel sequential test using a constant significance boundary derived
from Levy's inequalities. No tuning parameters required.

References
----------
.. [1] Kurennoy, A. (Meta) (2024).  "YEAST: Yet another Easy Always-valid
       Sequential Test."  arXiv preprint.
"""

from __future__ import annotations

import math

import numpy as np

from splita._types import YEASTResult, YEASTState
from splita._validation import (
    check_array_like,
    check_in_range,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


class YEASTSequentialTest:
    """Sequential test using Levy's inequality for constant boundaries.

    Unlike mSPRT, YEAST requires no tuning parameters (no tau, no
    mixing variance). The test statistic is a z-score and the boundary
    is derived from Levy's maximal inequality, providing always-valid
    Type I error control.

    Parameters
    ----------
    alpha : float, default 0.05
        Target Type I error rate.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> test = YEASTSequentialTest(alpha=0.05)
    >>> ctrl = rng.normal(0.0, 1.0, 500)
    >>> trt = rng.normal(0.5, 1.0, 500)
    >>> state = test.update(ctrl, trt)
    >>> state.should_stop
    True
    """

    def __init__(self, *, alpha: float = 0.05) -> None:
        check_in_range(alpha, "alpha", 0.0, 1.0)
        self._alpha = alpha

        # Levy boundary: for a standardised process, P(max Z_n > c) <= alpha
        # Using Levy's inequality: c = sqrt(2 * log(2 / alpha))
        self._boundary = math.sqrt(2.0 * math.log(2.0 / alpha))

        # Running statistics
        self._n_control: int = 0
        self._n_treatment: int = 0
        self._sum_control: float = 0.0
        self._sum_treatment: float = 0.0
        self._ss_control: float = 0.0
        self._ss_treatment: float = 0.0
        self._updates: int = 0
        self._state: YEASTState | None = None

    def update(
        self,
        control_obs: ArrayLike,
        treatment_obs: ArrayLike,
    ) -> YEASTState:
        """Ingest new observations and return the current test state.

        Parameters
        ----------
        control_obs : array-like
            New control-group observations.
        treatment_obs : array-like
            New treatment-group observations.

        Returns
        -------
        YEASTState
            Current state with z-score, boundary, and stopping decision.

        Raises
        ------
        ValueError
            If both arrays are empty on the first call.
        """
        ctrl = check_array_like(control_obs, "control_obs")
        trt = check_array_like(treatment_obs, "treatment_obs")

        if len(ctrl) == 0 and len(trt) == 0:
            if self._updates == 0:
                raise ValueError(
                    format_error(
                        "`control_obs` and `treatment_obs` can't both be empty "
                        "on the first update.",
                        "the test needs at least some data to begin.",
                        "pass non-empty arrays of observations.",
                    )
                )
            assert self._state is not None
            return self._state

        # Accumulate running statistics
        self._n_control += len(ctrl)
        self._n_treatment += len(trt)
        self._sum_control += float(np.sum(ctrl))
        self._sum_treatment += float(np.sum(trt))
        self._ss_control += float(np.sum(ctrl**2))
        self._ss_treatment += float(np.sum(trt**2))
        self._updates += 1

        n_c = self._n_control
        n_t = self._n_treatment

        # Need at least 1 obs per group
        if n_c == 0 or n_t == 0:
            self._state = YEASTState(
                n_control=n_c,
                n_treatment=n_t,
                z_statistic=0.0,
                boundary=self._boundary,
                pvalue=1.0,
                should_stop=False,
                current_effect_estimate=0.0,
            )
            return self._state

        mean_c = self._sum_control / n_c
        mean_t = self._sum_treatment / n_t
        delta_hat = mean_t - mean_c

        # Pooled variance estimate
        if n_c > 1:
            var_c = max(0.0, (self._ss_control - self._sum_control**2 / n_c) / (n_c - 1))
        else:
            var_c = 0.0

        if n_t > 1:
            var_t = max(0.0, (self._ss_treatment - self._sum_treatment**2 / n_t) / (n_t - 1))
        else:
            var_t = 0.0

        se = math.sqrt(var_c / n_c + var_t / n_t) if (var_c + var_t) > 0 else 0.0

        # Z-score
        z = delta_hat / se if se > 0 else 0.0

        # Sequential p-value using Levy's inequality
        # P(max|Z_n| > |z|) <= 2 * exp(-z^2 / 2)
        pvalue = min(1.0, 2.0 * math.exp(-z**2 / 2.0)) if z != 0 else 1.0

        should_stop = abs(z) >= self._boundary

        self._state = YEASTState(
            n_control=n_c,
            n_treatment=n_t,
            z_statistic=float(z),
            boundary=self._boundary,
            pvalue=float(pvalue),
            should_stop=should_stop,
            current_effect_estimate=float(delta_hat),
        )
        return self._state

    def result(self) -> YEASTResult:
        """Return the final result of the test.

        Returns
        -------
        YEASTResult
            Final result with stopping metadata.

        Raises
        ------
        RuntimeError
            If called before any :meth:`update` call.
        """
        if self._state is None:
            raise RuntimeError(
                format_error(
                    "can't produce a result before any data has been observed.",
                    "call update() at least once before calling result().",
                    "feed observations via update(control_obs, treatment_obs).",
                )
            )

        if self._state.should_stop:
            stopping_reason = "boundary_crossed"
        else:
            stopping_reason = "not_stopped"

        n_total = self._state.n_control + self._state.n_treatment

        return YEASTResult(
            n_control=self._state.n_control,
            n_treatment=self._state.n_treatment,
            z_statistic=self._state.z_statistic,
            boundary=self._state.boundary,
            pvalue=self._state.pvalue,
            should_stop=self._state.should_stop,
            current_effect_estimate=self._state.current_effect_estimate,
            stopping_reason=stopping_reason,
            total_observations=n_total,
        )
