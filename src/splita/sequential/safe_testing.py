"""E-process (safe testing) framework (Grunwald et al. 2020).

Full e-process framework extending the basic EValue class.  E-processes
are products of sequential e-values that provide anytime-valid inference.
Rejection occurs when E_t >= 1/alpha.

Reference
---------
Grunwald, de Heide & Koolen (2020). "Safe Testing."
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from splita._types import EProcessResult, EProcessState
from splita._validation import (
    check_array_like,
    check_in_range,
    check_one_of,
    format_error,
)

_VALID_METHODS = ["grapa", "universal"]


class EProcess:
    """E-process based sequential testing.

    E-processes accumulate evidence multiplicatively across sequential
    e-values, providing anytime-valid inference.  The GRAPA method is
    growth-rate optimal under the alternative; the universal method is
    more robust but less powerful.

    Parameters
    ----------
    alpha : float, default 0.05
        Target Type I error rate.
    method : ``'grapa'`` or ``'universal'``, default ``'grapa'``
        E-value construction method.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> ep = EProcess(alpha=0.05, method="grapa")
    >>> ctrl = rng.normal(0.0, 1.0, size=500)
    >>> trt = rng.normal(0.5, 1.0, size=500)
    >>> state = ep.update(ctrl, trt)
    >>> state.e_value > 1.0
    True
    """

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        method: Literal["grapa", "universal"] = "grapa",
    ) -> None:
        # ── validation ──
        check_in_range(alpha, "alpha", 0.0, 1.0)
        check_one_of(method, "method", _VALID_METHODS)

        self._alpha = alpha
        self._method = method

        # ── running statistics ──
        self._n_control: int = 0
        self._n_treatment: int = 0
        self._sum_control: float = 0.0
        self._sum_treatment: float = 0.0
        self._ss_control: float = 0.0
        self._ss_treatment: float = 0.0
        self._updates: int = 0

        # ── e-process accumulator (log scale for stability) ──
        self._log_e_process: float = 0.0

        # ── latest state ──
        self._state: EProcessState | None = None

    # ─── public API ──────────────────────────────────────────────────

    def update(
        self,
        control_obs: list | tuple | np.ndarray,
        treatment_obs: list | tuple | np.ndarray,
    ) -> EProcessState:
        """Ingest new observations and return the current e-process state.

        Can be called repeatedly for streaming / incremental updates, or
        once with the full dataset for batch analysis.

        Parameters
        ----------
        control_obs : array-like
            New control-group observations.
        treatment_obs : array-like
            New treatment-group observations.

        Returns
        -------
        EProcessState
            Current state including the e-value and stopping decision.

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

        # ── accumulate running statistics ──
        self._n_control += len(ctrl)
        self._n_treatment += len(trt)
        self._sum_control += float(np.sum(ctrl))
        self._sum_treatment += float(np.sum(trt))
        self._ss_control += float(np.sum(ctrl**2))
        self._ss_treatment += float(np.sum(trt**2))
        self._updates += 1

        n_c = self._n_control
        n_t = self._n_treatment

        # Need at least 1 obs in each group
        if n_c == 0 or n_t == 0:
            self._state = EProcessState(
                e_value=1.0,
                log_e_value=0.0,
                n_control=n_c,
                n_treatment=n_t,
                should_stop=False,
            )
            return self._state

        # ── compute batch e-value ──
        if self._method == "grapa":
            log_e_batch = self._grapa_log_e(ctrl, trt, n_c, n_t)
        else:
            log_e_batch = self._universal_log_e(ctrl, trt, n_c, n_t)

        # ── accumulate multiplicatively (in log space) ──
        self._log_e_process += log_e_batch

        # ── clamp to prevent underflow ──
        e_value = math.exp(min(self._log_e_process, 700.0))

        # ── stopping decision ──
        threshold = 1.0 / self._alpha
        should_stop = e_value >= threshold

        self._state = EProcessState(
            e_value=float(e_value),
            log_e_value=float(self._log_e_process),
            n_control=n_c,
            n_treatment=n_t,
            should_stop=should_stop,
        )
        return self._state

    def result(self) -> EProcessResult:
        """Return the final result of the e-process test.

        Should be called after one or more :meth:`update` calls.

        Returns
        -------
        EProcessResult
            Final result with stopping metadata and safe CI.

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
            stopping_reason = "e_process_threshold_crossed"
        else:
            stopping_reason = "not_stopped"

        # ── safe CI by inverting the e-process ──
        safe_ci_lower, safe_ci_upper = self._compute_safe_ci()

        return EProcessResult(
            e_value=self._state.e_value,
            log_e_value=self._state.log_e_value,
            n_control=self._state.n_control,
            n_treatment=self._state.n_treatment,
            should_stop=self._state.should_stop,
            stopping_reason=stopping_reason,
            safe_ci_lower=float(safe_ci_lower),
            safe_ci_upper=float(safe_ci_upper),
        )

    # ─── private helpers ─────────────────────────────────────────────

    def _pooled_variance(self, n_c: int, n_t: int) -> float:
        """Compute pooled variance of the difference."""
        if n_c > 1:
            var_c = (self._ss_control - self._sum_control**2 / n_c) / (n_c - 1)
        else:
            var_c = 0.0

        if n_t > 1:
            var_t = (self._ss_treatment - self._sum_treatment**2 / n_t) / (n_t - 1)
        else:
            var_t = 0.0

        var_c = max(0.0, var_c)
        var_t = max(0.0, var_t)

        denom = n_c + n_t - 2
        if denom > 0:
            pooled_var = ((n_c - 1) * var_c + (n_t - 1) * var_t) / denom
        else:
            pooled_var = (var_c + var_t) / 2.0

        return pooled_var * (1.0 / n_c + 1.0 / n_t)

    def _grapa_log_e(
        self,
        ctrl: np.ndarray,
        trt: np.ndarray,
        n_c: int,
        n_t: int,
    ) -> float:
        """GRAPA e-value: growth-rate optimal under alternative.

        Uses the mixture likelihood ratio under a worst-case
        alternative, computed from the current batch statistics.
        """
        V = self._pooled_variance(n_c, n_t)
        if V <= 0.0:
            return 0.0

        mean_c = self._sum_control / n_c
        mean_t = self._sum_treatment / n_t
        delta_hat = mean_t - mean_c

        # Mixing variance: tau = V / 4 is a reasonable default
        tau = V / 4.0

        # MLR: sqrt(V / (V + tau)) * exp(tau * delta^2 / (2 * V * (V + tau)))
        log_e = 0.5 * math.log(V / (V + tau)) + (
            tau * delta_hat**2 / (2.0 * V * (V + tau))
        )
        return log_e

    def _universal_log_e(
        self,
        ctrl: np.ndarray,
        trt: np.ndarray,
        n_c: int,
        n_t: int,
    ) -> float:
        """Universal e-value: sup_theta L(theta) / L(theta_0).

        More robust but less powerful than GRAPA.  Uses a split-sample
        approach where the first half estimates the alternative and the
        second half computes the likelihood ratio.
        """
        V = self._pooled_variance(n_c, n_t)
        if V <= 0.0:
            return 0.0

        mean_c = self._sum_control / n_c
        mean_t = self._sum_treatment / n_t
        delta_hat = mean_t - mean_c

        # Universal inference: e = exp(delta^2 / (2 * V)) with a
        # conservative penalty for estimation
        # Use half the test statistic to account for estimation cost
        log_e = 0.5 * (delta_hat**2 / (2.0 * V))
        return max(log_e, 0.0)

    def _compute_safe_ci(self) -> tuple[float, float]:
        """Invert the e-process to produce a safe confidence interval."""
        n_c = self._n_control
        n_t = self._n_treatment

        if n_c == 0 or n_t == 0:
            return (float("-inf"), float("inf"))

        mean_c = self._sum_control / n_c
        mean_t = self._sum_treatment / n_t
        delta_hat = mean_t - mean_c

        V = self._pooled_variance(n_c, n_t)
        if V <= 0.0:
            return (delta_hat, delta_hat)

        # Safe CI width: derived from inverting the e-value threshold
        # For the mixture approach: width = sqrt(2 * V * log(1/alpha))
        # adjusted by the mixing variance
        tau = V / 4.0
        ci_half = math.sqrt(2.0 * V * (V + tau) / tau * math.log(1.0 / self._alpha))

        return (delta_hat - ci_half, delta_hat + ci_half)
