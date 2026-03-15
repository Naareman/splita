"""E-value sequential testing.

E-values provide an alternative to p-values for sequential hypothesis testing.
The e-value at time t equals the mixture likelihood ratio (MLR), and rejection
occurs when E_t >= 1/alpha.  This provides an always-valid guarantee:
P(exists t: E_t >= 1/alpha) <= alpha.

Reference
---------
Grunwald, de Heide & Koolen (2020). "Safe Testing."
"""

from __future__ import annotations

import math
import warnings
from typing import Literal

import numpy as np

from splita._types import EValueResult, EValueState
from splita._utils import ensure_rng
from splita._validation import (
    check_array_like,
    check_in_range,
    check_one_of,
    check_positive,
    format_error,
)

_VALID_METRICS = ["conversion", "continuous"]


class EValue:
    """E-value based sequential testing.

    E-values are reciprocals of always-valid p-values derived from the
    mixture likelihood ratio.  Rejection occurs when ``e_value >= 1/alpha``,
    providing an anytime-valid guarantee against peeking.

    Parameters
    ----------
    alpha : float, default 0.05
        Target Type I error rate.
    metric : ``'conversion'`` or ``'continuous'``, default ``'continuous'``
        Type of metric being tested.
    tau : float or None, default None
        Mixing variance parameter.  If ``None``, auto-tuned from the first
        batch of data.
    random_state : int, Generator, or None, default None
        Seed or NumPy Generator for any stochastic operations.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> ev = EValue(alpha=0.05, metric="continuous")
    >>> ctrl = rng.normal(0.0, 1.0, size=500)
    >>> trt = rng.normal(0.5, 1.0, size=500)
    >>> state = ev.update(ctrl, trt)
    >>> state.e_value > 1.0
    True
    """

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        metric: Literal["conversion", "continuous"] = "continuous",
        tau: float | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        # ── validation ──
        check_in_range(alpha, "alpha", 0.0, 1.0)
        check_one_of(metric, "metric", _VALID_METRICS)
        if tau is not None:
            check_positive(float(tau), "tau")

        self._alpha = alpha
        self._metric = metric
        self._tau: float | None = tau
        self._rng = ensure_rng(random_state)

        # ── running statistics ──
        self._n_control: int = 0
        self._n_treatment: int = 0
        self._sum_control: float = 0.0
        self._sum_treatment: float = 0.0
        self._ss_control: float = 0.0
        self._ss_treatment: float = 0.0
        self._updates: int = 0

        # ── latest state ──
        self._state: EValueState | None = None

    # ─── public API ──────────────────────────────────────────────────

    def update(
        self,
        control_obs: list | tuple | np.ndarray,
        treatment_obs: list | tuple | np.ndarray,
    ) -> EValueState:
        """Ingest new observations and return the current e-value state.

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
        EValueState
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
            self._state = EValueState(
                e_value=1.0,
                n_control=n_c,
                n_treatment=n_t,
                should_stop=False,
            )
            return self._state

        # ── compute means and variance ──
        mean_c = self._sum_control / n_c
        mean_t = self._sum_treatment / n_t
        delta_hat = mean_t - mean_c

        if self._metric == "conversion":
            V = self._compute_variance_conversion(mean_c, mean_t, n_c, n_t)
        else:
            V = self._compute_variance_continuous(n_c, n_t)

        # Guard against degenerate variance
        if V <= 0.0:
            self._state = EValueState(
                e_value=1.0,
                n_control=n_c,
                n_treatment=n_t,
                should_stop=False,
            )
            return self._state

        # ── auto-tune tau if needed ──
        if self._tau is None:
            self._tau = V / 4.0
            warnings.warn(
                f"tau auto-set to {self._tau:.6f} from first batch. "
                f"Set tau explicitly for reproducibility.",
                RuntimeWarning,
                stacklevel=2,
            )

        tau = self._tau

        # ── e-value = mixture likelihood ratio ──
        e_value = math.sqrt(V / (V + tau)) * math.exp(tau * delta_hat**2 / (2.0 * V * (V + tau)))

        # ── stopping decision ──
        threshold = 1.0 / self._alpha
        should_stop = e_value >= threshold

        self._state = EValueState(
            e_value=float(e_value),
            n_control=n_c,
            n_treatment=n_t,
            should_stop=should_stop,
        )
        return self._state

    def result(self) -> EValueResult:
        """Return the final result of the test.

        Should be called after one or more :meth:`update` calls.

        Returns
        -------
        EValueResult
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

        stopping_reason = "e_value_threshold_crossed" if self._state.should_stop else "not_stopped"

        return EValueResult(
            e_value=self._state.e_value,
            n_control=self._state.n_control,
            n_treatment=self._state.n_treatment,
            should_stop=self._state.should_stop,
            stopping_reason=stopping_reason,
        )

    # ─── private helpers ─────────────────────────────────────────────

    def _compute_variance_conversion(
        self,
        mean_c: float,
        mean_t: float,
        n_c: int,
        n_t: int,
    ) -> float:
        """Compute the variance of the difference for conversion metric."""
        p_pool = (self._sum_control + self._sum_treatment) / (n_c + n_t)
        p_pool = max(1e-12, min(1.0 - 1e-12, p_pool))
        return p_pool * (1.0 - p_pool) * (1.0 / n_c + 1.0 / n_t)

    def _compute_variance_continuous(
        self,
        n_c: int,
        n_t: int,
    ) -> float:
        """Compute the variance of the difference for continuous metric."""
        var_c = (self._ss_control - self._sum_control**2 / n_c) / (n_c - 1) if n_c > 1 else 0.0

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
