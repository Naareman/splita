"""Mixture Sequential Probability Ratio Test (mSPRT).

Always-valid inference for A/B testing that eliminates the peeking problem.

Reference
---------
Johari, Pekelis, Walsh — "Always Valid Inference: Continuous Monitoring of
A/B Tests" (2015/2022).
"""

from __future__ import annotations

import math
import warnings
from typing import Literal

import numpy as np

from splita._types import mSPRTResult, mSPRTState
from splita._utils import ensure_rng
from splita._validation import (
    check_array_like,
    check_in_range,
    check_one_of,
    check_positive,
    format_error,
)

_VALID_METRICS = ["conversion", "continuous"]


class mSPRT:
    """Mixture Sequential Probability Ratio Test.

    Produces always-valid p-values that remain valid regardless of when the
    experimenter peeks at the data.  The peeking problem is eliminated by
    construction.

    Parameters
    ----------
    metric : ``'conversion'`` or ``'continuous'``, default ``'conversion'``
        Type of metric being tested.  ``'conversion'`` for Bernoulli
        (binary) data, ``'continuous'`` for normally-distributed data.
    alpha : float, default 0.05
        Target Type I error rate.  The always-valid guarantee holds for
        any stopping time.
    tau : float or None, default None
        Mixing variance parameter for the prior on the treatment effect.
        If ``None``, auto-tuned from the first batch of data.
    truncation : int or None, default None
        Maximum total sample size (``n_control + n_treatment``).  If
        ``None``, the test runs fully sequentially with no horizon.
    random_state : int, Generator, or None, default None
        Seed or NumPy Generator for any stochastic operations.

    Examples
    --------
    >>> import numpy as np
    >>> from splita.sequential.msprt import mSPRT
    >>> rng = np.random.default_rng(42)
    >>> test = mSPRT(metric="continuous", alpha=0.05)
    >>> ctrl = rng.normal(0.0, 1.0, size=500)
    >>> trt = rng.normal(0.5, 1.0, size=500)
    >>> state = test.update(ctrl, trt)
    >>> state.should_stop
    True
    """

    def __init__(
        self,
        *,
        metric: Literal["conversion", "continuous"] = "conversion",
        alpha: float = 0.05,
        tau: float | None = None,
        truncation: int | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        # ── validation ──
        check_one_of(metric, "metric", _VALID_METRICS)
        check_in_range(alpha, "alpha", 0.0, 1.0)
        if tau is not None:
            check_positive(float(tau), "tau")
        if truncation is not None:
            if not isinstance(truncation, (int, float)):
                raise TypeError(
                    format_error(
                        f"`truncation` must be an integer, got type {type(truncation).__name__}.",
                    )
                )
            if truncation <= 0:
                raise ValueError(
                    format_error(
                        f"`truncation` must be > 0, got {truncation}.",
                        "value must be strictly positive.",
                        "pass the maximum total sample size across both groups.",
                    )
                )

        self._metric = metric
        self._alpha = alpha
        self._tau: float | None = tau
        self._truncation = int(truncation) if truncation is not None else None
        self._rng = ensure_rng(random_state)

        # ── running statistics ──
        self._n_control: int = 0
        self._n_treatment: int = 0
        self._sum_control: float = 0.0
        self._sum_treatment: float = 0.0
        self._ss_control: float = 0.0  # sum of squares (for Welford)
        self._ss_treatment: float = 0.0
        self._updates: int = 0

        # ── latest state ──
        self._state: mSPRTState | None = None

    # ─── public API ──────────────────────────────────────────────────

    def update(
        self,
        control_obs: list | tuple | np.ndarray,
        treatment_obs: list | tuple | np.ndarray,
    ) -> mSPRTState:
        """Ingest new observations and return the current test state.

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
        mSPRTState
            Current state of the test including the always-valid p-value,
            mixture likelihood ratio, and stopping decision.

        Raises
        ------
        ValueError
            If both arrays are empty on the first call (no data to analyse).
        """
        ctrl = check_array_like(control_obs, "control_obs")
        trt = check_array_like(treatment_obs, "treatment_obs")

        # Allow empty arrays as no-ops when we already have data
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
            # No new data — return current state
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

        # Need at least 1 obs in each group for meaningful statistics
        if n_c == 0 or n_t == 0:
            self._state = mSPRTState(
                n_control=n_c,
                n_treatment=n_t,
                mixture_lr=1.0,
                always_valid_pvalue=1.0,
                always_valid_ci_lower=0.0,
                always_valid_ci_upper=0.0,
                should_stop=False,
                current_effect_estimate=0.0,
            )
            return self._state

        # ── compute means and variances ──
        mean_c = self._sum_control / n_c
        mean_t = self._sum_treatment / n_t
        delta_hat = mean_t - mean_c

        if self._metric == "conversion":
            V = self._compute_variance_conversion(mean_c, mean_t, n_c, n_t)
        else:
            V = self._compute_variance_continuous(n_c, n_t)

        # Guard against degenerate variance
        if V <= 0.0:
            self._state = mSPRTState(
                n_control=n_c,
                n_treatment=n_t,
                mixture_lr=1.0,
                always_valid_pvalue=1.0,
                always_valid_ci_lower=delta_hat,
                always_valid_ci_upper=delta_hat,
                should_stop=False,
                current_effect_estimate=delta_hat,
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

        # ── mixture likelihood ratio ──
        mlr = math.sqrt(V / (V + tau)) * math.exp(tau * delta_hat**2 / (2.0 * V * (V + tau)))

        # ── always-valid p-value ──
        p_value = min(1.0, 1.0 / mlr) if mlr > 0.0 else 1.0

        # ── always-valid CI ──
        ci_half = math.sqrt(2.0 * V * (V + tau) / tau * math.log(1.0 / self._alpha))
        ci_lower = delta_hat - ci_half
        ci_upper = delta_hat + ci_half

        # ── stopping decision ──
        should_stop = p_value < self._alpha
        if self._truncation is not None and n_c + n_t >= self._truncation:
            should_stop = True

        self._state = mSPRTState(
            n_control=n_c,
            n_treatment=n_t,
            mixture_lr=float(mlr),
            always_valid_pvalue=float(p_value),
            always_valid_ci_lower=float(ci_lower),
            always_valid_ci_upper=float(ci_upper),
            should_stop=should_stop,
            current_effect_estimate=float(delta_hat),
        )
        return self._state

    def result(self) -> mSPRTResult:
        """Return the final result of the test.

        Should be called after one or more :meth:`update` calls.

        Returns
        -------
        mSPRTResult
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

        # Determine stopping reason
        n_total = self._state.n_control + self._state.n_treatment
        if (
            self._truncation is not None
            and n_total >= self._truncation
            and self._state.always_valid_pvalue >= self._alpha
        ):
            stopping_reason = "truncation"
        elif self._state.should_stop:
            stopping_reason = "boundary_crossed"
        else:
            stopping_reason = "not_stopped"

        return mSPRTResult(
            n_control=self._state.n_control,
            n_treatment=self._state.n_treatment,
            mixture_lr=self._state.mixture_lr,
            always_valid_pvalue=self._state.always_valid_pvalue,
            always_valid_ci_lower=self._state.always_valid_ci_lower,
            always_valid_ci_upper=self._state.always_valid_ci_upper,
            should_stop=self._state.should_stop,
            current_effect_estimate=self._state.current_effect_estimate,
            stopping_reason=stopping_reason,
            total_observations=n_total,
            relative_speedup_vs_fixed_horizon=None,
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
        # Clamp to avoid negative variance with extreme proportions
        p_pool = max(1e-12, min(1.0 - 1e-12, p_pool))
        return p_pool * (1.0 - p_pool) * (1.0 / n_c + 1.0 / n_t)

    def _compute_variance_continuous(
        self,
        n_c: int,
        n_t: int,
    ) -> float:
        """Compute the variance of the difference for continuous metric."""
        # Sample variances via sum-of-squares formulation
        var_c = (self._ss_control - self._sum_control**2 / n_c) / (n_c - 1) if n_c > 1 else 0.0

        if n_t > 1:
            var_t = (self._ss_treatment - self._sum_treatment**2 / n_t) / (n_t - 1)
        else:
            var_t = 0.0

        # Guard against floating-point negative
        var_c = max(0.0, var_c)
        var_t = max(0.0, var_t)

        # Pooled variance
        denom = n_c + n_t - 2
        if denom > 0:
            pooled_var = ((n_c - 1) * var_c + (n_t - 1) * var_t) / denom
        else:
            pooled_var = (var_c + var_t) / 2.0

        return pooled_var * (1.0 / n_c + 1.0 / n_t)
