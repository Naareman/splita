"""Always-valid confidence sequences (Howard et al. 2021).

Confidence sequences provide time-uniform confidence intervals that are
valid at every stopping time.  They are tighter than mSPRT confidence
intervals for the same sample size.

Reference
---------
Howard, Ramdas, McAuliffe & Sekhon (2021). "Time-uniform, nonparametric,
nonasymptotic confidence sequences."
"""

from __future__ import annotations

import math
import warnings
from typing import Literal

import numpy as np

from splita._types import CSResult, CSState
from splita._validation import (
    check_array_like,
    check_in_range,
    check_one_of,
    check_positive,
    format_error,
)

_VALID_METHODS = ["normal_mixture", "stitched"]


class ConfidenceSequence:
    """Always-valid confidence sequences for sequential A/B testing.

    Produces confidence intervals that remain valid at every stopping
    time, without pre-specifying a sample size.  Tighter than mSPRT
    confidence intervals at the same sample.

    Parameters
    ----------
    alpha : float, default 0.05
        Target Type I error rate (miscoverage probability).
    sigma : float or None, default None
        Sub-Gaussian parameter (standard deviation).  If ``None``,
        auto-estimated from pooled data on the first update.
    method : ``'normal_mixture'`` or ``'stitched'``, default ``'normal_mixture'``
        CS construction method.  ``'normal_mixture'`` is simpler and
        most practical; ``'stitched'`` is tighter asymptotically.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> cs = ConfidenceSequence(alpha=0.05)
    >>> ctrl = rng.normal(0.0, 1.0, size=500)
    >>> trt = rng.normal(0.5, 1.0, size=500)
    >>> state = cs.update(ctrl, trt)
    >>> state.should_stop
    True
    """

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        sigma: float | None = None,
        method: Literal["normal_mixture", "stitched"] = "normal_mixture",
    ) -> None:
        # ── validation ──
        check_in_range(alpha, "alpha", 0.0, 1.0)
        check_one_of(method, "method", _VALID_METHODS)
        if sigma is not None:
            check_positive(float(sigma), "sigma")

        self._alpha = alpha
        self._sigma: float | None = sigma
        self._method = method

        # ── running statistics ──
        self._n_control: int = 0
        self._n_treatment: int = 0
        self._sum_control: float = 0.0
        self._sum_treatment: float = 0.0
        self._ss_control: float = 0.0
        self._ss_treatment: float = 0.0
        self._updates: int = 0

        # ── latest state ──
        self._state: CSState | None = None

    # ─── public API ──────────────────────────────────────────────────

    def update(
        self,
        control_obs: list | tuple | np.ndarray,
        treatment_obs: list | tuple | np.ndarray,
    ) -> CSState:
        """Ingest new observations and return the current CS state.

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
        CSState
            Current state including the confidence sequence bounds and
            stopping decision.

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
            self._state = CSState(
                n_control=n_c,
                n_treatment=n_t,
                effect_estimate=0.0,
                ci_lower=float("-inf"),
                ci_upper=float("inf"),
                width=float("inf"),
                should_stop=False,
            )
            return self._state

        # ── compute means ──
        mean_c = self._sum_control / n_c
        mean_t = self._sum_treatment / n_t
        delta_hat = mean_t - mean_c

        # ── auto-estimate sigma if needed ──
        sigma = self._sigma
        if sigma is None:
            sigma = self._estimate_pooled_sigma(n_c, n_t)
            self._sigma = sigma
            warnings.warn(
                f"sigma auto-set to {sigma:.6f} from pooled data. "
                f"Set sigma explicitly for reproducibility.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Guard against zero sigma
        if sigma <= 0.0:
            self._state = CSState(
                n_control=n_c,
                n_treatment=n_t,
                effect_estimate=float(delta_hat),
                ci_lower=float(delta_hat),
                ci_upper=float(delta_hat),
                width=0.0,
                should_stop=delta_hat != 0.0,
            )
            return self._state

        # ── compute CS width ──
        if self._method == "normal_mixture":
            w_t = self._normal_mixture_width(sigma, n_c, n_t)
        else:
            w_t = self._stitched_width(sigma, n_c, n_t)

        ci_lower = delta_hat - w_t
        ci_upper = delta_hat + w_t
        width = 2.0 * w_t

        # ── stopping decision: CI excludes zero ──
        should_stop = ci_lower > 0.0 or ci_upper < 0.0

        self._state = CSState(
            n_control=n_c,
            n_treatment=n_t,
            effect_estimate=float(delta_hat),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            width=float(width),
            should_stop=should_stop,
        )
        return self._state

    def result(self) -> CSResult:
        """Return the final result of the confidence sequence.

        Should be called after one or more :meth:`update` calls.

        Returns
        -------
        CSResult
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

        stopping_reason = "ci_excludes_zero" if self._state.should_stop else "not_stopped"

        return CSResult(
            n_control=self._state.n_control,
            n_treatment=self._state.n_treatment,
            effect_estimate=self._state.effect_estimate,
            ci_lower=self._state.ci_lower,
            ci_upper=self._state.ci_upper,
            width=self._state.width,
            should_stop=self._state.should_stop,
            stopping_reason=stopping_reason,
            total_observations=self._state.n_control + self._state.n_treatment,
        )

    # ─── private helpers ─────────────────────────────────────────────

    def _estimate_pooled_sigma(self, n_c: int, n_t: int) -> float:
        """Estimate sigma from pooled sample standard deviation."""
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

        return math.sqrt(max(0.0, pooled_var))

    def _normal_mixture_width(self, sigma: float, n_c: int, n_t: int) -> float:
        """Normal mixture CS width (Howard et al. 2021, Theorem 1).

        w_t = sqrt(2 * v_t * (log(log(2 * max(t, e))) + 0.72 * log(5.2 / alpha)))
        where v_t = sigma^2 * (1/n_c + 1/n_t).
        """
        v_t = sigma**2 * (1.0 / n_c + 1.0 / n_t)
        t = n_c + n_t
        inner = math.log(max(math.log(2.0 * max(t, math.e)), 1e-12))
        boundary = inner + 0.72 * math.log(5.2 / self._alpha)
        # Guard against negative argument to sqrt due to very small samples
        boundary = max(boundary, 0.0)
        return math.sqrt(2.0 * v_t * boundary)

    def _stitched_width(self, sigma: float, n_c: int, n_t: int) -> float:
        """Stitched CS width (Howard et al. 2021, Theorem 2).

        Uses a slightly different boundary function that is tighter
        asymptotically but wider for very small samples.
        """
        v_t = sigma**2 * (1.0 / n_c + 1.0 / n_t)
        t = n_c + n_t
        s = max(t, 2)
        # Stitched boundary: uses iterated log with correction
        log_s = math.log(max(s, math.e))
        eta = 1.7  # stitching parameter
        inner = eta * math.log(log_s) + 0.72 * math.log(5.2 / self._alpha)
        inner = max(inner, 0.0)
        return math.sqrt(2.0 * v_t * inner)
