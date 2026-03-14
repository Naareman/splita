"""StratifiedExperiment — Stratified A/B test with Neyman-style variance weighting.

Groups data by stratum labels, computes within-stratum treatment effects,
and combines them using inverse-variance (Neyman) weighting for a more
precise estimate of the average treatment effect.

References
----------
.. [1] Neyman, J. "On the Application of Probability Theory to Agricultural
       Experiments." Statistical Science, 5(4), 1990.
.. [2] Miratrix, L. W., Sekhon, J. S. & Yu, B. "Adjusting treatment effect
       estimates by post-stratification in randomized experiments."
       JRSS-B, 75(2), 2013.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from splita._types import StratifiedResult
from splita._validation import (
    check_array_like,
    check_in_range,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


class StratifiedExperiment:
    """Run a stratified A/B test with Neyman-style variance weighting.

    Stratification reduces variance when the outcome differs across
    strata.  Within each stratum the treatment effect is estimated
    separately, then combined using weights proportional to stratum
    size (``w_s = n_s / N``).

    Parameters
    ----------
    control : array-like
        Observations from the control group.
    treatment : array-like
        Observations from the treatment group.
    control_strata : array-like
        Stratum labels for each control observation (same length as
        *control*).  Labels may be strings or integers.
    treatment_strata : array-like
        Stratum labels for each treatment observation (same length as
        *treatment*).
    alpha : float, default 0.05
        Significance level for the two-sided test.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> ctrl = np.concatenate([rng.normal(5, 1, 50), rng.normal(10, 1, 50)])
    >>> trt = np.concatenate([rng.normal(5.5, 1, 50), rng.normal(10.5, 1, 50)])
    >>> strata_c = np.array(['A'] * 50 + ['B'] * 50)
    >>> strata_t = np.array(['A'] * 50 + ['B'] * 50)
    >>> result = StratifiedExperiment(
    ...     ctrl, trt, control_strata=strata_c, treatment_strata=strata_t,
    ... ).run()
    >>> result.n_strata
    2
    """

    def __init__(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
        *,
        control_strata: ArrayLike,
        treatment_strata: ArrayLike,
        alpha: float = 0.05,
    ) -> None:
        # ── validate alpha ──────────────────────────────────────────
        check_in_range(
            alpha,
            "alpha",
            0.0,
            1.0,
            hint="typical values are 0.05, 0.01, or 0.10",
        )

        # ── validate arrays ─────────────────────────────────────────
        self._control = check_array_like(control, "control", min_length=2)
        self._treatment = check_array_like(treatment, "treatment", min_length=2)

        # ── validate strata (categorical, not float) ────────────────
        self._control_strata = self._validate_strata(
            control_strata, "control_strata", len(self._control)
        )
        self._treatment_strata = self._validate_strata(
            treatment_strata, "treatment_strata", len(self._treatment)
        )

        # ── check strata overlap ────────────────────────────────────
        ctrl_labels = set(self._control_strata)
        trt_labels = set(self._treatment_strata)
        if not ctrl_labels & trt_labels:
            raise ValueError(
                format_error(
                    "Control and treatment strata must share at least one label.",
                    detail=f"control labels: {sorted(ctrl_labels)}, "
                    f"treatment labels: {sorted(trt_labels)}.",
                    hint="ensure both groups use the same stratum naming.",
                )
            )

        self._alpha = alpha

    # ── private helpers ──────────────────────────────────────────────

    @staticmethod
    def _validate_strata(strata: ArrayLike, name: str, expected_len: int) -> np.ndarray:
        """Validate and convert strata labels to a numpy array."""
        if not isinstance(strata, (list, tuple, np.ndarray)):
            raise TypeError(
                format_error(
                    f"`{name}` must be array-like (list, tuple, or ndarray), "
                    f"got type {type(strata).__name__}.",
                )
            )
        arr = np.asarray(strata)
        if arr.ndim != 1:
            raise ValueError(
                format_error(
                    f"`{name}` must be a 1-D array, got {arr.ndim}-D.",
                    hint="pass a flat list or 1-D array of stratum labels.",
                )
            )
        if len(arr) != expected_len:
            raise ValueError(
                format_error(
                    f"`{name}` must have the same length as its data array.",
                    detail=f"expected {expected_len} elements, got {len(arr)}.",
                )
            )
        return arr

    # ── public API ───────────────────────────────────────────────────

    def run(self) -> StratifiedResult:
        """Execute the stratified test and return results.

        Returns
        -------
        StratifiedResult
            Frozen dataclass with the stratified ATE, SE, p-value,
            confidence interval, and per-stratum effects.

        Raises
        ------
        ValueError
            If any stratum has fewer than 2 observations in either group.
        """
        # Only use strata present in BOTH groups
        ctrl_labels = set(self._control_strata)
        trt_labels = set(self._treatment_strata)
        common_labels = sorted(ctrl_labels & trt_labels, key=str)

        N = 0  # total observations across common strata
        strata_sizes: dict[str, int] = {}

        # First pass: compute total N for common strata
        for label in common_labels:
            n_c = int(np.sum(self._control_strata == label))
            n_t = int(np.sum(self._treatment_strata == label))
            strata_sizes[str(label)] = n_c + n_t
            N += n_c + n_t

        if N == 0:
            raise ValueError(
                format_error(
                    "No observations found in common strata.",
                    hint="check that strata labels match between groups.",
                )
            )

        # Second pass: compute per-stratum effects
        stratum_effects: list[dict] = []
        ate = 0.0
        var_ate = 0.0

        for label in common_labels:
            ctrl_mask = self._control_strata == label
            trt_mask = self._treatment_strata == label

            y_ctrl = self._control[ctrl_mask]
            y_trt = self._treatment[trt_mask]

            n_c = len(y_ctrl)
            n_t = len(y_trt)

            if n_c < 2:
                raise ValueError(
                    format_error(
                        f"Stratum {label!r} has fewer than 2 control observations.",
                        detail=f"found {n_c} control obs.",
                        hint="ensure each stratum has >= 2 obs per group.",
                    )
                )
            if n_t < 2:
                raise ValueError(
                    format_error(
                        f"Stratum {label!r} has fewer than 2 treatment observations.",
                        detail=f"found {n_t} treatment obs.",
                        hint="ensure each stratum has >= 2 obs per group.",
                    )
                )

            ate_s = float(np.mean(y_trt) - np.mean(y_ctrl))
            var_ctrl_s = float(np.var(y_ctrl, ddof=1))
            var_trt_s = float(np.var(y_trt, ddof=1))
            se_s = float(np.sqrt(var_ctrl_s / n_c + var_trt_s / n_t))

            w_s = strata_sizes[str(label)] / N

            ate += w_s * ate_s
            var_ate += w_s**2 * (var_ctrl_s / n_c + var_trt_s / n_t)

            stratum_effects.append(
                {
                    "stratum": (
                        label if not isinstance(label, np.generic) else label.item()
                    ),
                    "ate": ate_s,
                    "se": se_s,
                    "n_control": n_c,
                    "n_treatment": n_t,
                    "weight": w_s,
                }
            )

        se = float(np.sqrt(var_ate))

        # z-test
        z = ate / se if se > 0.0 else 0.0
        pvalue = float(2 * norm.sf(abs(z)))

        z_crit = float(norm.ppf(1 - self._alpha / 2))
        ci_lower = ate - z_crit * se
        ci_upper = ate + z_crit * se

        return StratifiedResult(
            ate=ate,
            se=se,
            pvalue=pvalue,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            significant=pvalue < self._alpha,
            n_strata=len(common_labels),
            stratum_effects=stratum_effects,
            alpha=self._alpha,
        )
