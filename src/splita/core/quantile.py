"""QuantileExperiment — test differences at arbitrary quantiles using bootstrap.

Compares quantile values (median, p90, p99, etc.) between control and treatment
groups using bootstrap inference, since there is no closed-form confidence
interval for quantile differences.
"""

from __future__ import annotations

import numpy as np

from splita._types import QuantileResult
from splita._utils import ensure_rng
from splita._validation import (
    check_array_like,
    check_in_range,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


class QuantileExperiment:
    """Test differences at arbitrary quantiles using bootstrap inference.

    Parameters
    ----------
    control : array-like
        Observations from the control group.
    treatment : array-like
        Observations from the treatment group.
    quantiles : list[float] or float, default 0.5
        Quantile(s) to test, each in (0, 1).  A single float is treated
        as a one-element list.
    alpha : float, default 0.05
        Significance level.
    n_bootstrap : int, default 5000
        Number of bootstrap resamples.
    random_state : int, Generator, or None, default None
        Seed for reproducibility of bootstrap resampling.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> ctrl = rng.normal(10, 2, size=500)
    >>> trt = ctrl + 1
    >>> result = QuantileExperiment(ctrl, trt, quantiles=0.5, random_state=0).run()
    >>> result.significant[0]
    True
    """

    def __init__(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
        *,
        quantiles: list[float] | float = 0.5,
        alpha: float = 0.05,
        n_bootstrap: int = 5000,
        random_state: int | np.random.Generator | None = None,
    ):
        # ── validate alpha ──────────────────────────────────────────
        check_in_range(
            alpha,
            "alpha",
            0.0,
            1.0,
            hint="typical values are 0.05, 0.01, or 0.10",
        )

        # ── validate n_bootstrap ────────────────────────────────────
        if n_bootstrap < 100:
            raise ValueError(
                format_error(
                    f"`n_bootstrap` must be >= 100, got {n_bootstrap}.",
                    "too few bootstrap iterations for reliable inference.",
                    "typical values are 1000-10000.",
                )
            )
        if n_bootstrap > 1_000_000:
            raise ValueError(
                format_error(
                    f"`n_bootstrap` must be at most 1,000,000, got {n_bootstrap}.",
                    "very large bootstrap counts cause excessive memory allocation.",
                    "2,000-10,000 is sufficient for most use cases.",
                )
            )

        # ── validate quantiles ──────────────────────────────────────
        if isinstance(quantiles, (int, float)):
            quantiles = [float(quantiles)]
        else:
            quantiles = [float(q) for q in quantiles]

        if len(quantiles) == 0:
            raise ValueError(
                format_error(
                    "`quantiles` can't be empty.",
                    "received a list with 0 elements.",
                    "pass a float or list of floats, e.g. [0.25, 0.5, 0.75].",
                )
            )

        for q in quantiles:
            check_in_range(
                q,
                "quantiles",
                0.0,
                1.0,
                hint="quantiles must be between 0 and 1 exclusive, e.g. 0.5 for median.",
            )

        # ── convert & clean arrays ──────────────────────────────────
        self._control = check_array_like(
            control,
            "control",
            min_length=2,
        )
        self._treatment = check_array_like(
            treatment,
            "treatment",
            min_length=2,
        )

        # ── store config ────────────────────────────────────────────
        self._quantiles = quantiles
        self._alpha = alpha
        self._n_bootstrap = n_bootstrap
        self._rng = ensure_rng(random_state)

    def run(self) -> QuantileResult:
        """Execute the bootstrap quantile test and return results.

        For each quantile *q*:

        1. Compute observed quantile difference:
           ``diff = np.quantile(treatment, q) - np.quantile(control, q)``
        2. Bootstrap: resample control and treatment *n_bootstrap* times,
           compute quantile difference each time.
        3. CI: percentile method on bootstrap differences.
        4. P-value: shifted bootstrap (centre bootstrap diffs under H0).

        Returns
        -------
        QuantileResult
            Frozen dataclass with per-quantile test outputs.
        """
        ctrl = self._control
        trt = self._treatment
        n1 = len(ctrl)
        n2 = len(trt)
        n_boot = self._n_bootstrap
        rng = self._rng

        # Vectorized bootstrap resampling (done once, reused for all quantiles)
        boot_ctrl_idx = rng.integers(0, n1, size=(n_boot, n1))
        boot_trt_idx = rng.integers(0, n2, size=(n_boot, n2))
        boot_ctrl = ctrl[boot_ctrl_idx]  # (n_boot, n1)
        boot_trt = trt[boot_trt_idx]  # (n_boot, n2)

        control_quantiles: list[float] = []
        treatment_quantiles: list[float] = []
        differences: list[float] = []
        ci_lower: list[float] = []
        ci_upper: list[float] = []
        pvalues: list[float] = []
        significant: list[bool] = []

        for q in self._quantiles:
            # Observed quantile values
            q_ctrl = float(np.quantile(ctrl, q))
            q_trt = float(np.quantile(trt, q))
            observed_diff = q_trt - q_ctrl

            # Bootstrap quantile differences (vectorized across bootstrap samples)
            boot_q_ctrl = np.quantile(boot_ctrl, q, axis=1)  # (n_boot,)
            boot_q_trt = np.quantile(boot_trt, q, axis=1)  # (n_boot,)
            boot_diffs = boot_q_trt - boot_q_ctrl  # (n_boot,)

            # CI: percentile method
            lo = float(np.percentile(boot_diffs, 100 * self._alpha / 2))
            hi = float(np.percentile(boot_diffs, 100 * (1 - self._alpha / 2)))

            # P-value: shifted bootstrap (two-sided)
            centered_diffs = boot_diffs - np.mean(boot_diffs)
            pval = float(np.mean(np.abs(centered_diffs) >= abs(observed_diff)))
            # Clamp to avoid p=0 (minimum is 1/n_bootstrap)
            pval = max(pval, 1.0 / n_boot)

            control_quantiles.append(q_ctrl)
            treatment_quantiles.append(q_trt)
            differences.append(observed_diff)
            ci_lower.append(lo)
            ci_upper.append(hi)
            pvalues.append(pval)
            significant.append(pval < self._alpha)

        return QuantileResult(
            quantiles=self._quantiles,
            control_quantiles=control_quantiles,
            treatment_quantiles=treatment_quantiles,
            differences=differences,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            pvalues=pvalues,
            significant=significant,
            alpha=self._alpha,
            control_n=n1,
            treatment_n=n2,
        )
