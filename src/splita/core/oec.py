"""OECBuilder -- Combine multiple metrics into a single Overall Evaluation Criterion.

The OEC is a weighted sum of (optionally z-score-normalised) metrics.
Lower-is-better metrics are flipped so that a positive OEC always means
"treatment is better".  The combined score is tested with a two-sample
t-test.

References
----------
.. [1] Deng, A., Xu, Y., Kohavi, R. & Walker, T. "Improving the Sensitivity
       of Online Controlled Experiments by Utilizing Pre-Experiment Data."
       WSDM, 2013.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.stats import ttest_ind, norm

from splita._types import OECResult
from splita._validation import (
    check_array_like,
    check_positive,
    check_one_of,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


class OECBuilder:
    """Build and evaluate an Overall Evaluation Criterion from multiple metrics.

    Parameters
    ----------
    normalize : bool, default True
        If True, each metric is z-score normalised before weighting.
        Recommended when metrics have different scales.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> builder = OECBuilder()
    >>> builder.add_metric("revenue", rng.normal(10, 2, 500), rng.normal(11, 2, 500))
    >>> builder.add_metric("latency", rng.normal(200, 30, 500), rng.normal(190, 30, 500),
    ...                    direction="lower_is_better")
    >>> result = builder.run()
    >>> result.significant
    True
    """

    _DIRECTIONS = ("higher_is_better", "lower_is_better")

    def __init__(self, *, normalize: bool = True) -> None:
        if not isinstance(normalize, bool):
            raise TypeError(
                format_error(
                    "`normalize` must be a bool.",
                    detail=f"got type {type(normalize).__name__}.",
                )
            )
        self._normalize = normalize
        self._metrics: list[dict] = []

    def add_metric(
        self,
        name: str,
        control: ArrayLike,
        treatment: ArrayLike,
        weight: float = 1.0,
        direction: Literal["higher_is_better", "lower_is_better"] = "higher_is_better",
    ) -> None:
        """Register a metric for inclusion in the OEC.

        Parameters
        ----------
        name : str
            Human-readable metric name.
        control : array-like
            Control group observations for this metric.
        treatment : array-like
            Treatment group observations for this metric.
        weight : float, default 1.0
            Relative importance weight (must be positive).
        direction : ``"higher_is_better"`` or ``"lower_is_better"``, default ``"higher_is_better"``
            Whether higher values of the raw metric are desirable.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                format_error(
                    "`name` must be a non-empty string.",
                    detail=f"got {name!r}.",
                )
            )
        check_positive(weight, "weight")
        check_one_of(direction, "direction", list(self._DIRECTIONS))

        ctrl = check_array_like(control, f"control ({name})", min_length=2)
        trt = check_array_like(treatment, f"treatment ({name})", min_length=2)

        self._metrics.append(
            {
                "name": name,
                "control": ctrl,
                "treatment": trt,
                "weight": weight,
                "direction": direction,
            }
        )

    def run(self) -> OECResult:
        """Compute the OEC and test for significance.

        Returns
        -------
        OECResult
            Frozen dataclass with OEC lift, p-value, CI, and per-metric
            contributions.

        Raises
        ------
        ValueError
            If no metrics have been added.
        """
        if not self._metrics:
            raise ValueError(
                format_error(
                    "No metrics have been added.",
                    hint="call add_metric() at least once before run().",
                )
            )

        # Normalise weights to sum to 1
        raw_weights = np.array([m["weight"] for m in self._metrics])
        norm_weights = raw_weights / raw_weights.sum()

        # Find the minimum sample size across all metrics for alignment
        min_n_ctrl = min(len(m["control"]) for m in self._metrics)
        min_n_trt = min(len(m["treatment"]) for m in self._metrics)

        # Build per-user OEC scores
        oec_ctrl = np.zeros(min_n_ctrl)
        oec_trt = np.zeros(min_n_trt)
        contributions = []

        for i, m in enumerate(self._metrics):
            ctrl = m["control"][:min_n_ctrl].copy()
            trt = m["treatment"][:min_n_trt].copy()

            # Flip direction
            if m["direction"] == "lower_is_better":
                ctrl = -ctrl
                trt = -trt

            # Normalise to z-scores
            if self._normalize:
                pooled = np.concatenate([ctrl, trt])
                mu = pooled.mean()
                sigma = pooled.std(ddof=1)
                if sigma > 0:
                    ctrl = (ctrl - mu) / sigma
                    trt = (trt - mu) / sigma

            w = float(norm_weights[i])
            oec_ctrl += w * ctrl
            oec_trt += w * trt
            contributions.append(float(w * (trt.mean() - ctrl.mean())))

        # Two-sample t-test
        stat, pvalue = ttest_ind(oec_trt, oec_ctrl, equal_var=False)
        oec_lift = float(oec_trt.mean() - oec_ctrl.mean())

        # CI via normal approximation
        se = float(np.sqrt(
            np.var(oec_ctrl, ddof=1) / len(oec_ctrl)
            + np.var(oec_trt, ddof=1) / len(oec_trt)
        ))
        z_crit = float(norm.ppf(0.975))
        ci_lower = oec_lift - z_crit * se
        ci_upper = oec_lift + z_crit * se

        return OECResult(
            oec_lift=oec_lift,
            pvalue=float(pvalue),
            significant=float(pvalue) < 0.05,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            metric_contributions=contributions,
            weights=[float(w) for w in norm_weights],
        )
