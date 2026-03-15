"""MetricDecomposition — decompose metric into additive components.

Tests each component separately for higher sensitivity.

References
----------
.. [1] Deng, A. et al.  "Metric decomposition for improved A/B test analysis."
       KDD 2024.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ttest_ind

from splita._types import MetricDecompResult
from splita._validation import (
    check_array_like,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


class MetricDecomposition:
    """Decompose a metric into additive components and test each one.

    Useful when a total metric (e.g. revenue) can be expressed as the
    sum of components (e.g. conversion * AOV). Testing each component
    separately reveals *which part* of the metric was affected and
    provides higher sensitivity when only one component changes.

    Parameters
    ----------
    alpha : float, default 0.05
        Significance level for per-component tests.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n = 500
    >>> ctrl_total = rng.normal(10, 2, n)
    >>> trt_total = rng.normal(10.5, 2, n)
    >>> comps_ctrl = {"conv": rng.normal(5, 1, n), "aov": rng.normal(5, 1, n)}
    >>> comps_trt = {"conv": rng.normal(5.3, 1, n), "aov": rng.normal(5.2, 1, n)}
    >>> result = MetricDecomposition().decompose(
    ...     ctrl_total, trt_total, comps_ctrl, comps_trt
    ... )
    >>> isinstance(result.dominant_component, str)
    True
    """

    def __init__(self, *, alpha: float = 0.05) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(
                format_error(
                    f"`alpha` must be in (0, 1), got {alpha}.",
                    hint="typical values are 0.05, 0.01, or 0.10.",
                )
            )
        self._alpha = alpha

    def decompose(
        self,
        total_control: ArrayLike,
        total_treatment: ArrayLike,
        components_control: dict[str, ArrayLike],
        components_treatment: dict[str, ArrayLike],
    ) -> MetricDecompResult:
        """Decompose and test each component.

        Parameters
        ----------
        total_control : array-like
            Total metric values for the control group.
        total_treatment : array-like
            Total metric values for the treatment group.
        components_control : dict of str to array-like
            Named component arrays for the control group.
        components_treatment : dict of str to array-like
            Named component arrays for the treatment group.

        Returns
        -------
        MetricDecompResult
            Per-component lift, pvalue, and which component dominates.

        Raises
        ------
        ValueError
            If component names don't match, or inputs are invalid.
        """
        ctrl_total = check_array_like(total_control, "total_control", min_length=2)
        trt_total = check_array_like(total_treatment, "total_treatment", min_length=2)

        if not components_control:
            raise ValueError(
                format_error(
                    "`components_control` can't be empty.",
                    detail="at least one component is required for decomposition.",
                    hint="pass a dict mapping component names to arrays.",
                )
            )

        if set(components_control.keys()) != set(components_treatment.keys()):
            raise ValueError(
                format_error(
                    "`components_control` and `components_treatment` must have the same keys.",
                    detail=f"control keys: {sorted(components_control.keys())}, "
                    f"treatment keys: {sorted(components_treatment.keys())}.",
                    hint="ensure both dicts contain the same component names.",
                )
            )

        # Test total metric
        total_res = ttest_ind(trt_total, ctrl_total, equal_var=False)
        total_lift = float(np.mean(trt_total) - np.mean(ctrl_total))
        total_pvalue = float(total_res.pvalue)

        # Test each component
        component_results: dict[str, dict] = {}
        best_component: str | None = None
        best_tstat: float = 0.0

        for name in sorted(components_control.keys()):
            comp_ctrl = check_array_like(
                components_control[name], f"components_control['{name}']", min_length=2
            )
            comp_trt = check_array_like(
                components_treatment[name],
                f"components_treatment['{name}']",
                min_length=2,
            )

            res = ttest_ind(comp_trt, comp_ctrl, equal_var=False)
            lift = float(np.mean(comp_trt) - np.mean(comp_ctrl))
            pvalue = float(res.pvalue)
            significant = pvalue < self._alpha

            # Contribution: what fraction of total lift does this component explain
            contribution = lift / total_lift if total_lift != 0.0 else 0.0

            component_results[name] = {
                "lift": lift,
                "pvalue": pvalue,
                "significant": significant,
                "contribution": contribution,
            }

            if abs(float(res.statistic)) > abs(best_tstat):
                best_tstat = float(res.statistic)
                best_component = name

        # Only mark a dominant component if it is significant
        dominant = None
        if best_component is not None and component_results[best_component]["significant"]:
            dominant = best_component

        return MetricDecompResult(
            total_lift=total_lift,
            total_pvalue=total_pvalue,
            component_results=component_results,
            dominant_component=dominant,
        )
