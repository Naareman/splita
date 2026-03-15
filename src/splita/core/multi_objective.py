"""MultiObjectiveExperiment — multi-metric A/B test analysis.

Collects multiple metrics for the same experiment, runs a standard
:class:`~splita.Experiment` on each, applies multiple testing correction,
and identifies Pareto-optimal decisions.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from splita._types import MultiObjectiveResult
from splita._validation import (
    check_array_like,
    check_in_range,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


class MultiObjectiveExperiment:
    """Analyse multiple metrics for a single A/B experiment.

    Collects control/treatment data for each metric, runs an
    :class:`~splita.Experiment` on each, applies Benjamini-Hochberg
    correction by default, and determines whether the treatment is
    Pareto-dominant, should be rejected, or involves tradeoffs.

    Parameters
    ----------
    alpha : float, default 0.05
        Significance level (before correction).
    metric_names : list[str] or None, default None
        Optional human-readable names for the metrics (in order of
        :meth:`add_metric` calls).  If None, metrics are auto-numbered.
    correction : {'bh', 'bonferroni', 'holm', 'by'}, default 'bh'
        Multiple testing correction method.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> exp = MultiObjectiveExperiment(metric_names=["revenue", "latency"])
    >>> exp.add_metric(rng.normal(10, 2, 500), rng.normal(11, 2, 500))
    >>> exp.add_metric(rng.normal(100, 10, 500), rng.normal(95, 10, 500))
    >>> result = exp.run()
    >>> result.recommendation in ("adopt", "reject", "tradeoff")
    True
    """

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        metric_names: list[str] | None = None,
        correction: Literal["bh", "bonferroni", "holm", "by"] = "bh",
    ) -> None:
        check_in_range(
            alpha,
            "alpha",
            0.0,
            1.0,
            hint="typical values are 0.05, 0.01, or 0.10.",
        )

        self._alpha = alpha
        self._metric_names = list(metric_names) if metric_names is not None else None
        self._correction = correction
        self._metrics: list[tuple[np.ndarray, np.ndarray, str]] = []

    def add_metric(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
        name: str | None = None,
    ) -> None:
        """Add a metric to the experiment.

        Parameters
        ----------
        control : array-like
            Control group observations for this metric.
        treatment : array-like
            Treatment group observations for this metric.
        name : str or None, default None
            Human-readable name.  If None, uses metric_names from init
            or auto-numbers.
        """
        ctrl = check_array_like(control, "control", min_length=2)
        trt = check_array_like(treatment, "treatment", min_length=2)

        idx = len(self._metrics)
        if name is not None:
            metric_name = name
        elif self._metric_names is not None and idx < len(self._metric_names):
            metric_name = self._metric_names[idx]
        else:
            metric_name = f"metric_{idx}"

        self._metrics.append((ctrl, trt, metric_name))

    def run(self) -> MultiObjectiveResult:
        """Run the multi-objective analysis.

        Returns
        -------
        MultiObjectiveResult
            Frozen dataclass with per-metric results, corrected p-values,
            Pareto dominance flag, tradeoff list, and recommendation.

        Raises
        ------
        ValueError
            If no metrics have been added.
        """
        if len(self._metrics) == 0:
            raise ValueError(
                format_error(
                    "No metrics have been added.",
                    hint="call add_metric() at least once before run().",
                )
            )

        from splita.core.correction import MultipleCorrection
        from splita.core.experiment import Experiment

        # Run each metric
        results = []
        for ctrl, trt, _name in self._metrics:
            exp = Experiment(ctrl, trt, alpha=self._alpha)
            result = exp.run()
            results.append(result)

        # Collect raw p-values and apply correction
        raw_pvalues = [r.pvalue for r in results]

        if len(raw_pvalues) == 1:
            # No correction needed for a single metric
            corrected = raw_pvalues
        else:
            correction_result = MultipleCorrection(
                raw_pvalues,
                method=self._correction,
                alpha=self._alpha,
            ).run()
            corrected = correction_result.adjusted_pvalues

        # Determine directions: positive lift = treatment better
        metric_names = [m[2] for m in self._metrics]
        sig_positive = []  # significant metrics where treatment is better
        sig_negative = []  # significant metrics where treatment is worse
        for i, r in enumerate(results):
            if corrected[i] < self._alpha:
                if r.lift > 0:
                    sig_positive.append(metric_names[i])
                else:
                    sig_negative.append(metric_names[i])

        # Pareto dominance: treatment is significantly better on ALL metrics
        all_significant = all(p < self._alpha for p in corrected)
        all_positive = all(r.lift > 0 for r in results)
        pareto_dominant = all_significant and all_positive

        # Tradeoffs: metrics where significant results have conflicting directions
        tradeoffs = []
        if sig_positive and sig_negative:
            tradeoffs = sig_positive + sig_negative

        # Recommendation
        if pareto_dominant:
            recommendation = "adopt"
        elif len(sig_positive) > 0 and len(sig_negative) > 0:
            recommendation = "tradeoff"
        elif (len(sig_negative) > 0 and len(sig_positive) == 0) or (
            all(r.lift <= 0 for r in results) and any(p < self._alpha for p in corrected)
        ):
            recommendation = "reject"
        else:
            # No significant negatives, possibly some significant positives
            # but not all — still could be "adopt" if all point positive
            if all(r.lift > 0 for r in results) and any(p < self._alpha for p in corrected):
                recommendation = "adopt"
            elif len(sig_negative) == 0 and len(sig_positive) == 0:
                # Nothing significant
                recommendation = "reject"
            else:
                recommendation = "tradeoff"

        return MultiObjectiveResult(
            metric_results=results,
            pareto_dominant=pareto_dominant,
            tradeoffs=tradeoffs,
            corrected_pvalues=[float(p) for p in corrected],
            recommendation=recommendation,
        )
