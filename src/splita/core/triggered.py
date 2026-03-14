"""Triggered experiment and interaction test.

Provides :class:`TriggeredExperiment` for ITT / per-protocol analysis
when only a subset of users are actually exposed (triggered), and
:class:`InteractionTest` for detecting whether treatment effects differ
across segments.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import chi2

from splita._types import InteractionResult, TriggeredResult
from splita._validation import (
    check_array_like,
    check_in_range,
    check_same_length,
    format_error,
)
from splita.core.experiment import Experiment

ArrayLike = list | tuple | np.ndarray


class TriggeredExperiment:
    """Analyse a triggered (partial-exposure) experiment.

    Computes both intent-to-treat (ITT) and per-protocol analyses.

    Parameters
    ----------
    control : array-like
        Outcome values for all users assigned to control.
    treatment : array-like
        Outcome values for all users assigned to treatment.
    control_triggered : array-like or None, default None
        Boolean mask indicating which control users were triggered.
        If ``None``, all control users are assumed triggered.
    treatment_triggered : array-like or None, default None
        Boolean mask indicating which treatment users were triggered.
        If ``None``, all treatment users are assumed triggered.
    **kwargs
        Additional keyword arguments passed to :class:`Experiment`
        (e.g. ``alpha``, ``method``, ``metric``).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> ctrl = rng.normal(10, 2, 500)
    >>> trt = rng.normal(11, 2, 500)
    >>> triggered_ctrl = np.ones(500, dtype=bool)
    >>> triggered_trt = rng.random(500) > 0.3
    >>> result = TriggeredExperiment(
    ...     ctrl, trt,
    ...     control_triggered=triggered_ctrl,
    ...     treatment_triggered=triggered_trt,
    ... ).run()
    >>> result.trigger_rate_treatment < 1.0
    True
    """

    def __init__(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
        *,
        control_triggered: ArrayLike | None = None,
        treatment_triggered: ArrayLike | None = None,
        **kwargs: Any,
    ):
        self._control = check_array_like(control, "control", min_length=2)
        self._treatment = check_array_like(treatment, "treatment", min_length=2)

        # Process trigger masks
        if control_triggered is not None:
            self._ctrl_triggered = np.asarray(control_triggered, dtype=bool)
            check_same_length(
                self._control,
                self._ctrl_triggered,
                "control",
                "control_triggered",
            )
        else:
            self._ctrl_triggered = np.ones(len(self._control), dtype=bool)

        if treatment_triggered is not None:
            self._trt_triggered = np.asarray(treatment_triggered, dtype=bool)
            check_same_length(
                self._treatment,
                self._trt_triggered,
                "treatment",
                "treatment_triggered",
            )
        else:
            self._trt_triggered = np.ones(len(self._treatment), dtype=bool)

        self._kwargs = kwargs

    def run(self) -> TriggeredResult:
        """Execute both ITT and per-protocol analyses.

        Returns
        -------
        TriggeredResult
            Frozen dataclass with ITT result, per-protocol result,
            and trigger rates.

        Raises
        ------
        ValueError
            If either triggered subset has fewer than 2 observations.
        """
        # ITT: all assigned users
        itt_result = Experiment(self._control, self._treatment, **self._kwargs).run()

        # Per-protocol: only triggered users
        ctrl_pp = self._control[self._ctrl_triggered]
        trt_pp = self._treatment[self._trt_triggered]

        if len(ctrl_pp) < 2:
            raise ValueError(
                format_error(
                    "Triggered control group must have at least 2 observations.",
                    f"only {len(ctrl_pp)} control users were triggered.",
                    "check your control_triggered mask.",
                )
            )
        if len(trt_pp) < 2:
            raise ValueError(
                format_error(
                    "Triggered treatment group must have at least 2 observations.",
                    f"only {len(trt_pp)} treatment users were triggered.",
                    "check your treatment_triggered mask.",
                )
            )

        pp_result = Experiment(ctrl_pp, trt_pp, **self._kwargs).run()

        trigger_rate_ctrl = float(np.mean(self._ctrl_triggered))
        trigger_rate_trt = float(np.mean(self._trt_triggered))

        return TriggeredResult(
            itt_result=itt_result,
            per_protocol_result=pp_result,
            trigger_rate_control=trigger_rate_ctrl,
            trigger_rate_treatment=trigger_rate_trt,
        )


class InteractionTest:
    """Test whether the treatment effect differs across segments.

    Runs separate experiments per segment and tests for interaction
    (heterogeneity) using a chi-square test on the segment-level effects.

    Parameters
    ----------
    control : array-like
        Outcome values for the control group.
    treatment : array-like
        Outcome values for the treatment group.
    segments : array-like
        Segment labels for each observation.  Must have length equal to
        ``len(control) + len(treatment)``; the first ``len(control)``
        entries correspond to the control group, the rest to treatment.
    alpha : float, default 0.05
        Significance level for the interaction test.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> ctrl = rng.normal(10, 2, 200)
    >>> trt = rng.normal(11, 2, 200)
    >>> segs = np.array(["A"] * 100 + ["B"] * 100 + ["A"] * 100 + ["B"] * 100)
    >>> result = InteractionTest(ctrl, trt, segments=segs).run()
    >>> isinstance(result.has_interaction, bool)
    True
    """

    def __init__(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
        *,
        segments: ArrayLike,
        alpha: float = 0.05,
    ):
        self._control = check_array_like(control, "control", min_length=2)
        self._treatment = check_array_like(treatment, "treatment", min_length=2)
        check_in_range(
            alpha,
            "alpha",
            0.0,
            1.0,
            hint="typical values are 0.05, 0.01, or 0.10",
        )
        self._alpha = alpha

        segments_arr = np.asarray(segments)
        expected_len = len(self._control) + len(self._treatment)
        if len(segments_arr) != expected_len:
            raise ValueError(
                format_error(
                    "`segments` must have length equal to "
                    "len(control) + len(treatment).",
                    f"expected {expected_len}, got {len(segments_arr)}.",
                    "concatenate segment labels in [control, treatment] order.",
                )
            )

        self._segments_ctrl = segments_arr[: len(self._control)]
        self._segments_trt = segments_arr[len(self._control) :]

    def run(self) -> InteractionResult:
        """Run the interaction test.

        Returns
        -------
        InteractionResult
            Frozen dataclass with per-segment results, interaction
            p-value, and strongest segment.

        Raises
        ------
        ValueError
            If any segment has fewer than 2 observations in either group.
        """
        unique_segments = np.unique(
            np.concatenate([self._segments_ctrl, self._segments_trt])
        )

        if len(unique_segments) < 2:
            raise ValueError(
                format_error(
                    "InteractionTest requires at least 2 unique segments.",
                    f"got {len(unique_segments)} unique segment(s).",
                    "pass segment labels with at least 2 distinct values.",
                )
            )

        segment_results: list[dict[str, Any]] = []
        effects: list[float] = []
        se_list: list[float] = []

        for seg in unique_segments:
            ctrl_mask = self._segments_ctrl == seg
            trt_mask = self._segments_trt == seg
            ctrl_seg = self._control[ctrl_mask]
            trt_seg = self._treatment[trt_mask]

            if len(ctrl_seg) < 2:
                raise ValueError(
                    format_error(
                        f"Segment {seg!r} has fewer than 2 control observations.",
                        f"got {len(ctrl_seg)} control observations.",
                        "ensure each segment has sufficient data.",
                    )
                )
            if len(trt_seg) < 2:
                raise ValueError(
                    format_error(
                        f"Segment {seg!r} has fewer than 2 treatment observations.",
                        f"got {len(trt_seg)} treatment observations.",
                        "ensure each segment has sufficient data.",
                    )
                )

            exp_result = Experiment(ctrl_seg, trt_seg, alpha=self._alpha).run()

            lift = exp_result.lift
            # Approximate SE from CI width
            ci_width = exp_result.ci_upper - exp_result.ci_lower
            z_crit = float(norm_ppf(1 - self._alpha / 2))
            se = ci_width / (2.0 * z_crit) if z_crit > 0 else 1.0

            effects.append(lift)
            se_list.append(se)

            segment_results.append(
                {
                    "segment": str(seg),
                    "lift": lift,
                    "pvalue": exp_result.pvalue,
                    "ci_lower": exp_result.ci_lower,
                    "ci_upper": exp_result.ci_upper,
                    "control_n": exp_result.control_n,
                    "treatment_n": exp_result.treatment_n,
                    "significant": exp_result.significant,
                }
            )

        # Cochran's Q test for heterogeneity
        effects_arr = np.array(effects)
        se_arr = np.array(se_list)
        weights = 1.0 / (se_arr**2 + 1e-12)
        weighted_mean = np.sum(weights * effects_arr) / np.sum(weights)
        q_stat = float(np.sum(weights * (effects_arr - weighted_mean) ** 2))
        df = len(unique_segments) - 1
        interaction_pvalue = float(chi2.sf(q_stat, df))

        has_interaction = interaction_pvalue < self._alpha

        # Find strongest segment
        abs_effects = [abs(e) for e in effects]
        strongest_idx = int(np.argmax(abs_effects))
        strongest_segment = str(unique_segments[strongest_idx])

        return InteractionResult(
            segment_results=segment_results,
            has_interaction=has_interaction,
            interaction_pvalue=interaction_pvalue,
            strongest_segment=strongest_segment,
        )


def norm_ppf(q: float) -> float:
    """Thin wrapper around scipy.stats.norm.ppf for module-level use."""
    from scipy.stats import norm

    return float(norm.ppf(q))
