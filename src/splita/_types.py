from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Any


def _to_python(val: Any) -> Any:
    """Convert numpy scalars and arrays to plain Python types."""
    try:
        import numpy as np

        if isinstance(val, np.integer):
            return int(val)
        if isinstance(val, np.floating):
            return float(val)
        if isinstance(val, np.bool_):
            return bool(val)
        if isinstance(val, np.ndarray):
            return val.tolist()
    except ImportError:  # pragma: no cover
        pass

    if isinstance(val, list):
        return [_to_python(v) for v in val]
    if isinstance(val, tuple):
        return tuple(_to_python(v) for v in val)
    if isinstance(val, dict):
        return {k: _to_python(v) for k, v in val.items()}
    return val


def _line(width: int = 36) -> str:
    return "\u2500" * width


def _fmt(val: Any, as_pct: bool = False) -> str:
    if isinstance(val, bool):
        return f"{val} \u2713" if val else f"{val} \u2717"
    if isinstance(val, float):
        if as_pct:
            return f"{val * 100:.2f}%"
        if abs(val) < 0.0001 and val != 0.0:
            return f"{val:.2e}"
        return f"{val:.4f}"
    return str(val)


class _DictMixin:
    """Mixin providing to_dict() for frozen dataclasses."""

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a plain Python dictionary.

        NumPy scalars and arrays are converted to built-in Python types
        so that the dictionary is JSON-serialisable.

        Returns
        -------
        dict[str, Any]
            Dictionary with one key per dataclass field.

        Examples
        --------
        >>> from splita import ExperimentResult
        >>> r = ExperimentResult(
        ...     control_mean=0.10, treatment_mean=0.12,
        ...     lift=0.02, relative_lift=0.2, pvalue=0.03,
        ...     statistic=2.1, ci_lower=0.002, ci_upper=0.038,
        ...     significant=True, alpha=0.05, method="ztest",
        ...     metric="conversion", control_n=1000,
        ...     treatment_n=1000, power=0.65, effect_size=0.06,
        ... )
        >>> d = r.to_dict()
        >>> isinstance(d, dict)
        True
        """
        result: dict[str, Any] = {}
        for f in fields(self):  # type: ignore[arg-type]
            result[f.name] = _to_python(getattr(self, f.name))
        return result


# ─── ExperimentResult ────────────────────────────────────────────────


@dataclass(frozen=True)
class ExperimentResult(_DictMixin):
    """Result of a frequentist A/B test via :class:`~splita.Experiment`.

    All fields are read-only (frozen dataclass).  Use :meth:`to_dict` to
    serialise for logging or storage.

    Attributes
    ----------
    control_mean : float
        Mean of the control group.
    treatment_mean : float
        Mean of the treatment group.
    lift : float
        Absolute difference (treatment - control).
    relative_lift : float
        Relative difference ``(treatment - control) / |control|``.
    pvalue : float
        p-value of the statistical test.
    statistic : float
        Test statistic (z, t, U, or chi-square depending on method).
    ci_lower : float
        Lower bound of the confidence interval for the lift.
    ci_upper : float
        Upper bound of the confidence interval for the lift.
    significant : bool
        Whether ``pvalue < alpha``.
    alpha : float
        Significance level used.
    method : str
        Statistical test that was applied.
    metric : str
        Metric type (``'conversion'``, ``'continuous'``, or ``'ratio'``).
    control_n : int
        Number of observations in the control group.
    treatment_n : int
        Number of observations in the treatment group.
    power : float
        Post-hoc power estimate.
    effect_size : float
        Standardised effect size (Cohen's h or d).

    Examples
    --------
    >>> import numpy as np
    >>> from splita import Experiment
    >>> ctrl = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0])
    >>> trt  = np.array([1, 1, 0, 1, 1, 0, 1, 0, 1, 1])
    >>> result = Experiment(ctrl, trt).run()
    >>> result.significant
    True
    """

    control_mean: float
    treatment_mean: float
    lift: float
    relative_lift: float
    pvalue: float
    statistic: float
    ci_lower: float
    ci_upper: float
    significant: bool
    alpha: float
    method: str
    metric: str
    control_n: int
    treatment_n: int
    power: float
    effect_size: float

    @property
    def _effect_size_label(self) -> str:
        if self.metric == "conversion":
            return " (Cohen's h)"
        if self.metric == "continuous":
            return " (Cohen's d)"
        return ""

    def __repr__(self) -> str:
        w = 36
        lines = [
            "ExperimentResult",
            _line(w),
            f"  {'metric':<16}{self.metric}",
            f"  {'method':<16}{self.method}",
            f"  {'control_n':<16}{self.control_n}",
            f"  {'treatment_n':<16}{self.treatment_n}",
            _line(w),
            f"  {'control_mean':<16}{_fmt(self.control_mean)}",
            f"  {'treatment_mean':<16}{_fmt(self.treatment_mean)}",
            f"  {'lift':<16}{_fmt(self.lift)}",
            f"  {'relative_lift':<16}{_fmt(self.relative_lift, as_pct=True)}",
            _line(w),
            f"  {'statistic':<16}{_fmt(self.statistic)}",
            f"  {'pvalue':<16}{_fmt(self.pvalue)}",
            f"  {'ci':<16}[{_fmt(self.ci_lower)}, {_fmt(self.ci_upper)}]",
            f"  {'significant':<16}{_fmt(self.significant)}",
            f"  {'alpha':<16}{_fmt(self.alpha)}",
            _line(w),
            f"  {'effect_size':<16}{_fmt(self.effect_size)}{self._effect_size_label}",
            f"  {'power':<16}{_fmt(self.power)} (post-hoc*)",
            "",
            "  * Post-hoc power is a function of the p-value",
            "    and does not provide additional information.",
            "    Use SampleSize for prospective power analysis.",
        ]
        return "\n".join(lines)


# ─── SampleSizeResult ────────────────────────────────────────────────


@dataclass(frozen=True)
class SampleSizeResult(_DictMixin):
    """Result of a sample-size calculation via :class:`~splita.SampleSize`.

    Attributes
    ----------
    n_per_variant : int
        Required sample size per variant.
    n_total : int
        Total sample size across all variants.
    alpha : float
        Significance level.
    power : float
        Statistical power.
    mde : float
        Minimum detectable effect (absolute).
    relative_mde : float or None
        Relative MDE, if provided.
    baseline : float
        Baseline metric value.
    metric : str
        Metric type used in the calculation.
    effect_size : float or None
        Standardised effect size, if applicable.
    days_needed : int or None
        Estimated experiment duration (set by :meth:`duration`).

    Examples
    --------
    >>> from splita import SampleSize
    >>> res = SampleSize.for_proportion(0.10, 0.02)
    >>> res.n_per_variant
    3843
    """

    n_per_variant: int
    n_total: int
    alpha: float
    power: float
    mde: float
    relative_mde: float | None
    baseline: float
    metric: str
    effect_size: float | None
    days_needed: int | None

    def duration(
        self,
        daily_users: int,
        traffic_fraction: float = 1.0,
        ramp_days: int = 0,
    ) -> SampleSizeResult:
        """Estimate experiment duration and return an updated result.

        Returns a new :class:`SampleSizeResult` with ``days_needed``
        populated.

        Parameters
        ----------
        daily_users : int
            Expected number of users per day.
        traffic_fraction : float, default 1.0
            Fraction of daily traffic allocated to the experiment,
            in (0, 1].
        ramp_days : int, default 0
            Additional ramp-up days to add to the estimate.

        Returns
        -------
        SampleSizeResult
            Copy of this result with ``days_needed`` set.

        Raises
        ------
        ValueError
            If *daily_users* is not positive or *traffic_fraction*
            is out of range.

        Examples
        --------
        >>> from splita import SampleSize
        >>> res = SampleSize.for_proportion(0.10, 0.02)
        >>> res.duration(1000).days_needed
        8
        """
        if daily_users <= 0:
            raise ValueError(
                f"`daily_users` must be > 0, got {daily_users}.\n"
                f"  Detail: value must be strictly positive.\n"
                f"  Hint: pass the expected number of users per day."
            )
        if not 0 < traffic_fraction <= 1.0:
            raise ValueError(
                f"`traffic_fraction` must be in (0, 1], got {traffic_fraction}.\n"
                f"  Detail: fraction of daily traffic allocated to the experiment.\n"
                f"  Hint: typical values are 0.1, 0.5, or 1.0."
            )
        days = math.ceil(self.n_total / (daily_users * traffic_fraction)) + ramp_days
        return replace(self, days_needed=days)

    def __repr__(self) -> str:
        w = 36
        lines = [
            "SampleSizeResult",
            _line(w),
            f"  {'metric':<16}{self.metric}",
            f"  {'baseline':<16}{_fmt(self.baseline)}",
            f"  {'mde':<16}{_fmt(self.mde)}",
        ]
        if self.relative_mde is not None:
            rmde = _fmt(self.relative_mde, as_pct=True)
            lines.append(f"  {'relative_mde':<16}{rmde}")
        lines += [
            _line(w),
            f"  {'n_per_variant':<16}{self.n_per_variant}",
            f"  {'n_total':<16}{self.n_total}",
            f"  {'alpha':<16}{_fmt(self.alpha)}",
            f"  {'power':<16}{_fmt(self.power)}",
        ]
        if self.effect_size is not None:
            lines.append(f"  {'effect_size':<16}{_fmt(self.effect_size)}")
        if self.days_needed is not None:
            lines += [
                _line(w),
                f"  {'days_needed':<16}{self.days_needed}",
            ]
        return "\n".join(lines)


# ─── SRMResult ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class SRMResult(_DictMixin):
    """Result of a Sample Ratio Mismatch check via :class:`~splita.SRMCheck`.

    Attributes
    ----------
    observed : list[int]
        Observed user counts per variant.
    expected_counts : list[float]
        Expected counts under the configured split.
    chi2_statistic : float
        Chi-square goodness-of-fit statistic.
    pvalue : float
        p-value of the chi-square test.
    passed : bool
        ``True`` if no mismatch detected (``pvalue >= alpha``).
    alpha : float
        Significance level used.
    deviations_pct : list[float]
        Per-variant percentage deviation from expected.
    worst_variant : int
        Index of the variant with the largest absolute deviation.
    message : str
        Human-readable summary.

    Examples
    --------
    >>> from splita import SRMCheck
    >>> result = SRMCheck([4850, 5150]).run()
    >>> result.passed
    True
    """

    observed: list[int]
    expected_counts: list[float]
    chi2_statistic: float
    pvalue: float
    passed: bool
    alpha: float
    deviations_pct: list[float]
    worst_variant: int
    message: str

    def __repr__(self) -> str:
        w = 28
        dev_strs = [f"{d:+.1f}%" for d in self.deviations_pct]
        lines = [
            "SRMResult",
            _line(w),
            f"  {'observed':<12}{self.observed}",
            f"  {'expected':<12}{[round(e, 1) for e in self.expected_counts]}",
            f"  {'deviation':<12}[{', '.join(dev_strs)}]",
            f"  {'chi2':<12}{_fmt(self.chi2_statistic)}",
            f"  {'pvalue':<12}{_fmt(self.pvalue)}",
            f"  {'passed':<12}{_fmt(self.passed)}",
            f"  {'alpha':<12}{_fmt(self.alpha)}",
            _line(w),
        ]
        if not self.passed:
            dev = self.deviations_pct[self.worst_variant]
            worst_dev = f"{dev:+.1f}%"
            lines += [
                "  \u26a0 Sample ratio mismatch detected!",
                f"  Variant {self.worst_variant} deviates by {worst_dev}.",
                "  Experiment results cannot be trusted.",
            ]
        else:
            lines.append(f"  {self.message}")
        return "\n".join(lines)


# ─── CorrectionResult ───────────────────────────────────────────────


@dataclass(frozen=True)
class CorrectionResult(_DictMixin):
    """Result of multiple-testing correction via :class:`~splita.MultipleCorrection`.

    Attributes
    ----------
    pvalues : list[float]
        Original (unadjusted) p-values.
    adjusted_pvalues : list[float]
        Corrected p-values.
    rejected : list[bool]
        Whether each null hypothesis is rejected after correction.
    alpha : float
        Significance level used.
    method : str
        Human-readable name of the correction method.
    n_rejected : int
        Number of rejected hypotheses.
    n_tests : int
        Total number of tests.
    labels : list[str] or None
        Optional labels for each test.

    Examples
    --------
    >>> from splita import MultipleCorrection
    >>> result = MultipleCorrection([0.01, 0.04, 0.20]).run()
    >>> result.n_rejected
    2
    """

    pvalues: list[float]
    adjusted_pvalues: list[float]
    rejected: list[bool]
    alpha: float
    method: str
    n_rejected: int
    n_tests: int
    labels: list[str] | None

    def __repr__(self) -> str:
        w = 36
        lines = [
            "CorrectionResult",
            _line(w),
            f"  {'method':<18}{self.method}",
            f"  {'alpha':<18}{_fmt(self.alpha)}",
            f"  {'n_tests':<18}{self.n_tests}",
            f"  {'n_rejected':<18}{self.n_rejected}",
            _line(w),
        ]
        for i in range(self.n_tests):
            label = self.labels[i] if self.labels else f"test_{i}"
            rej = "\u2713" if self.rejected[i] else "\u2717"
            lines.append(
                f"  {label:<12} p={_fmt(self.pvalues[i])}"
                f"  adj={_fmt(self.adjusted_pvalues[i])}  {rej}"
            )
        return "\n".join(lines)


# ─── mSPRTState ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class mSPRTState(_DictMixin):
    """Intermediate state of a mixture Sequential Probability Ratio Test.

    Attributes
    ----------
    n_control : int
        Cumulative control observations so far.
    n_treatment : int
        Cumulative treatment observations so far.
    mixture_lr : float
        Mixture likelihood ratio.
    always_valid_pvalue : float
        Always-valid p-value (``1 / mixture_lr``, capped at 1).
    always_valid_ci_lower : float
        Lower bound of the always-valid confidence interval.
    always_valid_ci_upper : float
        Upper bound of the always-valid confidence interval.
    should_stop : bool
        Whether the stopping boundary has been crossed.
    current_effect_estimate : float
        Current point estimate of the treatment effect.
    """

    n_control: int
    n_treatment: int
    mixture_lr: float
    always_valid_pvalue: float
    always_valid_ci_lower: float
    always_valid_ci_upper: float
    should_stop: bool
    current_effect_estimate: float

    def __repr__(self) -> str:
        w = 40
        lines = [
            "mSPRTState",
            _line(w),
            f"  {'n_control':<24}{self.n_control}",
            f"  {'n_treatment':<24}{self.n_treatment}",
            f"  {'mixture_lr':<24}{_fmt(self.mixture_lr)}",
            f"  {'always_valid_pvalue':<24}{_fmt(self.always_valid_pvalue)}",
            (
                f"  {'ci':<24}"
                f"[{_fmt(self.always_valid_ci_lower)}, "
                f"{_fmt(self.always_valid_ci_upper)}]"
            ),
            f"  {'should_stop':<24}{_fmt(self.should_stop)}",
            (f"  {'effect_estimate':<24}{_fmt(self.current_effect_estimate)}"),
        ]
        return "\n".join(lines)


# ─── mSPRTResult ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class mSPRTResult(_DictMixin):
    """Final result of a mixture Sequential Probability Ratio Test.

    Extends :class:`mSPRTState` with stopping metadata.

    Attributes
    ----------
    n_control : int
        Total control observations.
    n_treatment : int
        Total treatment observations.
    mixture_lr : float
        Final mixture likelihood ratio.
    always_valid_pvalue : float
        Always-valid p-value at termination.
    always_valid_ci_lower : float
        Lower bound of the always-valid CI.
    always_valid_ci_upper : float
        Upper bound of the always-valid CI.
    should_stop : bool
        Whether the test recommended stopping.
    current_effect_estimate : float
        Final effect estimate.
    stopping_reason : str
        Human-readable reason for stopping.
    total_observations : int
        ``n_control + n_treatment``.
    relative_speedup_vs_fixed_horizon : float or None
        Fraction of fixed-horizon sample size saved, if available.
    """

    n_control: int
    n_treatment: int
    mixture_lr: float
    always_valid_pvalue: float
    always_valid_ci_lower: float
    always_valid_ci_upper: float
    should_stop: bool
    current_effect_estimate: float
    stopping_reason: str
    total_observations: int
    relative_speedup_vs_fixed_horizon: float | None

    def __repr__(self) -> str:
        w = 40
        lines = [
            "mSPRTResult",
            _line(w),
            f"  {'n_control':<24}{self.n_control}",
            f"  {'n_treatment':<24}{self.n_treatment}",
            f"  {'total_observations':<24}{self.total_observations}",
            f"  {'mixture_lr':<24}{_fmt(self.mixture_lr)}",
            f"  {'always_valid_pvalue':<24}{_fmt(self.always_valid_pvalue)}",
            (
                f"  {'ci':<24}"
                f"[{_fmt(self.always_valid_ci_lower)}, "
                f"{_fmt(self.always_valid_ci_upper)}]"
            ),
            f"  {'should_stop':<24}{_fmt(self.should_stop)}",
            (f"  {'effect_estimate':<24}{_fmt(self.current_effect_estimate)}"),
            _line(w),
            f"  {'stopping_reason':<24}{self.stopping_reason}",
        ]
        if self.relative_speedup_vs_fixed_horizon is not None:
            speedup = _fmt(
                self.relative_speedup_vs_fixed_horizon,
                as_pct=True,
            )
            lines.append(f"  {'speedup_vs_fixed':<24}{speedup}")
        return "\n".join(lines)


# ─── BoundaryResult ──────────────────────────────────────────────────


@dataclass(frozen=True)
class BoundaryResult(_DictMixin):
    """Spending-function boundaries for a group-sequential design.

    Attributes
    ----------
    efficacy_boundaries : list[float]
        Critical z-values for efficacy at each interim look.
    futility_boundaries : list[float] or None
        Critical z-values for futility, if requested.
    information_fractions : list[float]
        Planned information fractions at each look.
    alpha_spent : list[float]
        Cumulative alpha spent at each look.
    adjusted_alpha : float
        Adjusted significance level for the final look.
    """

    efficacy_boundaries: list[float]
    futility_boundaries: list[float] | None
    information_fractions: list[float]
    alpha_spent: list[float]
    adjusted_alpha: float

    def __repr__(self) -> str:
        w = 40
        n_looks = len(self.information_fractions)
        lines = [
            "BoundaryResult",
            _line(w),
            f"  {'adjusted_alpha':<20}{_fmt(self.adjusted_alpha)}",
            f"  {'n_looks':<20}{n_looks}",
            _line(w),
        ]
        header = f"  {'look':<6}{'info_frac':<12}{'efficacy':<12}"
        if self.futility_boundaries is not None:
            header += f"{'futility':<12}"
        header += f"{'alpha_spent':<12}"
        lines.append(header)
        for i in range(n_looks):
            info = _fmt(self.information_fractions[i])
            eff = _fmt(self.efficacy_boundaries[i])
            row = f"  {i + 1:<6}{info:<12}{eff:<12}"
            if self.futility_boundaries is not None:
                row += f"{_fmt(self.futility_boundaries[i]):<12}"
            row += f"{_fmt(self.alpha_spent[i]):<12}"
            lines.append(row)
        return "\n".join(lines)


# ─── GSResult ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GSResult(_DictMixin):
    """Result of a group-sequential analysis.

    Attributes
    ----------
    analysis_results : list[dict[str, Any]]
        Per-look analysis details.
    crossed_efficacy : bool
        Whether the efficacy boundary was crossed.
    crossed_futility : bool
        Whether the futility boundary was crossed.
    recommended_action : str
        Human-readable recommendation (e.g. ``'stop_for_efficacy'``).
    """

    analysis_results: list[dict[str, Any]]
    crossed_efficacy: bool
    crossed_futility: bool
    recommended_action: str

    def __repr__(self) -> str:
        w = 36
        lines = [
            "GSResult",
            _line(w),
            f"  {'crossed_efficacy':<20}{_fmt(self.crossed_efficacy)}",
            f"  {'crossed_futility':<20}{_fmt(self.crossed_futility)}",
            f"  {'recommended':<20}{self.recommended_action}",
            _line(w),
            f"  {len(self.analysis_results)} analysis result(s)",
        ]
        return "\n".join(lines)


# ─── BanditResult ────────────────────────────────────────────────────


@dataclass(frozen=True)
class BanditResult(_DictMixin):
    """Result of a multi-armed bandit simulation.

    Attributes
    ----------
    n_pulls_per_arm : list[int]
        Number of pulls allocated to each arm.
    prob_best : list[float]
        Posterior probability of each arm being the best.
    expected_loss : list[float]
        Expected loss for choosing each arm.
    current_best_arm : int
        Index of the arm with the highest probability of being best.
    should_stop : bool
        Whether the stopping criterion has been met.
    total_reward : float
        Cumulative reward across all pulls.
    cumulative_regret : float or None
        Cumulative regret, if a true best arm is known.
    arm_means : list[float]
        Posterior mean estimate for each arm.
    arm_credible_intervals : list[tuple[float, float]]
        Posterior credible intervals for each arm.
    """

    n_pulls_per_arm: list[int]
    prob_best: list[float]
    expected_loss: list[float]
    current_best_arm: int
    should_stop: bool
    total_reward: float
    cumulative_regret: float | None
    arm_means: list[float]
    arm_credible_intervals: list[tuple[float, float]]

    def __repr__(self) -> str:
        w = 40
        n_arms = len(self.n_pulls_per_arm)
        lines = [
            "BanditResult",
            _line(w),
            f"  {'n_arms':<20}{n_arms}",
            f"  {'current_best_arm':<20}{self.current_best_arm}",
            f"  {'should_stop':<20}{_fmt(self.should_stop)}",
            f"  {'total_reward':<20}{_fmt(self.total_reward)}",
        ]
        if self.cumulative_regret is not None:
            lines.append(f"  {'cumulative_regret':<20}{_fmt(self.cumulative_regret)}")
        lines += [
            _line(w),
            f"  {'arm':<6}{'pulls':<10}{'mean':<10}{'P(best)':<10}{'loss':<10}",
        ]
        for i in range(n_arms):
            lines.append(
                f"  {i:<6}{self.n_pulls_per_arm[i]:<10}"
                f"{_fmt(self.arm_means[i]):<10}"
                f"{_fmt(self.prob_best[i]):<10}"
                f"{_fmt(self.expected_loss[i]):<10}"
            )
        return "\n".join(lines)
