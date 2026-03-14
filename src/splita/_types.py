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


# ─── EValueState ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class EValueState(_DictMixin):
    """Intermediate state of an E-value sequential test.

    Attributes
    ----------
    e_value : float
        Current e-value (mixture likelihood ratio).
    n_control : int
        Cumulative control observations so far.
    n_treatment : int
        Cumulative treatment observations so far.
    should_stop : bool
        Whether the rejection threshold has been reached.
    """

    e_value: float
    n_control: int
    n_treatment: int
    should_stop: bool

    def __repr__(self) -> str:
        w = 36
        lines = [
            "EValueState",
            _line(w),
            f"  {'e_value':<20}{_fmt(self.e_value)}",
            f"  {'n_control':<20}{self.n_control}",
            f"  {'n_treatment':<20}{self.n_treatment}",
            f"  {'should_stop':<20}{_fmt(self.should_stop)}",
        ]
        return "\n".join(lines)


# ─── EValueResult ────────────────────────────────────────────────────


@dataclass(frozen=True)
class EValueResult(_DictMixin):
    """Final result of an E-value sequential test.

    Extends :class:`EValueState` with stopping metadata.

    Attributes
    ----------
    e_value : float
        Final e-value at termination.
    n_control : int
        Total control observations.
    n_treatment : int
        Total treatment observations.
    should_stop : bool
        Whether the test recommended stopping.
    stopping_reason : str
        Human-readable reason for stopping.
    """

    e_value: float
    n_control: int
    n_treatment: int
    should_stop: bool
    stopping_reason: str

    def __repr__(self) -> str:
        w = 36
        lines = [
            "EValueResult",
            _line(w),
            f"  {'e_value':<20}{_fmt(self.e_value)}",
            f"  {'n_control':<20}{self.n_control}",
            f"  {'n_treatment':<20}{self.n_treatment}",
            f"  {'should_stop':<20}{_fmt(self.should_stop)}",
            f"  {'stopping_reason':<20}{self.stopping_reason}",
        ]
        return "\n".join(lines)


# ─── BayesianStoppingResult ─────────────────────────────────────────


@dataclass(frozen=True)
class BayesianStoppingResult(_DictMixin):
    """Result of a Bayesian stopping rule evaluation.

    Attributes
    ----------
    should_stop : bool
        Whether the stopping criterion has been met.
    rule : str
        The stopping rule used.
    threshold : float
        The threshold for the stopping rule.
    current_value : float
        The current value of the stopping metric.
    message : str
        Human-readable summary of the evaluation.
    """

    should_stop: bool
    rule: str
    threshold: float
    current_value: float
    message: str

    def __repr__(self) -> str:
        w = 36
        lines = [
            "BayesianStoppingResult",
            _line(w),
            f"  {'should_stop':<20}{_fmt(self.should_stop)}",
            f"  {'rule':<20}{self.rule}",
            f"  {'threshold':<20}{_fmt(self.threshold)}",
            f"  {'current_value':<20}{_fmt(self.current_value)}",
            _line(w),
            f"  {self.message}",
        ]
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


# ─── BayesianResult ──────────────────────────────────────────────


@dataclass(frozen=True)
class BayesianResult(_DictMixin):
    """Result of a Bayesian A/B test via :class:`~splita.BayesianExperiment`.

    All fields are read-only (frozen dataclass).  Use :meth:`to_dict` to
    serialise for logging or storage.

    Attributes
    ----------
    prob_b_beats_a : float
        Posterior probability that treatment > control.
    expected_loss_a : float
        Expected loss if choosing control (A).
    expected_loss_b : float
        Expected loss if choosing treatment (B).
    lift : float
        Posterior mean of (treatment - control).
    relative_lift : float
        Posterior mean of relative lift.
    ci_lower : float
        Lower bound of the 95% credible interval for the lift.
    ci_upper : float
        Upper bound of the 95% credible interval for the lift.
    credible_level : float
        Credible interval level (default 0.95).
    control_mean : float
        Posterior mean of the control group.
    treatment_mean : float
        Posterior mean of the treatment group.
    prob_in_rope : float or None
        P(effect in ROPE) if ROPE was set, else None.
    rope : tuple of (float, float) or None
        ROPE bounds, if set.
    metric : str
        Metric type (``'conversion'`` or ``'continuous'``).
    control_n : int
        Number of observations in the control group.
    treatment_n : int
        Number of observations in the treatment group.
    """

    prob_b_beats_a: float
    expected_loss_a: float
    expected_loss_b: float
    lift: float
    relative_lift: float
    ci_lower: float
    ci_upper: float
    credible_level: float
    control_mean: float
    treatment_mean: float
    prob_in_rope: float | None
    rope: tuple[float, float] | None
    metric: str
    control_n: int
    treatment_n: int

    def __repr__(self) -> str:
        w = 36
        lines = [
            "BayesianResult",
            _line(w),
            f"  {'metric':<16}{self.metric}",
            f"  {'control_n':<16}{self.control_n}",
            f"  {'treatment_n':<16}{self.treatment_n}",
            _line(w),
            f"  {'P(B > A)':<16}{_fmt(self.prob_b_beats_a)}",
            f"  {'expected_loss_a':<16}{_fmt(self.expected_loss_a)}",
            f"  {'expected_loss_b':<16}{_fmt(self.expected_loss_b)}",
            _line(w),
            f"  {'lift':<16}{_fmt(self.lift)}",
            f"  {'relative_lift':<16}{_fmt(self.relative_lift, as_pct=True)}",
            (
                f"  {int(self.credible_level * 100)}% credible"
                f"      [{_fmt(self.ci_lower)}, {_fmt(self.ci_upper)}]"
            ),
        ]
        if self.prob_in_rope is not None:
            lines.append(f"  {'P(in ROPE)':<16}{_fmt(self.prob_in_rope)}")
            lines.append(f"  {'ROPE':<16}[{_fmt(self.rope[0])}, {_fmt(self.rope[1])}]")
        lines.append(_line(w))

        # Decision line
        if self.prob_b_beats_a > 0.5:
            decision = "Choose Treatment (B)"
            loss = self.expected_loss_b
        else:
            decision = "Choose Control (A)"
            loss = self.expected_loss_a
        lines.append(f"  Decision: {decision}")
        lines.append(f"  Expected loss: {_fmt(loss, as_pct=True)}")
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


# ─── NoveltyCurveResult ─────────────────────────────────────────────


@dataclass(frozen=True)
class NoveltyCurveResult(_DictMixin):
    """Result of a novelty/primacy effect analysis.

    Attributes
    ----------
    windows : list[dict[str, Any]]
        Per-window results.  Each dict contains ``window_start``,
        ``lift``, ``pvalue``, ``ci_lower``, and ``ci_upper``.
    has_novelty_effect : bool
        ``True`` if the treatment effect decreases significantly over time
        (last window lift < first window lift * 0.5).
    trend_direction : str
        One of ``'stable'``, ``'decreasing'``, or ``'increasing'``.

    Examples
    --------
    >>> from splita._types import NoveltyCurveResult
    >>> r = NoveltyCurveResult(
    ...     windows=[{"window_start": 0, "lift": 0.05, "pvalue": 0.01,
    ...               "ci_lower": 0.01, "ci_upper": 0.09}],
    ...     has_novelty_effect=False,
    ...     trend_direction="stable",
    ... )
    >>> r.has_novelty_effect
    False
    """

    windows: list[dict[str, Any]]
    has_novelty_effect: bool
    trend_direction: str

    def __repr__(self) -> str:
        w = 36
        lines = [
            "NoveltyCurveResult",
            _line(w),
            f"  {'n_windows':<20}{len(self.windows)}",
            f"  {'has_novelty_effect':<20}{_fmt(self.has_novelty_effect)}",
            f"  {'trend_direction':<20}{self.trend_direction}",
            _line(w),
        ]
        for win in self.windows:
            lines.append(
                f"  start={win['window_start']}  "
                f"lift={_fmt(win['lift'])}  "
                f"p={_fmt(win['pvalue'])}"
            )
        return "\n".join(lines)


# ─── AATestResult ───────────────────────────────────────────────────


@dataclass(frozen=True)
class AATestResult(_DictMixin):
    """Result of an A/A test validation.

    Attributes
    ----------
    false_positive_rate : float
        Observed fraction of simulations that were significant.
    expected_rate : float
        Expected false positive rate (equal to alpha).
    passed : bool
        ``True`` if the false positive rate is within expected bounds.
    p_values : list[float]
        p-values from each simulation.
    n_simulations : int
        Number of random splits performed.
    message : str
        Human-readable summary.

    Examples
    --------
    >>> from splita._types import AATestResult
    >>> r = AATestResult(
    ...     false_positive_rate=0.048,
    ...     expected_rate=0.05,
    ...     passed=True,
    ...     p_values=[0.1, 0.9],
    ...     n_simulations=2,
    ...     message="A/A test passed.",
    ... )
    >>> r.passed
    True
    """

    false_positive_rate: float
    expected_rate: float
    passed: bool
    p_values: list[float]
    n_simulations: int
    message: str

    def __repr__(self) -> str:
        w = 36
        lines = [
            "AATestResult",
            _line(w),
            f"  {'false_positive_rate':<22}{_fmt(self.false_positive_rate)}",
            f"  {'expected_rate':<22}{_fmt(self.expected_rate)}",
            f"  {'passed':<22}{_fmt(self.passed)}",
            f"  {'n_simulations':<22}{self.n_simulations}",
            _line(w),
            f"  {self.message}",
        ]
        return "\n".join(lines)


# ─── EffectTimeSeriesResult ─────────────────────────────────────────


@dataclass(frozen=True)
class EffectTimeSeriesResult(_DictMixin):
    """Result of a cumulative treatment-effect time series analysis.

    Attributes
    ----------
    time_points : list[dict[str, Any]]
        Per-timestamp cumulative results.  Each dict contains
        ``timestamp``, ``cumulative_lift``, ``pvalue``, ``ci_lower``,
        ``ci_upper``, ``n_control``, and ``n_treatment``.
    is_stable : bool
        ``True`` if the treatment effect is stable over time.
    final_lift : float
        Cumulative lift at the last time point.
    final_pvalue : float
        p-value at the last time point.

    Examples
    --------
    >>> from splita._types import EffectTimeSeriesResult
    >>> r = EffectTimeSeriesResult(
    ...     time_points=[{"timestamp": 1, "cumulative_lift": 0.05,
    ...                   "pvalue": 0.03, "ci_lower": 0.01,
    ...                   "ci_upper": 0.09, "n_control": 100,
    ...                   "n_treatment": 100}],
    ...     is_stable=True,
    ...     final_lift=0.05,
    ...     final_pvalue=0.03,
    ... )
    >>> r.is_stable
    True
    """

    time_points: list[dict[str, Any]]
    is_stable: bool
    final_lift: float
    final_pvalue: float

    def __repr__(self) -> str:
        w = 36
        lines = [
            "EffectTimeSeriesResult",
            _line(w),
            f"  {'n_time_points':<20}{len(self.time_points)}",
            f"  {'is_stable':<20}{_fmt(self.is_stable)}",
            f"  {'final_lift':<20}{_fmt(self.final_lift)}",
            f"  {'final_pvalue':<20}{_fmt(self.final_pvalue)}",
            _line(w),
        ]
        for tp in self.time_points:
            lines.append(
                f"  t={tp['timestamp']}  "
                f"lift={_fmt(tp['cumulative_lift'])}  "
                f"p={_fmt(tp['pvalue'])}  "
                f"n={tp['n_control']}+{tp['n_treatment']}"
            )
        return "\n".join(lines)


# ─── PowerSimulationResult ────────────────────────────────────────


@dataclass(frozen=True)
class PowerSimulationResult(_DictMixin):
    """Result of a Monte Carlo power simulation via :class:`~splita.PowerSimulation`.

    Attributes
    ----------
    power : float
        Fraction of simulations that rejected H0.
    rejection_rate : float
        Alias for *power*.
    n_simulations : int
        Number of Monte Carlo replications.
    n_per_variant : int
        Sample size per variant.
    alpha : float
        Significance level used in each simulation.
    mean_effect : float
        Average observed effect (lift) across simulations.
    mean_pvalue : float
        Average p-value across simulations.
    ci_power_lower : float
        Lower bound of the 95% Wilson score CI on the power estimate.
    ci_power_upper : float
        Upper bound of the 95% Wilson score CI on the power estimate.

    Examples
    --------
    >>> from splita import PowerSimulation
    >>> result = PowerSimulation.for_mean(
    ...     10.0, 2.0, 1.0, 500, n_simulations=200, random_state=0
    ... )
    >>> 0.0 <= result.power <= 1.0
    True
    """

    power: float
    rejection_rate: float
    n_simulations: int
    n_per_variant: int
    alpha: float
    mean_effect: float
    mean_pvalue: float
    ci_power_lower: float
    ci_power_upper: float

    def __repr__(self) -> str:
        w = 36
        lines = [
            "PowerSimulationResult",
            _line(w),
            f"  {'power':<20}{_fmt(self.power)}",
            f"  {'n_simulations':<20}{self.n_simulations}",
            f"  {'n_per_variant':<20}{self.n_per_variant}",
            f"  {'alpha':<20}{_fmt(self.alpha)}",
            _line(w),
            f"  {'mean_effect':<20}{_fmt(self.mean_effect)}",
            f"  {'mean_pvalue':<20}{_fmt(self.mean_pvalue)}",
            f"  {'95% CI on power':<20}"
            f"[{_fmt(self.ci_power_lower)}, {_fmt(self.ci_power_upper)}]",
        ]
        return "\n".join(lines)


# ─── QuantileResult ───────────────────────────────────────────────


@dataclass(frozen=True)
class QuantileResult(_DictMixin):
    """Result of a quantile-based A/B test via :class:`~splita.QuantileExperiment`.

    All fields are read-only (frozen dataclass).  Use :meth:`to_dict` to
    serialise for logging or storage.

    Attributes
    ----------
    quantiles : list[float]
        Which quantiles were tested.
    control_quantiles : list[float]
        Quantile values for the control group.
    treatment_quantiles : list[float]
        Quantile values for the treatment group.
    differences : list[float]
        Treatment minus control at each quantile.
    ci_lower : list[float]
        Lower bound of the confidence interval for each difference.
    ci_upper : list[float]
        Upper bound of the confidence interval for each difference.
    pvalues : list[float]
        Bootstrap p-value for each quantile.
    significant : list[bool]
        Whether ``pvalue < alpha`` for each quantile.
    alpha : float
        Significance level used.
    control_n : int
        Number of observations in the control group.
    treatment_n : int
        Number of observations in the treatment group.

    Examples
    --------
    >>> from splita import QuantileExperiment
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> ctrl = rng.normal(10, 2, size=500)
    >>> trt = ctrl + 1
    >>> result = QuantileExperiment(ctrl, trt, quantiles=0.5, random_state=0).run()
    >>> result.significant[0]
    True
    """

    quantiles: list[float]
    control_quantiles: list[float]
    treatment_quantiles: list[float]
    differences: list[float]
    ci_lower: list[float]
    ci_upper: list[float]
    pvalues: list[float]
    significant: list[bool]
    alpha: float
    control_n: int
    treatment_n: int

    def __repr__(self) -> str:
        w = 56
        lines = [
            "QuantileResult",
            _line(w),
            f"  {'control_n':<16}{self.control_n}",
            f"  {'treatment_n':<16}{self.treatment_n}",
            f"  {'alpha':<16}{_fmt(self.alpha)}",
            _line(w),
            f"  {'quantile':<10}{'diff':<12}{'ci':<24}{'pvalue':<10}{'sig':<6}",
        ]
        for i, q in enumerate(self.quantiles):
            ci_str = f"[{_fmt(self.ci_lower[i])}, {_fmt(self.ci_upper[i])}]"
            sig_str = "\u2713" if self.significant[i] else "\u2717"
            lines.append(
                f"  {_fmt(q):<10}{_fmt(self.differences[i]):<12}"
                f"{ci_str:<24}{_fmt(self.pvalues[i]):<10}{sig_str}"
            )
        return "\n".join(lines)


# ─── HTEResult ───────────────────────────────────────────────────


@dataclass(frozen=True)
class HTEResult(_DictMixin):
    """Result of heterogeneous treatment effect estimation.

    Attributes
    ----------
    cate_estimates : list[float]
        Conditional average treatment effect estimates for each observation.
    mean_cate : float
        Mean of the CATE estimates.
    cate_std : float
        Standard deviation of the CATE estimates.
    top_features : list[int] or None
        Indices of the most important features, if the estimator
        exposes ``feature_importances_``.
    method : str
        Estimation method used (``'t_learner'`` or ``'s_learner'``).
    """

    cate_estimates: list[float]
    mean_cate: float
    cate_std: float
    top_features: list[int] | None
    method: str

    def __repr__(self) -> str:
        w = 36
        lines = [
            "HTEResult",
            _line(w),
            f"  {'method':<16}{self.method}",
            f"  {'n_estimates':<16}{len(self.cate_estimates)}",
            f"  {'mean_cate':<16}{_fmt(self.mean_cate)}",
            f"  {'cate_std':<16}{_fmt(self.cate_std)}",
            _line(w),
        ]
        if self.top_features is not None:
            lines.append(f"  {'top_features':<16}{self.top_features}")
        return "\n".join(lines)


# ─── MetricSensitivityResult ─────────────────────────────────────


@dataclass(frozen=True)
class MetricSensitivityResult(_DictMixin):
    """Result of a metric sensitivity analysis.

    Attributes
    ----------
    estimated_std : float
        Estimated standard deviation of the metric.
    estimated_power : float
        Estimated statistical power at the given MDE.
    recommended_n : int
        Recommended sample size per variant to achieve 80% power.
    is_sensitive : bool
        Whether the metric can detect the given MDE with 80% power
        at a reasonable sample size.
    mde : float
        Minimum detectable effect used.
    n_simulations : int
        Number of simulations performed.
    alpha : float
        Significance level used.
    """

    estimated_std: float
    estimated_power: float
    recommended_n: int
    is_sensitive: bool
    mde: float
    n_simulations: int
    alpha: float

    def __repr__(self) -> str:
        w = 36
        lines = [
            "MetricSensitivityResult",
            _line(w),
            f"  {'estimated_std':<20}{_fmt(self.estimated_std)}",
            f"  {'estimated_power':<20}{_fmt(self.estimated_power)}",
            f"  {'recommended_n':<20}{self.recommended_n}",
            f"  {'is_sensitive':<20}{_fmt(self.is_sensitive)}",
            _line(w),
            f"  {'mde':<20}{_fmt(self.mde)}",
            f"  {'n_simulations':<20}{self.n_simulations}",
            f"  {'alpha':<20}{_fmt(self.alpha)}",
        ]
        return "\n".join(lines)


# ─── VarianceEstimateResult ──────────────────────────────────────


@dataclass(frozen=True)
class VarianceEstimateResult(_DictMixin):
    """Result of a variance estimation analysis.

    Attributes
    ----------
    mean : float
        Sample mean.
    std : float
        Sample standard deviation.
    skewness : float
        Sample skewness.
    kurtosis : float
        Sample excess kurtosis.
    percentiles : dict[str, float]
        Key percentiles (5, 25, 50, 75, 95).
    is_heavy_tailed : bool
        ``True`` if kurtosis > 5.
    is_skewed : bool
        ``True`` if ``|skewness| > 2``.
    recommendations : list[str]
        Suggested actions based on the distributional properties.
    """

    mean: float
    std: float
    skewness: float
    kurtosis: float
    percentiles: dict[str, float]
    is_heavy_tailed: bool
    is_skewed: bool
    recommendations: list[str]

    def __repr__(self) -> str:
        w = 36
        lines = [
            "VarianceEstimateResult",
            _line(w),
            f"  {'mean':<20}{_fmt(self.mean)}",
            f"  {'std':<20}{_fmt(self.std)}",
            f"  {'skewness':<20}{_fmt(self.skewness)}",
            f"  {'kurtosis':<20}{_fmt(self.kurtosis)}",
            _line(w),
            f"  {'is_heavy_tailed':<20}{_fmt(self.is_heavy_tailed)}",
            f"  {'is_skewed':<20}{_fmt(self.is_skewed)}",
            _line(w),
        ]
        for r in self.recommendations:
            lines.append(f"  {r}")
        return "\n".join(lines)


# ─── TriggeredResult ─────────────────────────────────────────────


@dataclass(frozen=True)
class TriggeredResult(_DictMixin):
    """Result of a triggered experiment analysis.

    Attributes
    ----------
    itt_result : ExperimentResult
        Intent-to-treat result (all assigned users).
    per_protocol_result : ExperimentResult
        Per-protocol result (only triggered users).
    trigger_rate_control : float
        Fraction of control users who were triggered.
    trigger_rate_treatment : float
        Fraction of treatment users who were triggered.
    """

    itt_result: ExperimentResult
    per_protocol_result: ExperimentResult
    trigger_rate_control: float
    trigger_rate_treatment: float

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a plain Python dictionary."""
        return {
            "itt_result": self.itt_result.to_dict(),
            "per_protocol_result": self.per_protocol_result.to_dict(),
            "trigger_rate_control": _to_python(self.trigger_rate_control),
            "trigger_rate_treatment": _to_python(self.trigger_rate_treatment),
        }

    def __repr__(self) -> str:
        w = 36
        lines = [
            "TriggeredResult",
            _line(w),
            f"  {'trigger_rate_ctrl':<20}{_fmt(self.trigger_rate_control)}",
            f"  {'trigger_rate_trt':<20}{_fmt(self.trigger_rate_treatment)}",
            _line(w),
            f"  {'ITT lift':<20}{_fmt(self.itt_result.lift)}",
            f"  {'ITT pvalue':<20}{_fmt(self.itt_result.pvalue)}",
            f"  {'PP lift':<20}{_fmt(self.per_protocol_result.lift)}",
            f"  {'PP pvalue':<20}{_fmt(self.per_protocol_result.pvalue)}",
        ]
        return "\n".join(lines)


# ─── InteractionResult ───────────────────────────────────────────


@dataclass(frozen=True)
class InteractionResult(_DictMixin):
    """Result of an interaction (heterogeneity) test across segments.

    Attributes
    ----------
    segment_results : list[dict[str, Any]]
        Per-segment experiment results.
    has_interaction : bool
        Whether the treatment effect significantly differs across segments.
    interaction_pvalue : float
        p-value of the heterogeneity (chi-square) test.
    strongest_segment : str
        Label of the segment with the largest absolute effect.
    """

    segment_results: list[dict[str, Any]]
    has_interaction: bool
    interaction_pvalue: float
    strongest_segment: str

    def __repr__(self) -> str:
        w = 36
        lines = [
            "InteractionResult",
            _line(w),
            f"  {'n_segments':<20}{len(self.segment_results)}",
            f"  {'has_interaction':<20}{_fmt(self.has_interaction)}",
            f"  {'interaction_pvalue':<20}{_fmt(self.interaction_pvalue)}",
            f"  {'strongest_segment':<20}{self.strongest_segment}",
            _line(w),
        ]
        for seg in self.segment_results:
            lines.append(
                f"  {seg['segment']!s:<12} "
                f"lift={_fmt(seg['lift'])}  "
                f"p={_fmt(seg['pvalue'])}"
            )
        return "\n".join(lines)


# ─── SurrogateResult ─────────────────────────────────────────────


@dataclass(frozen=True)
class SurrogateResult(_DictMixin):
    """Result of long-run effect estimation via surrogacy.

    Attributes
    ----------
    predicted_long_term_lift : float
        Predicted long-term treatment effect inferred from short-term data.
    prediction_ci : tuple[float, float]
        Confidence interval for the predicted long-term lift.
    surrogate_r2 : float
        R-squared of the surrogate model on the training data.
    is_valid_surrogate : bool
        Whether the surrogate is considered valid (r2 > 0.3).
    """

    predicted_long_term_lift: float
    prediction_ci: tuple[float, float]
    surrogate_r2: float
    is_valid_surrogate: bool

    def __repr__(self) -> str:
        w = 36
        lines = [
            "SurrogateResult",
            _line(w),
            f"  {'long_term_lift':<20}{_fmt(self.predicted_long_term_lift)}",
            f"  {'prediction_ci':<20}"
            f"({_fmt(self.prediction_ci[0])}, {_fmt(self.prediction_ci[1])})",
            f"  {'surrogate_r2':<20}{_fmt(self.surrogate_r2)}",
            f"  {'is_valid_surrogate':<20}{_fmt(self.is_valid_surrogate)}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── ClusterResult ───────────────────────────────────────────────


@dataclass(frozen=True)
class ClusterResult(_DictMixin):
    """Result of a cluster-randomized experiment.

    Attributes
    ----------
    lift : float
        Difference in means (treatment - control).
    pvalue : float
        p-value from the cluster-robust t-test.
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    significant : bool
        Whether the result is significant at the given alpha.
    n_clusters_control : int
        Number of clusters in the control group.
    n_clusters_treatment : int
        Number of clusters in the treatment group.
    icc : float
        Intraclass correlation coefficient.
    """

    lift: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    significant: bool
    n_clusters_control: int
    n_clusters_treatment: int
    icc: float

    def __repr__(self) -> str:
        w = 36
        lines = [
            "ClusterResult",
            _line(w),
            f"  {'lift':<20}{_fmt(self.lift)}",
            f"  {'pvalue':<20}{_fmt(self.pvalue)}",
            f"  {'ci':<20}({_fmt(self.ci_lower)}, {_fmt(self.ci_upper)})",
            f"  {'significant':<20}{_fmt(self.significant)}",
            f"  {'clusters_ctrl':<20}{self.n_clusters_control}",
            f"  {'clusters_trt':<20}{self.n_clusters_treatment}",
            f"  {'icc':<20}{_fmt(self.icc)}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── SwitchbackResult ────────────────────────────────────────────


@dataclass(frozen=True)
class SwitchbackResult(_DictMixin):
    """Result of a switchback experiment.

    Attributes
    ----------
    lift : float
        Difference in period-level means (treatment - control).
    pvalue : float
        p-value from the period-level t-test.
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    significant : bool
        Whether the result is significant at the given alpha.
    n_periods : int
        Total number of time periods.
    n_treatment_periods : int
        Number of treatment periods.
    n_control_periods : int
        Number of control periods.
    """

    lift: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    significant: bool
    n_periods: int
    n_treatment_periods: int
    n_control_periods: int

    def __repr__(self) -> str:
        w = 36
        lines = [
            "SwitchbackResult",
            _line(w),
            f"  {'lift':<20}{_fmt(self.lift)}",
            f"  {'pvalue':<20}{_fmt(self.pvalue)}",
            f"  {'ci':<20}({_fmt(self.ci_lower)}, {_fmt(self.ci_upper)})",
            f"  {'significant':<20}{_fmt(self.significant)}",
            f"  {'n_periods':<20}{self.n_periods}",
            f"  {'n_treatment':<20}{self.n_treatment_periods}",
            f"  {'n_control':<20}{self.n_control_periods}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── DiDResult ───────────────────────────────────────────────────


@dataclass(frozen=True)
class DiDResult(_DictMixin):
    """Result of a Difference-in-Differences analysis.

    Attributes
    ----------
    att : float
        Average treatment effect on the treated.
    se : float
        Standard error of the ATT estimate.
    pvalue : float
        p-value for the ATT.
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    significant : bool
        Whether the result is significant at the given alpha.
    pre_trend_diff : float
        Difference in pre-treatment trends (treatment - control).
    parallel_trends_pvalue : float
        p-value for the parallel trends assumption test.
    """

    att: float
    se: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    significant: bool
    pre_trend_diff: float
    parallel_trends_pvalue: float

    def __repr__(self) -> str:
        w = 36
        lines = [
            "DiDResult",
            _line(w),
            f"  {'att':<20}{_fmt(self.att)}",
            f"  {'se':<20}{_fmt(self.se)}",
            f"  {'pvalue':<20}{_fmt(self.pvalue)}",
            f"  {'ci':<20}({_fmt(self.ci_lower)}, {_fmt(self.ci_upper)})",
            f"  {'significant':<20}{_fmt(self.significant)}",
            f"  {'pre_trend_diff':<20}{_fmt(self.pre_trend_diff)}",
            f"  {'parallel_trends_p':<20}{_fmt(self.parallel_trends_pvalue)}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── SyntheticControlResult ──────────────────────────────────────


@dataclass(frozen=True)
class SyntheticControlResult(_DictMixin):
    """Result of a Synthetic Control analysis.

    Attributes
    ----------
    effect : float
        Average post-treatment effect (treated - synthetic).
    weights : tuple[float, ...]
        Weights assigned to each donor unit.
    pre_treatment_rmse : float
        Root mean squared error of the synthetic control in the pre-treatment period.
    donor_contributions : dict[int, float]
        Mapping from donor index to its weight.
    effect_series : list[float]
        Per-period post-treatment effects (treated - synthetic).
    """

    effect: float
    weights: tuple[float, ...]
    pre_treatment_rmse: float
    donor_contributions: dict[int, float]
    effect_series: list[float]

    def __repr__(self) -> str:
        w = 36
        lines = [
            "SyntheticControlResult",
            _line(w),
            f"  {'effect':<20}{_fmt(self.effect)}",
            f"  {'pre_rmse':<20}{_fmt(self.pre_treatment_rmse)}",
            f"  {'n_donors':<20}{len(self.weights)}",
            f"  {'effect_periods':<20}{len(self.effect_series)}",
            _line(w),
        ]
        return "\n".join(lines)
