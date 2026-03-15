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


# ─── RegressionAdjustmentResult ──────────────────────────────────


@dataclass(frozen=True)
class RegressionAdjustmentResult(_DictMixin):
    """Result of Lin's regression adjustment (Lin, 2013).

    Attributes
    ----------
    ate : float
        Average treatment effect (coefficient on the treatment indicator).
    se : float
        HC2 robust standard error of the ATE.
    pvalue : float
        Two-sided p-value from the t-distribution.
    ci_lower : float
        Lower bound of the confidence interval for the ATE.
    ci_upper : float
        Upper bound of the confidence interval for the ATE.
    significant : bool
        Whether ``pvalue < alpha``.
    alpha : float
        Significance level used.
    variance_reduction : float
        Fraction of variance reduced compared to the unadjusted
        difference-in-means estimator: ``1 - se_adj^2 / se_unadj^2``.
    r_squared : float
        R-squared of the regression model.
    """

    ate: float
    se: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    significant: bool
    alpha: float
    variance_reduction: float
    r_squared: float

    def __repr__(self) -> str:
        w = 36
        lines = [
            "RegressionAdjustmentResult",
            _line(w),
            f"  {'ate':<20}{_fmt(self.ate)}",
            f"  {'se':<20}{_fmt(self.se)}",
            f"  {'pvalue':<20}{_fmt(self.pvalue)}",
            _line(w),
            f"  {'ci_lower':<20}{_fmt(self.ci_lower)}",
            f"  {'ci_upper':<20}{_fmt(self.ci_upper)}",
            f"  {'significant':<20}{_fmt(self.significant)}",
            f"  {'alpha':<20}{_fmt(self.alpha)}",
            _line(w),
            f"  {'variance_reduction':<20}{_fmt(self.variance_reduction, as_pct=True)}",
            f"  {'r_squared':<20}{_fmt(self.r_squared)}",
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


# ─── StratifiedResult ────────────────────────────────────────────────


@dataclass(frozen=True)
class StratifiedResult(_DictMixin):
    """Result of a stratified A/B test via :class:`~splita.StratifiedExperiment`.

    Attributes
    ----------
    ate : float
        Average treatment effect (Neyman-weighted across strata).
    se : float
        Standard error of the ATE estimate.
    pvalue : float
        Two-sided p-value from a z-test on the stratified ATE.
    ci_lower : float
        Lower bound of the confidence interval for the ATE.
    ci_upper : float
        Upper bound of the confidence interval for the ATE.
    significant : bool
        Whether ``pvalue < alpha``.
    n_strata : int
        Number of strata.
    stratum_effects : list[dict]
        Per-stratum details: each dict contains ``stratum``, ``ate``,
        ``se``, ``n_control``, ``n_treatment``, and ``weight``.
    alpha : float
        Significance level used.
    """

    ate: float
    se: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    significant: bool
    n_strata: int
    stratum_effects: list[dict]
    alpha: float

    def __repr__(self) -> str:
        w = 36
        lines = [
            "StratifiedResult",
            _line(w),
            f"  {'ate':<16}{_fmt(self.ate)}",
            f"  {'se':<16}{_fmt(self.se)}",
            f"  {'pvalue':<16}{_fmt(self.pvalue)}",
            f"  {'ci':<16}[{_fmt(self.ci_lower)}, {_fmt(self.ci_upper)}]",
            f"  {'significant':<16}{_fmt(self.significant)}",
            f"  {'alpha':<16}{_fmt(self.alpha)}",
            _line(w),
            f"  {'n_strata':<16}{self.n_strata}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── CSState ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CSState(_DictMixin):
    """Intermediate state of a confidence sequence.

    Attributes
    ----------
    n_control : int
        Cumulative control observations so far.
    n_treatment : int
        Cumulative treatment observations so far.
    effect_estimate : float
        Current point estimate of the treatment effect.
    ci_lower : float
        Lower bound of the confidence sequence.
    ci_upper : float
        Upper bound of the confidence sequence.
    width : float
        Width of the confidence interval (``ci_upper - ci_lower``).
    should_stop : bool
        Whether the CI excludes zero.
    """

    n_control: int
    n_treatment: int
    effect_estimate: float
    ci_lower: float
    ci_upper: float
    width: float
    should_stop: bool

    def __repr__(self) -> str:
        w = 36
        lines = [
            "CSState",
            _line(w),
            f"  {'n_control':<20}{self.n_control}",
            f"  {'n_treatment':<20}{self.n_treatment}",
            f"  {'effect_estimate':<20}{_fmt(self.effect_estimate)}",
            f"  {'ci':<20}[{_fmt(self.ci_lower)}, {_fmt(self.ci_upper)}]",
            f"  {'width':<20}{_fmt(self.width)}",
            f"  {'should_stop':<20}{_fmt(self.should_stop)}",
        ]
        return "\n".join(lines)


# ─── CSResult ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CSResult(_DictMixin):
    """Final result of a confidence sequence.

    Extends :class:`CSState` with stopping metadata.

    Attributes
    ----------
    n_control : int
        Total control observations.
    n_treatment : int
        Total treatment observations.
    effect_estimate : float
        Final point estimate of the treatment effect.
    ci_lower : float
        Lower bound of the confidence sequence at termination.
    ci_upper : float
        Upper bound of the confidence sequence at termination.
    width : float
        Width of the confidence interval.
    should_stop : bool
        Whether the test recommended stopping.
    stopping_reason : str
        Human-readable reason for stopping.
    total_observations : int
        ``n_control + n_treatment``.
    """

    n_control: int
    n_treatment: int
    effect_estimate: float
    ci_lower: float
    ci_upper: float
    width: float
    should_stop: bool
    stopping_reason: str
    total_observations: int

    def __repr__(self) -> str:
        w = 40
        lines = [
            "CSResult",
            _line(w),
            f"  {'n_control':<24}{self.n_control}",
            f"  {'n_treatment':<24}{self.n_treatment}",
            f"  {'total_observations':<24}{self.total_observations}",
            f"  {'effect_estimate':<24}{_fmt(self.effect_estimate)}",
            f"  {'ci':<24}[{_fmt(self.ci_lower)}, {_fmt(self.ci_upper)}]",
            f"  {'width':<24}{_fmt(self.width)}",
            f"  {'should_stop':<24}{_fmt(self.should_stop)}",
            _line(w),
            f"  {'stopping_reason':<24}{self.stopping_reason}",
        ]
        return "\n".join(lines)


# ─── EProcessState ────────────────────────────────────────────────────


@dataclass(frozen=True)
class EProcessState(_DictMixin):
    """Intermediate state of an e-process sequential test.

    Attributes
    ----------
    e_value : float
        Current product e-value (e-process).
    log_e_value : float
        Log of the current e-value (for numerical stability).
    n_control : int
        Cumulative control observations so far.
    n_treatment : int
        Cumulative treatment observations so far.
    should_stop : bool
        Whether the rejection threshold has been reached.
    """

    e_value: float
    log_e_value: float
    n_control: int
    n_treatment: int
    should_stop: bool

    def __repr__(self) -> str:
        w = 36
        lines = [
            "EProcessState",
            _line(w),
            f"  {'e_value':<20}{_fmt(self.e_value)}",
            f"  {'log_e_value':<20}{_fmt(self.log_e_value)}",
            f"  {'n_control':<20}{self.n_control}",
            f"  {'n_treatment':<20}{self.n_treatment}",
            f"  {'should_stop':<20}{_fmt(self.should_stop)}",
        ]
        return "\n".join(lines)


# ─── EProcessResult ──────────────────────────────────────────────────


@dataclass(frozen=True)
class EProcessResult(_DictMixin):
    """Final result of an e-process sequential test.

    Extends :class:`EProcessState` with stopping metadata and safe CI.

    Attributes
    ----------
    e_value : float
        Final product e-value at termination.
    log_e_value : float
        Log of the final e-value.
    n_control : int
        Total control observations.
    n_treatment : int
        Total treatment observations.
    should_stop : bool
        Whether the test recommended stopping.
    stopping_reason : str
        Human-readable reason for stopping.
    safe_ci_lower : float
        Lower bound of the safe confidence interval.
    safe_ci_upper : float
        Upper bound of the safe confidence interval.
    """

    e_value: float
    log_e_value: float
    n_control: int
    n_treatment: int
    should_stop: bool
    stopping_reason: str
    safe_ci_lower: float
    safe_ci_upper: float

    def __repr__(self) -> str:
        w = 36
        lines = [
            "EProcessResult",
            _line(w),
            f"  {'e_value':<20}{_fmt(self.e_value)}",
            f"  {'log_e_value':<20}{_fmt(self.log_e_value)}",
            f"  {'n_control':<20}{self.n_control}",
            f"  {'n_treatment':<20}{self.n_treatment}",
            f"  {'should_stop':<20}{_fmt(self.should_stop)}",
            f"  {'stopping_reason':<20}{self.stopping_reason}",
            f"  {'safe_ci':<20}"
            f"[{_fmt(self.safe_ci_lower)}, {_fmt(self.safe_ci_upper)}]",
        ]
        return "\n".join(lines)


# ─── DoubleMLResult ─────────────────────────────────────────────────


@dataclass(frozen=True)
class DoubleMLResult(_DictMixin):
    """Result of Double ML estimation via :class:`~splita.DoubleML`.

    Attributes
    ----------
    ate : float
        Average treatment effect estimate.
    se : float
        Standard error of the ATE estimate.
    pvalue : float
        Two-sided p-value for H0: ATE = 0.
    ci_lower : float
        Lower bound of the 95% confidence interval.
    ci_upper : float
        Upper bound of the 95% confidence interval.
    significant : bool
        Whether ``pvalue < alpha``.
    outcome_r2 : float
        Cross-validated R-squared of the outcome model.
    propensity_r2 : float
        Cross-validated R-squared of the propensity model.
    variance_reduction : float
        Fraction of variance reduced compared to a naive estimate.
    """

    ate: float
    se: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    significant: bool
    outcome_r2: float
    propensity_r2: float
    variance_reduction: float

    def __repr__(self) -> str:
        w = 36
        lines = [
            "DoubleMLResult",
            _line(w),
            f"  {'ate':<20}{_fmt(self.ate)}",
            f"  {'se':<20}{_fmt(self.se)}",
            f"  {'pvalue':<20}{_fmt(self.pvalue)}",
            f"  {'ci':<20}[{_fmt(self.ci_lower)}, {_fmt(self.ci_upper)}]",
            f"  {'significant':<20}{_fmt(self.significant)}",
            _line(w),
            f"  {'outcome_r2':<20}{_fmt(self.outcome_r2)}",
            f"  {'propensity_r2':<20}{_fmt(self.propensity_r2)}",
            f"  {'variance_reduction':<20}{_fmt(self.variance_reduction, as_pct=True)}",
        ]
        return "\n".join(lines)


# ─── MultiObjectiveResult ───────────────────────────────────────────


@dataclass(frozen=True)
class MultiObjectiveResult(_DictMixin):
    """Result of a multi-objective experiment.

    Produced by :class:`~splita.MultiObjectiveExperiment`.

    Attributes
    ----------
    metric_results : list[ExperimentResult]
        Individual experiment results for each metric.
    pareto_dominant : bool
        True if treatment is significantly better on all metrics.
    tradeoffs : list[str]
        Metric names where treatment and control directions conflict.
    corrected_pvalues : list[float]
        P-values after multiple testing correction.
    recommendation : str
        One of ``'adopt'``, ``'reject'``, or ``'tradeoff'``.
    """

    metric_results: list
    pareto_dominant: bool
    tradeoffs: list
    corrected_pvalues: list
    recommendation: str

    def __repr__(self) -> str:
        w = 36
        n = len(self.metric_results)
        lines = [
            "MultiObjectiveResult",
            _line(w),
            f"  {'n_metrics':<20}{n}",
            f"  {'recommendation':<20}{self.recommendation}",
            f"  {'pareto_dominant':<20}{_fmt(self.pareto_dominant)}",
            f"  {'tradeoffs':<20}{self.tradeoffs}",
            _line(w),
        ]
        for i, r in enumerate(self.metric_results):
            name = getattr(r, "_metric_name", f"metric_{i}")
            sig = "sig" if self.corrected_pvalues[i] < 0.05 else "ns"
            p_str = _fmt(self.corrected_pvalues[i])
            lines.append(f"  {name:<16} lift={_fmt(r.lift)}  p={p_str}  [{sig}]")
        return "\n".join(lines)


# ─── InterferenceResult ──────────────────────────────────────────


@dataclass(frozen=True)
class InterferenceResult(_DictMixin):
    """Result of an interference experiment via :class:`~splita.InterferenceExperiment`.

    Attributes
    ----------
    ate : float
        Average treatment effect (Horvitz-Thompson estimate).
    se : float
        Cluster-robust standard error.
    pvalue : float
        Two-sided p-value for H0: ATE = 0.
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    significant : bool
        Whether ``pvalue < alpha``.
    n_clusters : int
        Total number of clusters.
    design_effect : float
        Ratio of cluster-robust SE to naive SE (> 1 indicates interference).
    """

    ate: float
    se: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    significant: bool
    n_clusters: int
    design_effect: float

    def __repr__(self) -> str:
        w = 36
        lines = [
            "InterferenceResult",
            _line(w),
            f"  {'ate':<20}{_fmt(self.ate)}",
            f"  {'se':<20}{_fmt(self.se)}",
            f"  {'pvalue':<20}{_fmt(self.pvalue)}",
            f"  {'ci':<20}[{_fmt(self.ci_lower)}, {_fmt(self.ci_upper)}]",
            f"  {'significant':<20}{_fmt(self.significant)}",
            f"  {'n_clusters':<20}{self.n_clusters}",
            f"  {'design_effect':<20}{_fmt(self.design_effect)}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── NonStationaryResult ─────────────────────────────────────────


@dataclass(frozen=True)
class NonStationaryResult(_DictMixin):
    """Result of non-stationarity detection via :class:`~splita.NonStationaryDetector`.

    Attributes
    ----------
    is_stationary : bool
        True if the treatment effect appears stable over time.
    change_points : list[int]
        Indices in the time series where change points were detected.
    effect_trend : str
        One of ``'stable'``, ``'increasing'``, ``'decreasing'``, ``'volatile'``.
    window_effects : list[dict]
        Per-window effect estimates with keys ``'start'``, ``'end'``, ``'effect'``.
    """

    is_stationary: bool
    change_points: list
    effect_trend: str
    window_effects: list

    def __repr__(self) -> str:
        w = 36
        lines = [
            "NonStationaryResult",
            _line(w),
            f"  {'is_stationary':<20}{_fmt(self.is_stationary)}",
            f"  {'effect_trend':<20}{self.effect_trend}",
            f"  {'n_change_points':<20}{len(self.change_points)}",
            f"  {'n_windows':<20}{len(self.window_effects)}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── SurrogateIndexResult ────────────────────────────────────────


@dataclass(frozen=True)
class SurrogateIndexResult(_DictMixin):
    """Result of surrogate index estimation via :class:`~splita.SurrogateIndex`.

    Attributes
    ----------
    predicted_effect : float
        Predicted long-term treatment effect via the surrogate index.
    se : float
        Standard error of the predicted effect (delta method).
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    surrogate_r2 : float
        R-squared of the surrogate model on held-out data.
    is_valid : bool
        True if ``surrogate_r2 > 0.3`` (surrogates are informative).
    n_surrogates : int
        Number of short-term surrogate outcomes used.
    """

    predicted_effect: float
    se: float
    ci_lower: float
    ci_upper: float
    surrogate_r2: float
    is_valid: bool
    n_surrogates: int

    def __repr__(self) -> str:
        w = 36
        lines = [
            "SurrogateIndexResult",
            _line(w),
            f"  {'predicted_effect':<20}{_fmt(self.predicted_effect)}",
            f"  {'se':<20}{_fmt(self.se)}",
            f"  {'ci':<20}[{_fmt(self.ci_lower)}, {_fmt(self.ci_upper)}]",
            f"  {'surrogate_r2':<20}{_fmt(self.surrogate_r2)}",
            f"  {'is_valid':<20}{_fmt(self.is_valid)}",
            f"  {'n_surrogates':<20}{self.n_surrogates}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── PairwiseDesignResult ────────────────────────────────────────


@dataclass(frozen=True)
class PairwiseDesignResult(_DictMixin):
    """Result of pairwise matching design via :class:`~splita.PairwiseDesign`.

    Attributes
    ----------
    assignments : list[int]
        Treatment assignment for each unit (0 = control, 1 = treatment).
    pairs : list[tuple[int, int]]
        Matched pairs as (control_index, treatment_index).
    balance_score : float
        Maximum standardised mean difference across features.
    n_pairs : int
        Number of matched pairs.
    """

    assignments: list
    pairs: list
    balance_score: float
    n_pairs: int

    def __repr__(self) -> str:
        w = 36
        lines = [
            "PairwiseDesignResult",
            _line(w),
            f"  {'n_pairs':<20}{self.n_pairs}",
            f"  {'balance_score':<20}{_fmt(self.balance_score)}",
            f"  {'n_units':<20}{len(self.assignments)}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── CausalForestResult ─────────────────────────────────────────


@dataclass(frozen=True)
class CausalForestResult(_DictMixin):
    """Result of causal forest estimation via :class:`~splita.CausalForest`.

    Attributes
    ----------
    mean_cate : float
        Mean of the CATE estimates across all observations.
    cate_std : float
        Standard deviation of the CATE estimates.
    feature_importances : list[float]
        Feature importance scores for each covariate.
    ci_lower : float
        Lower bound of the jackknife confidence interval for the mean CATE.
    ci_upper : float
        Upper bound of the jackknife confidence interval for the mean CATE.
    """

    mean_cate: float
    cate_std: float
    feature_importances: list
    ci_lower: float
    ci_upper: float

    def __repr__(self) -> str:
        w = 36
        lines = [
            "CausalForestResult",
            _line(w),
            f"  {'mean_cate':<20}{_fmt(self.mean_cate)}",
            f"  {'cate_std':<20}{_fmt(self.cate_std)}",
            f"  {'ci':<20}[{_fmt(self.ci_lower)}, {_fmt(self.ci_upper)}]",
            f"  {'n_features':<20}{len(self.feature_importances)}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── GuardrailResult ───────────────────────────────────────────────


@dataclass(frozen=True)
class GuardrailResult(_DictMixin):
    """Result of guardrail monitoring via :class:`~splita.GuardrailMonitor`.

    Attributes
    ----------
    all_passed : bool
        True if all guardrail checks passed (no significant degradation).
    guardrail_results : list[dict]
        Per-guardrail results. Each dict has keys: ``name``, ``passed``,
        ``pvalue``, ``lift``, ``direction``.
    message : str
        Human-readable summary.
    recommendation : str
        One of ``'safe'``, ``'warning'``, or ``'stop'``.
    """

    all_passed: bool
    guardrail_results: list
    message: str
    recommendation: str

    def __repr__(self) -> str:
        w = 40
        lines = [
            "GuardrailResult",
            _line(w),
            f"  {'all_passed':<22}{_fmt(self.all_passed)}",
            f"  {'recommendation':<22}{self.recommendation}",
            f"  {'n_guardrails':<22}{len(self.guardrail_results)}",
            _line(w),
            f"  {self.message}",
        ]
        return "\n".join(lines)


# ─── FlickerResult ────────────────────────────────────────────────


@dataclass(frozen=True)
class FlickerResult(_DictMixin):
    """Result of flicker detection via :class:`~splita.FlickerDetector`.

    Attributes
    ----------
    flicker_rate : float
        Fraction of users who switched variants.
    n_flickers : int
        Number of users who flickered.
    n_users : int
        Total number of unique users.
    is_problematic : bool
        True if the flicker rate exceeds the threshold.
    flicker_users : list
        List of user IDs that flickered.
    message : str
        Human-readable summary.
    """

    flicker_rate: float
    n_flickers: int
    n_users: int
    is_problematic: bool
    flicker_users: list
    message: str

    def __repr__(self) -> str:
        w = 36
        lines = [
            "FlickerResult",
            _line(w),
            f"  {'flicker_rate':<20}{_fmt(self.flicker_rate, as_pct=True)}",
            f"  {'n_flickers':<20}{self.n_flickers}",
            f"  {'n_users':<20}{self.n_users}",
            f"  {'is_problematic':<20}{_fmt(self.is_problematic)}",
            _line(w),
            f"  {self.message}",
        ]
        return "\n".join(lines)


# ─── TrimmedMeanResult ────────────────────────────────────────────


@dataclass(frozen=True)
class TrimmedMeanResult(_DictMixin):
    """Result of trimmed-mean estimation via :class:`~splita.TrimmedMeanEstimator`.

    Attributes
    ----------
    ate : float
        Average treatment effect (trimmed treatment mean - trimmed control mean).
    se : float
        Standard error of the ATE.
    pvalue : float
        P-value from the t-test on trimmed data.
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    significant : bool
        True if pvalue < alpha.
    n_trimmed_control : int
        Number of observations remaining in control after trimming.
    n_trimmed_treatment : int
        Number of observations remaining in treatment after trimming.
    trim_fraction : float
        Fraction trimmed from each tail.
    """

    ate: float
    se: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    significant: bool
    n_trimmed_control: int
    n_trimmed_treatment: int
    trim_fraction: float

    def __repr__(self) -> str:
        w = 40
        lines = [
            "TrimmedMeanResult",
            _line(w),
            f"  {'ate':<22}{_fmt(self.ate)}",
            f"  {'se':<22}{_fmt(self.se)}",
            f"  {'pvalue':<22}{_fmt(self.pvalue)}",
            f"  {'ci':<22}[{_fmt(self.ci_lower)}, {_fmt(self.ci_upper)}]",
            f"  {'significant':<22}{_fmt(self.significant)}",
            f"  {'trim_fraction':<22}{_fmt(self.trim_fraction)}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── PermutationResult ────────────────────────────────────────────


@dataclass(frozen=True)
class PermutationResult(_DictMixin):
    """Result of a permutation test via :class:`~splita.PermutationTest`.

    Attributes
    ----------
    observed_statistic : float
        The observed test statistic.
    pvalue : float
        Permutation-based p-value.
    significant : bool
        True if pvalue < alpha.
    n_permutations : int
        Number of permutations performed.
    alpha : float
        Significance level used.
    null_distribution_mean : float
        Mean of the null distribution.
    null_distribution_std : float
        Standard deviation of the null distribution.
    """

    observed_statistic: float
    pvalue: float
    significant: bool
    n_permutations: int
    alpha: float
    null_distribution_mean: float
    null_distribution_std: float

    def __repr__(self) -> str:
        w = 40
        lines = [
            "PermutationResult",
            _line(w),
            f"  {'observed_statistic':<22}{_fmt(self.observed_statistic)}",
            f"  {'pvalue':<22}{_fmt(self.pvalue)}",
            f"  {'significant':<22}{_fmt(self.significant)}",
            f"  {'n_permutations':<22}{self.n_permutations}",
            f"  {'alpha':<22}{_fmt(self.alpha)}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── OECResult ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class OECResult(_DictMixin):
    """Result of an Overall Evaluation Criterion analysis via :class:`~splita.OECBuilder`.

    Attributes
    ----------
    oec_lift : float
        Weighted sum of (optionally normalised) metric lifts.
    pvalue : float
        Two-sided p-value from a t-test on the OEC scores.
    significant : bool
        Whether ``pvalue < alpha``.
    ci_lower : float
        Lower bound of the confidence interval for the OEC lift.
    ci_upper : float
        Upper bound of the confidence interval for the OEC lift.
    metric_contributions : list[float]
        Weighted contribution of each metric to the OEC lift.
    weights : list[float]
        Normalised weights used for each metric.
    """

    oec_lift: float
    pvalue: float
    significant: bool
    ci_lower: float
    ci_upper: float
    metric_contributions: list
    weights: list

    def __repr__(self) -> str:
        w = 36
        lines = [
            "OECResult",
            _line(w),
            f"  {'oec_lift':<20}{_fmt(self.oec_lift)}",
            f"  {'pvalue':<20}{_fmt(self.pvalue)}",
            f"  {'ci':<20}[{_fmt(self.ci_lower)}, {_fmt(self.ci_upper)}]",
            f"  {'significant':<20}{_fmt(self.significant)}",
            f"  {'n_metrics':<20}{len(self.weights)}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── FunnelResult ──────────────────────────────────────────────────


@dataclass(frozen=True)
class FunnelResult(_DictMixin):
    """Result of a funnel experiment via :class:`~splita.FunnelExperiment`.

    Attributes
    ----------
    step_results : list[dict]
        Per-step results with keys ``name``, ``control_rate``, ``treatment_rate``,
        ``lift``, ``pvalue``, ``significant``, ``conditional_lift``, ``conditional_pvalue``.
    bottleneck_step : str
        Name of the step with the biggest negative lift (or smallest lift).
    overall_lift : float
        Overall conversion lift from first step to last step.
    """

    step_results: list
    bottleneck_step: str
    overall_lift: float

    def __repr__(self) -> str:
        w = 36
        lines = [
            "FunnelResult",
            _line(w),
            f"  {'n_steps':<20}{len(self.step_results)}",
            f"  {'overall_lift':<20}{_fmt(self.overall_lift)}",
            f"  {'bottleneck':<20}{self.bottleneck_step}",
            _line(w),
        ]
        for s in self.step_results:
            sig = "sig" if s.get("significant") else "ns"
            lines.append(
                f"  {s['name']:<16} lift={_fmt(s['lift'])}  p={_fmt(s['pvalue'])}  [{sig}]"
            )
        return "\n".join(lines)


# ─── PostStratResult ───────────────────────────────────────────────


@dataclass(frozen=True)
class PostStratResult(_DictMixin):
    """Result of post-stratification variance reduction via :class:`~splita.PostStratification`.

    Attributes
    ----------
    ate : float
        Average treatment effect (variance-optimally weighted across strata).
    se : float
        Standard error of the ATE estimate.
    pvalue : float
        Two-sided p-value for H0: ATE = 0.
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    significant : bool
        Whether ``pvalue < alpha``.
    variance_reduction : float
        Fraction of variance reduced compared to a naive (unstratified) estimate.
    n_strata : int
        Number of strata used.
    """

    ate: float
    se: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    significant: bool
    variance_reduction: float
    n_strata: int

    def __repr__(self) -> str:
        w = 36
        lines = [
            "PostStratResult",
            _line(w),
            f"  {'ate':<20}{_fmt(self.ate)}",
            f"  {'se':<20}{_fmt(self.se)}",
            f"  {'pvalue':<20}{_fmt(self.pvalue)}",
            f"  {'ci':<20}[{_fmt(self.ci_lower)}, {_fmt(self.ci_upper)}]",
            f"  {'significant':<20}{_fmt(self.significant)}",
            f"  {'variance_reduction':<20}{_fmt(self.variance_reduction, as_pct=True)}",
            f"  {'n_strata':<20}{self.n_strata}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── ClusterBootstrapResult ────────────────────────────────────────


@dataclass(frozen=True)
class ClusterBootstrapResult(_DictMixin):
    """Result of cluster bootstrap inference via :class:`~splita.ClusterBootstrap`.

    Attributes
    ----------
    ate : float
        Average treatment effect (difference of cluster-level means).
    se : float
        Bootstrap standard error.
    pvalue : float
        Two-sided p-value from the bootstrap distribution.
    ci_lower : float
        Lower bound of the bootstrap percentile confidence interval.
    ci_upper : float
        Upper bound of the bootstrap percentile confidence interval.
    significant : bool
        Whether ``pvalue < alpha``.
    n_clusters : int
        Total number of clusters across both groups.
    """

    ate: float
    se: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    significant: bool
    n_clusters: int

    def __repr__(self) -> str:
        w = 36
        lines = [
            "ClusterBootstrapResult",
            _line(w),
            f"  {'ate':<20}{_fmt(self.ate)}",
            f"  {'se':<20}{_fmt(self.se)}",
            f"  {'pvalue':<20}{_fmt(self.pvalue)}",
            f"  {'ci':<20}[{_fmt(self.ci_lower)}, {_fmt(self.ci_upper)}]",
            f"  {'significant':<20}{_fmt(self.significant)}",
            f"  {'n_clusters':<20}{self.n_clusters}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── RandomizationResult ──────────────────────────────────────────


@dataclass(frozen=True)
class RandomizationResult(_DictMixin):
    """Result of randomization balance check via :class:`~splita.RandomizationValidator`.

    Attributes
    ----------
    balanced : bool
        True if all covariates are balanced (no SMD exceeds threshold).
    smd_per_covariate : list[dict]
        Per-covariate standardised mean difference with keys ``name`` and ``smd``.
    max_smd : float
        Maximum absolute standardised mean difference across all covariates.
    omnibus_pvalue : float
        p-value from a chi-squared omnibus test for overall balance.
    imbalanced_covariates : list[str]
        Names of covariates with SMD exceeding the threshold (0.1).
    """

    balanced: bool
    smd_per_covariate: list
    max_smd: float
    omnibus_pvalue: float
    imbalanced_covariates: list

    def __repr__(self) -> str:
        w = 36
        lines = [
            "RandomizationResult",
            _line(w),
            f"  {'balanced':<20}{_fmt(self.balanced)}",
            f"  {'max_smd':<20}{_fmt(self.max_smd)}",
            f"  {'omnibus_pvalue':<20}{_fmt(self.omnibus_pvalue)}",
            f"  {'n_covariates':<20}{len(self.smd_per_covariate)}",
            f"  {'n_imbalanced':<20}{len(self.imbalanced_covariates)}",
            _line(w),
        ]
        if self.imbalanced_covariates:
            lines.append(f"  {'imbalanced':<20}{self.imbalanced_covariates}")
        return "\n".join(lines)


# ─── InterleavingResult ───────────────────────────────────────────


@dataclass(frozen=True)
class InterleavingResult(_DictMixin):
    """Result of an interleaving experiment for ranking comparison.

    Attributes
    ----------
    preference_a : float
        Fraction of queries where ranking A was preferred.
    preference_b : float
        Fraction of queries where ranking B was preferred.
    pvalue : float
        p-value from a binomial test of preference.
    significant : bool
        Whether the result is significant at the given alpha.
    winner : str
        ``"A"``, ``"B"``, or ``"tie"``.
    n_queries : int
        Number of queries evaluated.
    delta : float
        Effect size (preference_a - preference_b).
    """

    preference_a: float
    preference_b: float
    pvalue: float
    significant: bool
    winner: str
    n_queries: int
    delta: float

    def __repr__(self) -> str:
        w = 40
        lines = [
            "InterleavingResult",
            _line(w),
            f"  {'preference_a':<22}{_fmt(self.preference_a)}",
            f"  {'preference_b':<22}{_fmt(self.preference_b)}",
            f"  {'pvalue':<22}{_fmt(self.pvalue)}",
            f"  {'significant':<22}{_fmt(self.significant)}",
            f"  {'winner':<22}{self.winner}",
            f"  {'n_queries':<22}{self.n_queries}",
            f"  {'delta':<22}{_fmt(self.delta)}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── CarryoverResult ──────────────────────────────────────────────


@dataclass(frozen=True)
class CarryoverResult(_DictMixin):
    """Result of a carryover effect detection.

    Attributes
    ----------
    has_carryover : bool
        Whether carryover contamination was detected.
    control_change_pvalue : float
        p-value for the control group pre/post difference.
    control_pre_mean : float
        Mean of control group pre-experiment.
    control_post_mean : float
        Mean of control group post-experiment.
    message : str
        Human-readable summary.
    """

    has_carryover: bool
    control_change_pvalue: float
    control_pre_mean: float
    control_post_mean: float
    message: str

    def __repr__(self) -> str:
        w = 40
        lines = [
            "CarryoverResult",
            _line(w),
            f"  {'has_carryover':<22}{_fmt(self.has_carryover)}",
            f"  {'control_change_pvalue':<22}{_fmt(self.control_change_pvalue)}",
            f"  {'control_pre_mean':<22}{_fmt(self.control_pre_mean)}",
            f"  {'control_post_mean':<22}{_fmt(self.control_post_mean)}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── RobustMeanResult ─────────────────────────────────────────────


@dataclass(frozen=True)
class RobustMeanResult(_DictMixin):
    """Result of a robust mean estimation.

    Attributes
    ----------
    ate : float
        Average treatment effect (robust).
    se : float
        Standard error of the ATE estimate.
    pvalue : float
        p-value for the ATE.
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    significant : bool
        Whether the result is significant at the given alpha.
    method : str
        Estimation method used.
    """

    ate: float
    se: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    significant: bool
    method: str

    def __repr__(self) -> str:
        w = 40
        lines = [
            "RobustMeanResult",
            _line(w),
            f"  {'ate':<22}{_fmt(self.ate)}",
            f"  {'se':<22}{_fmt(self.se)}",
            f"  {'pvalue':<22}{_fmt(self.pvalue)}",
            f"  {'ci_lower':<22}{_fmt(self.ci_lower)}",
            f"  {'ci_upper':<22}{_fmt(self.ci_upper)}",
            f"  {'significant':<22}{_fmt(self.significant)}",
            f"  {'method':<22}{self.method}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── IVResult ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class IVResult(_DictMixin):
    """Result of an instrumental variables (2SLS) estimation.

    Attributes
    ----------
    late : float
        Local Average Treatment Effect.
    se : float
        Standard error of the LATE estimate.
    pvalue : float
        p-value for the LATE.
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    significant : bool
        Whether the result is significant at the given alpha.
    first_stage_f : float
        F-statistic from the first stage regression.
    weak_instrument : bool
        Whether the instrument is weak (F < 10).
    """

    late: float
    se: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    significant: bool
    first_stage_f: float
    weak_instrument: bool

    def __repr__(self) -> str:
        w = 40
        lines = [
            "IVResult",
            _line(w),
            f"  {'late':<22}{_fmt(self.late)}",
            f"  {'se':<22}{_fmt(self.se)}",
            f"  {'pvalue':<22}{_fmt(self.pvalue)}",
            f"  {'ci_lower':<22}{_fmt(self.ci_lower)}",
            f"  {'ci_upper':<22}{_fmt(self.ci_upper)}",
            f"  {'significant':<22}{_fmt(self.significant)}",
            f"  {'first_stage_f':<22}{_fmt(self.first_stage_f)}",
            f"  {'weak_instrument':<22}{_fmt(self.weak_instrument)}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── PSMResult ────────────────────────────────────────────────────


@dataclass(frozen=True)
class PSMResult(_DictMixin):
    """Result of propensity score matching.

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
    n_matched : int
        Number of matched treatment units.
    n_unmatched : int
        Number of unmatched treatment units (dropped by caliper).
    balance_before : dict
        Standardised mean differences before matching per covariate.
    balance_after : dict
        Standardised mean differences after matching per covariate.
    """

    att: float
    se: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    significant: bool
    n_matched: int
    n_unmatched: int
    balance_before: dict
    balance_after: dict

    def __repr__(self) -> str:
        w = 40
        lines = [
            "PSMResult",
            _line(w),
            f"  {'att':<22}{_fmt(self.att)}",
            f"  {'se':<22}{_fmt(self.se)}",
            f"  {'pvalue':<22}{_fmt(self.pvalue)}",
            f"  {'ci_lower':<22}{_fmt(self.ci_lower)}",
            f"  {'ci_upper':<22}{_fmt(self.ci_upper)}",
            f"  {'significant':<22}{_fmt(self.significant)}",
            f"  {'n_matched':<22}{self.n_matched}",
            f"  {'n_unmatched':<22}{self.n_unmatched}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── RARResult ────────────────────────────────────────────────────


@dataclass(frozen=True)
class RARResult(_DictMixin):
    """Result of response-adaptive randomization.

    Attributes
    ----------
    allocations : list[float]
        Current allocation probabilities per arm.
    n_per_arm : list[int]
        Number of observations per arm.
    best_arm : int
        Index of the arm with the highest allocation.
    total_observations : int
        Total number of observations across all arms.
    """

    allocations: list
    n_per_arm: list
    best_arm: int
    total_observations: int

    def __repr__(self) -> str:
        w = 40
        lines = [
            "RARResult",
            _line(w),
            f"  {'allocations':<22}{self.allocations}",
            f"  {'n_per_arm':<22}{self.n_per_arm}",
            f"  {'best_arm':<22}{self.best_arm}",
            f"  {'total_observations':<22}{self.total_observations}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── InExperimentVRResult ─────────────────────────────────────────────


@dataclass(frozen=True)
class InExperimentVRResult(_DictMixin):
    """Result of in-experiment variance reduction.

    Attributes
    ----------
    theta : float
        Fitted adjustment coefficient.
    variance_reduction : float
        Fraction of variance explained (R-squared) by the covariate.
    control_adjusted : list[float]
        Adjusted control observations.
    treatment_adjusted : list[float]
        Adjusted treatment observations.
    """

    theta: float
    variance_reduction: float
    control_adjusted: list[float]
    treatment_adjusted: list[float]

    def __repr__(self) -> str:
        w = 36
        lines = [
            "InExperimentVRResult",
            _line(w),
            f"  {'theta':<20}{_fmt(self.theta)}",
            f"  {'variance_reduction':<20}{_fmt(self.variance_reduction, as_pct=True)}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── MetricDecompResult ──────────────────────────────────────────────


@dataclass(frozen=True)
class MetricDecompResult(_DictMixin):
    """Result of metric decomposition analysis.

    Attributes
    ----------
    total_lift : float
        Absolute lift on the total metric.
    total_pvalue : float
        p-value for the total metric test.
    component_results : dict
        Per-component results with keys: lift, pvalue, significant, contribution.
    dominant_component : str or None
        Name of the component driving the overall effect, or None.
    """

    total_lift: float
    total_pvalue: float
    component_results: dict
    dominant_component: str | None

    def __repr__(self) -> str:
        w = 40
        lines = [
            "MetricDecompResult",
            _line(w),
            f"  {'total_lift':<22}{_fmt(self.total_lift)}",
            f"  {'total_pvalue':<22}{_fmt(self.total_pvalue)}",
            f"  {'dominant_component':<22}{self.dominant_component or 'none'}",
            _line(w),
        ]
        for name, info in self.component_results.items():
            lines.append(
                f"  {name:<16} lift={_fmt(info['lift'])}  "
                f"p={_fmt(info['pvalue'])}"
            )
        return "\n".join(lines)


# ─── ObjectiveBayesianResult ─────────────────────────────────────────


@dataclass(frozen=True)
class ObjectiveBayesianResult(_DictMixin):
    """Result of an objective Bayesian experiment.

    Attributes
    ----------
    prior_mean : float
        Mean of the learned (or specified) prior.
    prior_std : float
        Standard deviation of the learned (or specified) prior.
    posterior_mean : float
        Mean of the posterior distribution for the treatment effect.
    posterior_std : float
        Standard deviation of the posterior.
    prob_positive : float
        Posterior probability that the effect is positive.
    ci_lower : float
        Lower bound of the 95% credible interval.
    ci_upper : float
        Upper bound of the 95% credible interval.
    shrinkage : float
        Fraction by which the posterior is shrunk toward the prior mean.
    """

    prior_mean: float
    prior_std: float
    posterior_mean: float
    posterior_std: float
    prob_positive: float
    ci_lower: float
    ci_upper: float
    shrinkage: float

    def __repr__(self) -> str:
        w = 36
        lines = [
            "ObjectiveBayesianResult",
            _line(w),
            f"  {'prior_mean':<20}{_fmt(self.prior_mean)}",
            f"  {'prior_std':<20}{_fmt(self.prior_std)}",
            f"  {'posterior_mean':<20}{_fmt(self.posterior_mean)}",
            f"  {'posterior_std':<20}{_fmt(self.posterior_std)}",
            f"  {'prob_positive':<20}{_fmt(self.prob_positive, as_pct=True)}",
            f"  {'ci_lower':<20}{_fmt(self.ci_lower)}",
            f"  {'ci_upper':<20}{_fmt(self.ci_upper)}",
            f"  {'shrinkage':<20}{_fmt(self.shrinkage, as_pct=True)}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── RiskAwareResult ─────────────────────────────────────────────────


@dataclass(frozen=True)
class RiskAwareResult(_DictMixin):
    """Result of a risk-aware decision.

    Attributes
    ----------
    decision : str
        One of ``'ship'``, ``'hold'``, or ``'investigate'``.
    constraints_met : bool
        Whether all metric constraints are satisfied.
    violations : list[str]
        Names of metrics that violated their constraints.
    metric_details : dict
        Per-metric analysis: lift, pvalue, ci_lower, ci_upper, significant.
    """

    decision: str
    constraints_met: bool
    violations: list[str]
    metric_details: dict

    def __repr__(self) -> str:
        w = 36
        lines = [
            "RiskAwareResult",
            _line(w),
            f"  {'decision':<20}{self.decision}",
            f"  {'constraints_met':<20}{_fmt(self.constraints_met)}",
            f"  {'n_violations':<20}{len(self.violations)}",
            _line(w),
        ]
        if self.violations:
            lines.append(f"  {'violations':<20}{self.violations}")
        return "\n".join(lines)


# ─── ProxyResult ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProxyResult(_DictMixin):
    """Result of optimal proxy metric learning.

    Attributes
    ----------
    weights : list[float]
        Learned weights for each candidate metric.
    correlation_with_north_star : float
        Pearson correlation of the optimal proxy with the north star metric.
    optimal_proxy_values : list[float]
        The composite proxy metric values.
    """

    weights: list[float]
    correlation_with_north_star: float
    optimal_proxy_values: list[float]

    def __repr__(self) -> str:
        w = 36
        lines = [
            "ProxyResult",
            _line(w),
            f"  {'correlation':<20}{_fmt(self.correlation_with_north_star)}",
            f"  {'n_metrics':<20}{len(self.weights)}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── DoublyRobustResult ─────────────────────────────────────────────


@dataclass(frozen=True)
class DoublyRobustResult(_DictMixin):
    """Result of doubly robust (AIPW) estimation.

    Attributes
    ----------
    ate : float
        Average treatment effect estimate.
    se : float
        Standard error of the ATE.
    pvalue : float
        Two-sided p-value for the null H0: ATE = 0.
    ci_lower : float
        Lower bound of the 95% confidence interval.
    ci_upper : float
        Upper bound of the 95% confidence interval.
    outcome_r2 : float
        R-squared of the outcome model.
    propensity_auc : float
        AUC of the propensity score model.
    """

    ate: float
    se: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    outcome_r2: float
    propensity_auc: float

    def __repr__(self) -> str:
        w = 36
        lines = [
            "DoublyRobustResult",
            _line(w),
            f"  {'ate':<20}{_fmt(self.ate)}",
            f"  {'se':<20}{_fmt(self.se)}",
            f"  {'pvalue':<20}{_fmt(self.pvalue)}",
            f"  {'ci_lower':<20}{_fmt(self.ci_lower)}",
            f"  {'ci_upper':<20}{_fmt(self.ci_upper)}",
            f"  {'outcome_r2':<20}{_fmt(self.outcome_r2)}",
            f"  {'propensity_auc':<20}{_fmt(self.propensity_auc)}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── ReestimationResult ─────────────────────────────────────────────


@dataclass(frozen=True)
class ReestimationResult(_DictMixin):
    """Result of sample size re-estimation.

    Attributes
    ----------
    new_n_per_variant : int
        Updated sample size per variant.
    conditional_power_current : float
        Conditional power at the current sample size.
    conditional_power_new : float
        Conditional power at the updated sample size.
    increase_ratio : float
        Ratio of new sample size to current sample size.
    """

    new_n_per_variant: int
    conditional_power_current: float
    conditional_power_new: float
    increase_ratio: float

    def __repr__(self) -> str:
        w = 36
        lines = [
            "ReestimationResult",
            _line(w),
            f"  {'new_n_per_variant':<22}{self.new_n_per_variant}",
            f"  {'cond_power_current':<22}{_fmt(self.conditional_power_current, as_pct=True)}",
            f"  {'cond_power_new':<22}{_fmt(self.conditional_power_new, as_pct=True)}",
            f"  {'increase_ratio':<22}{_fmt(self.increase_ratio)}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── YEASTState / YEASTResult ────────────────────────────────────────


@dataclass(frozen=True)
class YEASTState(_DictMixin):
    """Intermediate state of a YEAST sequential test.

    Attributes
    ----------
    n_control : int
        Cumulative control observations.
    n_treatment : int
        Cumulative treatment observations.
    z_statistic : float
        Current z-score test statistic.
    boundary : float
        Current significance boundary from Levy's inequality.
    pvalue : float
        Current sequential p-value.
    should_stop : bool
        Whether the boundary has been crossed.
    current_effect_estimate : float
        Current estimate of the treatment effect.
    """

    n_control: int
    n_treatment: int
    z_statistic: float
    boundary: float
    pvalue: float
    should_stop: bool
    current_effect_estimate: float

    def __repr__(self) -> str:
        w = 36
        lines = [
            "YEASTState",
            _line(w),
            f"  {'n_control':<22}{self.n_control}",
            f"  {'n_treatment':<22}{self.n_treatment}",
            f"  {'z_statistic':<22}{_fmt(self.z_statistic)}",
            f"  {'boundary':<22}{_fmt(self.boundary)}",
            f"  {'should_stop':<22}{_fmt(self.should_stop)}",
            _line(w),
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class YEASTResult(_DictMixin):
    """Final result of a YEAST sequential test.

    Attributes
    ----------
    n_control : int
        Total control observations.
    n_treatment : int
        Total treatment observations.
    z_statistic : float
        Final z-score test statistic.
    boundary : float
        Significance boundary used.
    pvalue : float
        Sequential p-value.
    should_stop : bool
        Whether the test concluded.
    current_effect_estimate : float
        Final treatment effect estimate.
    stopping_reason : str
        One of ``'boundary_crossed'``, ``'not_stopped'``.
    total_observations : int
        Total observations across both groups.
    """

    n_control: int
    n_treatment: int
    z_statistic: float
    boundary: float
    pvalue: float
    should_stop: bool
    current_effect_estimate: float
    stopping_reason: str
    total_observations: int

    def __repr__(self) -> str:
        w = 36
        lines = [
            "YEASTResult",
            _line(w),
            f"  {'n_control':<22}{self.n_control}",
            f"  {'n_treatment':<22}{self.n_treatment}",
            f"  {'z_statistic':<22}{_fmt(self.z_statistic)}",
            f"  {'pvalue':<22}{_fmt(self.pvalue)}",
            f"  {'should_stop':<22}{_fmt(self.should_stop)}",
            f"  {'stopping_reason':<22}{self.stopping_reason}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── RDDResult ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class RDDResult(_DictMixin):
    """Result of a regression discontinuity design analysis.

    Attributes
    ----------
    late : float
        Local average treatment effect at the cutoff.
    se : float
        Standard error of the LATE estimate.
    pvalue : float
        Two-sided p-value.
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    bandwidth_used : float
        Bandwidth used for local linear regression.
    n_left : int
        Number of observations to the left of the cutoff.
    n_right : int
        Number of observations to the right of the cutoff.
    """

    late: float
    se: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    bandwidth_used: float
    n_left: int
    n_right: int

    def __repr__(self) -> str:
        w = 36
        lines = [
            "RDDResult",
            _line(w),
            f"  {'late':<20}{_fmt(self.late)}",
            f"  {'se':<20}{_fmt(self.se)}",
            f"  {'pvalue':<20}{_fmt(self.pvalue)}",
            f"  {'ci':<20}[{_fmt(self.ci_lower)}, {_fmt(self.ci_upper)}]",
            f"  {'bandwidth_used':<20}{_fmt(self.bandwidth_used)}",
            f"  {'n_left':<20}{self.n_left}",
            f"  {'n_right':<20}{self.n_right}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── NonstationaryAdjResult ──────────────────────────────────────


@dataclass(frozen=True)
class NonstationaryAdjResult(_DictMixin):
    """Result of nonstationary adjustment (Chen et al. 2024).

    Attributes
    ----------
    ate_corrected : float
        Bias-corrected average treatment effect.
    ate_naive : float
        Naive (unadjusted) average treatment effect.
    bias : float
        Estimated bias from nonstationarity.
    se : float
        Standard error of the corrected estimator.
    pvalue : float
        Two-sided p-value for the corrected estimator.
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    """

    ate_corrected: float
    ate_naive: float
    bias: float
    se: float
    pvalue: float
    ci_lower: float
    ci_upper: float

    def __repr__(self) -> str:
        w = 36
        lines = [
            "NonstationaryAdjResult",
            _line(w),
            f"  {'ate_corrected':<20}{_fmt(self.ate_corrected)}",
            f"  {'ate_naive':<20}{_fmt(self.ate_naive)}",
            f"  {'bias':<20}{_fmt(self.bias)}",
            f"  {'se':<20}{_fmt(self.se)}",
            f"  {'pvalue':<20}{_fmt(self.pvalue)}",
            f"  {'ci':<20}[{_fmt(self.ci_lower)}, {_fmt(self.ci_upper)}]",
            _line(w),
        ]
        return "\n".join(lines)


# ─── GeoResult ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class GeoResult(_DictMixin):
    """Result of a geo experiment analysis.

    Attributes
    ----------
    incremental_effect : float
        Estimated incremental treatment effect.
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    pre_rmse : float
        Root-mean-square error of the synthetic control in the pre-period.
    n_treated_regions : int
        Number of treated regions.
    n_control_regions : int
        Number of control regions.
    """

    incremental_effect: float
    ci_lower: float
    ci_upper: float
    pre_rmse: float
    n_treated_regions: int
    n_control_regions: int

    def __repr__(self) -> str:
        w = 36
        lines = [
            "GeoResult",
            _line(w),
            f"  {'incremental_effect':<20}{_fmt(self.incremental_effect)}",
            f"  {'ci':<20}[{_fmt(self.ci_lower)}, {_fmt(self.ci_upper)}]",
            f"  {'pre_rmse':<20}{_fmt(self.pre_rmse)}",
            f"  {'n_treated_regions':<20}{self.n_treated_regions}",
            f"  {'n_control_regions':<20}{self.n_control_regions}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── PPIResult ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class PPIResult(_DictMixin):
    """Result of prediction-powered inference (Angelopoulos et al. 2023).

    Attributes
    ----------
    mean_estimate : float
        PPI mean estimate.
    se : float
        Standard error.
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    n_labeled : int
        Number of labeled observations.
    n_unlabeled : int
        Number of unlabeled observations.
    """

    mean_estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    n_labeled: int
    n_unlabeled: int

    def __repr__(self) -> str:
        w = 36
        lines = [
            "PPIResult",
            _line(w),
            f"  {'mean_estimate':<20}{_fmt(self.mean_estimate)}",
            f"  {'se':<20}{_fmt(self.se)}",
            f"  {'ci':<20}[{_fmt(self.ci_lower)}, {_fmt(self.ci_upper)}]",
            f"  {'n_labeled':<20}{self.n_labeled}",
            f"  {'n_unlabeled':<20}{self.n_unlabeled}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── TransportResult ──────────────────────────────────────────────


@dataclass(frozen=True)
class TransportResult(_DictMixin):
    """Result of effect transportability (Rosenman et al. 2025).

    Attributes
    ----------
    transported_ate : float
        Transported average treatment effect in target population.
    se : float
        Standard error of the transported estimate.
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    weight_diagnostics : dict
        Diagnostics about the importance weights (max, mean, effective_n).
    """

    transported_ate: float
    se: float
    ci_lower: float
    ci_upper: float
    weight_diagnostics: dict

    def __repr__(self) -> str:
        w = 36
        lines = [
            "TransportResult",
            _line(w),
            f"  {'transported_ate':<20}{_fmt(self.transported_ate)}",
            f"  {'se':<20}{_fmt(self.se)}",
            f"  {'ci':<20}[{_fmt(self.ci_lower)}, {_fmt(self.ci_upper)}]",
            _line(w),
        ]
        return "\n".join(lines)


# ─── TMLEResult ────────────────────────────────────────────────────


@dataclass(frozen=True)
class TMLEResult(_DictMixin):
    """Result of Targeted Maximum Likelihood Estimation.

    Attributes
    ----------
    ate : float
        Average treatment effect estimate.
    se : float
        Standard error of the ATE estimate.
    pvalue : float
        Two-sided p-value.
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    initial_estimate : float
        Initial (plug-in) ATE estimate before targeting.
    targeted_estimate : float
        ATE estimate after the targeting step.
    """

    ate: float
    se: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    initial_estimate: float
    targeted_estimate: float

    def __repr__(self) -> str:
        w = 36
        lines = [
            "TMLEResult",
            _line(w),
            f"  {'ate':<20}{_fmt(self.ate)}",
            f"  {'se':<20}{_fmt(self.se)}",
            f"  {'pvalue':<20}{_fmt(self.pvalue)}",
            f"  {'ci':<20}[{_fmt(self.ci_lower)}, {_fmt(self.ci_upper)}]",
            f"  {'initial_estimate':<20}{_fmt(self.initial_estimate)}",
            f"  {'targeted_estimate':<20}{_fmt(self.targeted_estimate)}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── DilutionResult ───────────────────────────────────────────────


@dataclass(frozen=True)
class DilutionResult(_DictMixin):
    """Result of dilution analysis (Deng et al. 2015).

    Attributes
    ----------
    diluted_ate : float
        Intent-to-treat effect (diluted back to full population).
    diluted_se : float
        Standard error of the diluted estimate.
    diluted_pvalue : float
        Two-sided p-value for the diluted estimate.
    trigger_rate : float
        Fraction of users who were triggered.
    triggered_ate : float
        Treatment effect among triggered users.
    """

    diluted_ate: float
    diluted_se: float
    diluted_pvalue: float
    trigger_rate: float
    triggered_ate: float

    def __repr__(self) -> str:
        w = 36
        lines = [
            "DilutionResult",
            _line(w),
            f"  {'diluted_ate':<20}{_fmt(self.diluted_ate)}",
            f"  {'diluted_se':<20}{_fmt(self.diluted_se)}",
            f"  {'diluted_pvalue':<20}{_fmt(self.diluted_pvalue)}",
            f"  {'trigger_rate':<20}{_fmt(self.trigger_rate, as_pct=True)}",
            f"  {'triggered_ate':<20}{_fmt(self.triggered_ate)}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── PHackingResult ───────────────────────────────────────────────


@dataclass(frozen=True)
class PHackingResult(_DictMixin):
    """Result of p-hacking detection via p-curve (Simonsohn et al. 2014).

    Attributes
    ----------
    suspicious : bool
        True if the p-value distribution is suspicious.
    p_curve_test_pvalue : float
        p-value of the right-skewness test on significant p-values.
    bunching_near_05 : bool
        True if there is excess bunching of p-values just below 0.05.
    n_experiments : int
        Number of experiments (p-values) analysed.
    message : str
        Human-readable summary of the finding.
    """

    suspicious: bool
    p_curve_test_pvalue: float
    bunching_near_05: bool
    n_experiments: int
    message: str

    def __repr__(self) -> str:
        w = 36
        lines = [
            "PHackingResult",
            _line(w),
            f"  {'suspicious':<20}{_fmt(self.suspicious)}",
            f"  {'p_curve_test_pvalue':<20}{_fmt(self.p_curve_test_pvalue)}",
            f"  {'bunching_near_05':<20}{_fmt(self.bunching_near_05)}",
            f"  {'n_experiments':<20}{self.n_experiments}",
            _line(w),
            f"  {self.message}",
        ]
        return "\n".join(lines)


# ─── DynamicResult ──────────────────────────────────────────────────


@dataclass(frozen=True)
class DynamicResult(_DictMixin):
    """Result of dynamic causal effect estimation (Shi, Deng et al. JASA 2022).

    Attributes
    ----------
    effects_over_time : list[dict]
        Per-period effect estimates with keys ``period``, ``effect``,
        ``se``, and ``pvalue``.
    cumulative_effect : float
        Cumulative treatment effect across all periods.
    pvalue : float
        Overall p-value for the null of no cumulative effect.
    trend : str
        One of ``'stable'``, ``'increasing'``, or ``'decreasing'``.
    """

    effects_over_time: list[dict]
    cumulative_effect: float
    pvalue: float
    trend: str

    def __repr__(self) -> str:
        w = 36
        lines = [
            "DynamicResult",
            _line(w),
            f"  {'n_periods':<20}{len(self.effects_over_time)}",
            f"  {'cumulative_effect':<20}{_fmt(self.cumulative_effect)}",
            f"  {'pvalue':<20}{_fmt(self.pvalue)}",
            f"  {'trend':<20}{self.trend}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── BayesOptResult ─────────────────────────────────────────────────


@dataclass(frozen=True)
class BayesOptResult(_DictMixin):
    """Result of Bayesian experiment optimization (Meta 2025).

    Attributes
    ----------
    best_params : dict
        Treatment parameters with the best predicted long-term outcome.
    predicted_long_term : float
        Predicted long-term outcome for the best parameters.
    n_experiments : int
        Number of experiments run so far.
    surrogate_r2 : float
        R-squared of the surrogate model mapping short-term to long-term.
    """

    best_params: dict
    predicted_long_term: float
    n_experiments: int
    surrogate_r2: float

    def __repr__(self) -> str:
        w = 36
        lines = [
            "BayesOptResult",
            _line(w),
            f"  {'best_params':<20}{self.best_params}",
            f"  {'predicted_long':<20}{_fmt(self.predicted_long_term)}",
            f"  {'n_experiments':<20}{self.n_experiments}",
            f"  {'surrogate_r2':<20}{_fmt(self.surrogate_r2)}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── EnrichmentResult ───────────────────────────────────────────────


@dataclass(frozen=True)
class EnrichmentResult(_DictMixin):
    """Result of adaptive enrichment analysis (Simon & Simon 2013).

    Attributes
    ----------
    selected_subgroups : list[str]
        Subgroups selected to continue enrollment.
    dropped_subgroups : list[str]
        Subgroups dropped due to lack of treatment effect.
    enrichment_ratios : dict[str, float]
        Ratio of treatment effect to SE for each subgroup.
    stage : int
        Current stage of the adaptive enrichment design.
    """

    selected_subgroups: list[str]
    dropped_subgroups: list[str]
    enrichment_ratios: dict[str, float]
    stage: int

    def __repr__(self) -> str:
        w = 36
        lines = [
            "EnrichmentResult",
            _line(w),
            f"  {'stage':<20}{self.stage}",
            f"  {'selected':<20}{self.selected_subgroups}",
            f"  {'dropped':<20}{self.dropped_subgroups}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── MarketplaceResult ──────────────────────────────────────────────


@dataclass(frozen=True)
class MarketplaceResult(_DictMixin):
    """Result of a marketplace experiment analysis (Bajari et al. 2023).

    Attributes
    ----------
    ate : float
        Average treatment effect.
    se : float
        Standard error of the ATE estimate.
    pvalue : float
        Two-sided p-value for H0: ATE = 0.
    estimated_bias : float
        Estimated bias from marketplace interference.
    design_effect : float
        Variance inflation factor due to clustering.
    recommended_side : str
        Recommended randomization side (``'buyer'`` or ``'seller'``).
    """

    ate: float
    se: float
    pvalue: float
    estimated_bias: float
    design_effect: float
    recommended_side: str

    def __repr__(self) -> str:
        w = 36
        lines = [
            "MarketplaceResult",
            _line(w),
            f"  {'ate':<20}{_fmt(self.ate)}",
            f"  {'se':<20}{_fmt(self.se)}",
            f"  {'pvalue':<20}{_fmt(self.pvalue)}",
            f"  {'estimated_bias':<20}{_fmt(self.estimated_bias)}",
            f"  {'design_effect':<20}{_fmt(self.design_effect)}",
            f"  {'recommended_side':<20}{self.recommended_side}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── BudgetSplitResult ──────────────────────────────────────────────


@dataclass(frozen=True)
class BudgetSplitResult(_DictMixin):
    """Result of a budget-split experiment (Liu et al. KDD 2021).

    Attributes
    ----------
    ate : float
        Average treatment effect.
    pvalue : float
        Two-sided p-value for H0: ATE = 0.
    significant : bool
        Whether ``pvalue < alpha``.
    treatment_budget : float
        Total budget allocated to the treatment market.
    control_budget : float
        Total budget allocated to the control market.
    cannibalization_free : bool
        Whether the design eliminates cannibalization bias.
    """

    ate: float
    pvalue: float
    significant: bool
    treatment_budget: float
    control_budget: float
    cannibalization_free: bool

    def __repr__(self) -> str:
        w = 36
        lines = [
            "BudgetSplitResult",
            _line(w),
            f"  {'ate':<22}{_fmt(self.ate)}",
            f"  {'pvalue':<22}{_fmt(self.pvalue)}",
            f"  {'significant':<22}{_fmt(self.significant)}",
            f"  {'treatment_budget':<22}{_fmt(self.treatment_budget)}",
            f"  {'control_budget':<22}{_fmt(self.control_budget)}",
            f"  {'cannibalization_free':<22}{_fmt(self.cannibalization_free)}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── BipartiteResult ────────────────────────────────────────────────


@dataclass(frozen=True)
class BipartiteResult(_DictMixin):
    """Result of a bipartite experiment analysis (Harshaw et al. 2023).

    Attributes
    ----------
    buyer_side_effect : float
        Estimated treatment effect on buyer outcomes.
    seller_side_effect : float
        Estimated treatment effect on seller outcomes (via exposure mapping).
    cross_side_pvalue : float
        p-value for the seller-side (cross-side) treatment effect.
    n_exposed_sellers : int
        Number of sellers classified as exposed to treatment.
    """

    buyer_side_effect: float
    seller_side_effect: float
    cross_side_pvalue: float
    n_exposed_sellers: int

    def __repr__(self) -> str:
        w = 36
        lines = [
            "BipartiteResult",
            _line(w),
            f"  {'buyer_side_effect':<22}{_fmt(self.buyer_side_effect)}",
            f"  {'seller_side_effect':<22}{_fmt(self.seller_side_effect)}",
            f"  {'cross_side_pvalue':<22}{_fmt(self.cross_side_pvalue)}",
            f"  {'n_exposed_sellers':<22}{self.n_exposed_sellers}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── SurvivalResult ──────────────────────────────────────────────


@dataclass(frozen=True)
class SurvivalResult(_DictMixin):
    """Result of a survival / time-to-event experiment.

    Attributes
    ----------
    hazard_ratio : float
        Estimated hazard ratio (treatment / control).
    logrank_pvalue : float
        p-value from the log-rank test.
    significant : bool
        Whether ``logrank_pvalue < alpha``.
    median_survival_ctrl : float | None
        Median survival time for the control group, or None if not reached.
    median_survival_trt : float | None
        Median survival time for the treatment group, or None if not reached.
    ci_lower : float
        Lower bound of the 95% CI for the hazard ratio.
    ci_upper : float
        Upper bound of the 95% CI for the hazard ratio.
    alpha : float
        Significance level used.
    n_ctrl : int
        Number of subjects in the control group.
    n_trt : int
        Number of subjects in the treatment group.
    n_events_ctrl : int
        Number of events observed in the control group.
    n_events_trt : int
        Number of events observed in the treatment group.
    """

    hazard_ratio: float
    logrank_pvalue: float
    significant: bool
    median_survival_ctrl: float | None
    median_survival_trt: float | None
    ci_lower: float
    ci_upper: float
    alpha: float
    n_ctrl: int
    n_trt: int
    n_events_ctrl: int
    n_events_trt: int

    def __repr__(self) -> str:
        w = 40
        lines = [
            "SurvivalResult",
            _line(w),
            f"  {'hazard_ratio':<22}{_fmt(self.hazard_ratio)}",
            f"  {'logrank_pvalue':<22}{_fmt(self.logrank_pvalue)}",
            f"  {'significant':<22}{_fmt(self.significant)}",
            f"  {'median_ctrl':<22}{self.median_survival_ctrl}",
            f"  {'median_trt':<22}{self.median_survival_trt}",
            f"  {'ci':<22}[{_fmt(self.ci_lower)}, {_fmt(self.ci_upper)}]",
            _line(w),
            f"  {'n_ctrl':<22}{self.n_ctrl}",
            f"  {'n_trt':<22}{self.n_trt}",
            f"  {'events_ctrl':<22}{self.n_events_ctrl}",
            f"  {'events_trt':<22}{self.n_events_trt}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── MediationResult ─────────────────────────────────────────────


@dataclass(frozen=True)
class MediationResult(_DictMixin):
    """Result of a causal mediation analysis.

    Attributes
    ----------
    total_effect : float
        Total effect of treatment on outcome (c path).
    direct_effect : float
        Direct effect of treatment on outcome controlling for mediator (c').
    indirect_effect : float
        Indirect effect (ACME = a * b).
    proportion_mediated : float
        Fraction of total effect explained by the mediator.
    acme_pvalue : float
        p-value for the indirect (ACME) effect via Sobel test.
    acme_ci : tuple[float, float]
        Confidence interval for the ACME.
    a_path : float
        Coefficient of treatment on mediator.
    b_path : float
        Coefficient of mediator on outcome (controlling for treatment).
    n : int
        Number of observations.
    """

    total_effect: float
    direct_effect: float
    indirect_effect: float
    proportion_mediated: float
    acme_pvalue: float
    acme_ci: tuple[float, float]
    a_path: float
    b_path: float
    n: int

    def __repr__(self) -> str:
        w = 40
        lines = [
            "MediationResult",
            _line(w),
            f"  {'total_effect':<22}{_fmt(self.total_effect)}",
            f"  {'direct_effect':<22}{_fmt(self.direct_effect)}",
            f"  {'indirect_effect':<22}{_fmt(self.indirect_effect)}",
            f"  {'proportion_mediated':<22}{_fmt(self.proportion_mediated, as_pct=True)}",
            _line(w),
            f"  {'acme_pvalue':<22}{_fmt(self.acme_pvalue)}",
            f"  {'acme_ci':<22}[{_fmt(self.acme_ci[0])}, {_fmt(self.acme_ci[1])}]",
            f"  {'a_path':<22}{_fmt(self.a_path)}",
            f"  {'b_path':<22}{_fmt(self.b_path)}",
            f"  {'n':<22}{self.n}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── MixedEffectsResult ──────────────────────────────────────────


@dataclass(frozen=True)
class MixedEffectsResult(_DictMixin):
    """Result of a mixed-effects experiment with repeated measures.

    Attributes
    ----------
    ate : float
        Average treatment effect (fixed effect for treatment).
    se : float
        Standard error of the treatment effect.
    pvalue : float
        p-value for the treatment effect.
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    significant : bool
        Whether ``pvalue < alpha``.
    icc : float
        Intraclass correlation coefficient.
    n_users : int
        Number of unique users/clusters.
    n_observations : int
        Total number of observations.
    alpha : float
        Significance level used.
    """

    ate: float
    se: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    significant: bool
    icc: float
    n_users: int
    n_observations: int
    alpha: float

    def __repr__(self) -> str:
        w = 40
        lines = [
            "MixedEffectsResult",
            _line(w),
            f"  {'ate':<22}{_fmt(self.ate)}",
            f"  {'se':<22}{_fmt(self.se)}",
            f"  {'pvalue':<22}{_fmt(self.pvalue)}",
            f"  {'ci':<22}[{_fmt(self.ci_lower)}, {_fmt(self.ci_upper)}]",
            f"  {'significant':<22}{_fmt(self.significant)}",
            _line(w),
            f"  {'icc':<22}{_fmt(self.icc)}",
            f"  {'n_users':<22}{self.n_users}",
            f"  {'n_observations':<22}{self.n_observations}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── FactorialResult ─────────────────────────────────────────────


@dataclass(frozen=True)
class FactorialResult(_DictMixin):
    """Result of a fractional factorial experiment analysis.

    Attributes
    ----------
    main_effects : dict[str, float]
        Estimated main effect for each factor.
    interactions : dict[str, float]
        Estimated two-factor interaction effects.
    significant_factors : list[str]
        Factors with statistically significant main effects.
    effect_sizes : dict[str, float]
        Standardised effect sizes for each factor.
    n_runs : int
        Number of experimental runs in the design.
    n_factors : int
        Number of factors.
    resolution : int
        Design resolution.
    """

    main_effects: dict
    interactions: dict
    significant_factors: list
    effect_sizes: dict
    n_runs: int
    n_factors: int
    resolution: int

    def __repr__(self) -> str:
        w = 40
        lines = [
            "FactorialResult",
            _line(w),
            f"  {'n_factors':<22}{self.n_factors}",
            f"  {'n_runs':<22}{self.n_runs}",
            f"  {'resolution':<22}{self.resolution}",
            _line(w),
            f"  significant factors: {self.significant_factors}",
        ]
        for name, eff in self.main_effects.items():
            lines.append(f"  {name:<22}{_fmt(eff)}")
        lines.append(_line(w))
        return "\n".join(lines)


# ─── DoseResponseResult ──────────────────────────────────────────


@dataclass(frozen=True)
class DoseResponseResult(_DictMixin):
    """Result of a continuous treatment effect (dose-response) analysis.

    Attributes
    ----------
    dose_response_curve : list[tuple[float, float]]
        List of (dose, estimated_effect) tuples along the dose range.
    optimal_dose : float
        Dose level that maximises the estimated effect.
    slope_at_mean : float
        Estimated slope of the dose-response curve at the mean dose.
    r_squared : float
        R-squared of the fitted model.
    n : int
        Number of observations.
    """

    dose_response_curve: list
    optimal_dose: float
    slope_at_mean: float
    r_squared: float
    n: int

    def __repr__(self) -> str:
        w = 40
        lines = [
            "DoseResponseResult",
            _line(w),
            f"  {'optimal_dose':<22}{_fmt(self.optimal_dose)}",
            f"  {'slope_at_mean':<22}{_fmt(self.slope_at_mean)}",
            f"  {'r_squared':<22}{_fmt(self.r_squared)}",
            f"  {'n':<22}{self.n}",
            f"  {'curve_points':<22}{len(self.dose_response_curve)}",
            _line(w),
        ]
        return "\n".join(lines)


# ─── OfflineResult ────────────────────────────────────────────────


@dataclass(frozen=True)
class OfflineResult(_DictMixin):
    """Result of an offline policy evaluation.

    Attributes
    ----------
    estimated_value : float
        Estimated value of the target policy.
    se : float
        Standard error of the estimate.
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    effective_sample_size : float
        Effective sample size after importance weighting.
    method : str
        Estimation method used (``'ips'`` or ``'doubly_robust'``).
    n : int
        Number of logged observations used.
    """

    estimated_value: float
    se: float
    ci_lower: float
    ci_upper: float
    effective_sample_size: float
    method: str
    n: int

    def __repr__(self) -> str:
        w = 40
        lines = [
            "OfflineResult",
            _line(w),
            f"  {'estimated_value':<22}{_fmt(self.estimated_value)}",
            f"  {'se':<22}{_fmt(self.se)}",
            f"  {'ci':<22}[{_fmt(self.ci_lower)}, {_fmt(self.ci_upper)}]",
            f"  {'ess':<22}{_fmt(self.effective_sample_size)}",
            f"  {'method':<22}{self.method}",
            f"  {'n':<22}{self.n}",
            _line(w),
        ]
        return "\n".join(lines)
