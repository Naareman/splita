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
    except ImportError:
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
        result: dict[str, Any] = {}
        for f in fields(self):  # type: ignore[arg-type]
            result[f.name] = _to_python(getattr(self, f.name))
        return result


# ─── ExperimentResult ────────────────────────────────────────────────

@dataclass(frozen=True)
class ExperimentResult(_DictMixin):
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
            lines.append(f"  {'relative_mde':<16}{_fmt(self.relative_mde, as_pct=True)}")
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
            worst_dev = f"{self.deviations_pct[self.worst_variant]:+.1f}%"
            lines += [
                f"  \u26a0 Sample ratio mismatch detected!",
                f"  Variant {self.worst_variant} has the largest deviation ({worst_dev}).",
                f"  Experiment results cannot be trusted.",
            ]
        else:
            lines.append(f"  {self.message}")
        return "\n".join(lines)


# ─── CorrectionResult ───────────────────────────────────────────────

@dataclass(frozen=True)
class CorrectionResult(_DictMixin):
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
            f"  {'ci':<24}[{_fmt(self.always_valid_ci_lower)}, {_fmt(self.always_valid_ci_upper)}]",
            f"  {'should_stop':<24}{_fmt(self.should_stop)}",
            f"  {'effect_estimate':<24}{_fmt(self.current_effect_estimate)}",
        ]
        return "\n".join(lines)


# ─── mSPRTResult ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class mSPRTResult(_DictMixin):
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
            f"  {'ci':<24}[{_fmt(self.always_valid_ci_lower)}, {_fmt(self.always_valid_ci_upper)}]",
            f"  {'should_stop':<24}{_fmt(self.should_stop)}",
            f"  {'effect_estimate':<24}{_fmt(self.current_effect_estimate)}",
            _line(w),
            f"  {'stopping_reason':<24}{self.stopping_reason}",
        ]
        if self.relative_speedup_vs_fixed_horizon is not None:
            lines.append(
                f"  {'speedup_vs_fixed':<24}{_fmt(self.relative_speedup_vs_fixed_horizon, as_pct=True)}"
            )
        return "\n".join(lines)


# ─── BoundaryResult ──────────────────────────────────────────────────

@dataclass(frozen=True)
class BoundaryResult(_DictMixin):
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
            row = f"  {i + 1:<6}{_fmt(self.information_fractions[i]):<12}{_fmt(self.efficacy_boundaries[i]):<12}"
            if self.futility_boundaries is not None:
                row += f"{_fmt(self.futility_boundaries[i]):<12}"
            row += f"{_fmt(self.alpha_spent[i]):<12}"
            lines.append(row)
        return "\n".join(lines)


# ─── GSResult ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GSResult(_DictMixin):
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
            ci = self.arm_credible_intervals[i]
            lines.append(
                f"  {i:<6}{self.n_pulls_per_arm[i]:<10}"
                f"{_fmt(self.arm_means[i]):<10}"
                f"{_fmt(self.prob_best[i]):<10}"
                f"{_fmt(self.expected_loss[i]):<10}"
            )
        return "\n".join(lines)
