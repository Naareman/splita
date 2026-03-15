"""Plain-English interpretation of splita result objects.

This module provides the :func:`explain` function, which converts any
splita result dataclass into a human-readable paragraph with actionable
suggestions.

Examples
--------
>>> from splita import Experiment, explain
>>> import numpy as np
>>> ctrl = np.random.binomial(1, 0.10, 5000)
>>> trt  = np.random.binomial(1, 0.12, 5000)
>>> result = Experiment(ctrl, trt).run()
>>> print(explain(result))  # doctest: +SKIP
"""

from __future__ import annotations

from typing import Any


def _effect_label(effect_size: float) -> str:
    """Classify Cohen's d / h magnitude."""
    d = abs(effect_size)
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


def _fmt_pct(val: float) -> str:
    return f"{val * 100:.2f}%"


def _fmt_num(val: float) -> str:
    if abs(val) < 0.0001 and val != 0.0:
        return f"{val:.2e}"
    return f"{val:.2f}"


def _explain_experiment(result: Any) -> str:
    """Interpret an ExperimentResult."""
    lines: list[str] = []

    direction = "increased" if result.lift > 0 else "decreased"
    lines.append(
        f"Your treatment {direction} the mean by {_fmt_num(result.lift)} "
        f"(95% CI: {_fmt_num(result.ci_lower)} to {_fmt_num(result.ci_upper)})."
    )

    if result.significant:
        lines.append(
            f"This is statistically significant at alpha={result.alpha} "
            f"(p={_fmt_num(result.pvalue)})."
        )
        label = _effect_label(result.effect_size)
        lines.append(
            f"The effect size (Cohen's d = {_fmt_num(result.effect_size)}) "
            f"is {label}."
        )
        lines.append(
            f"Post-hoc power is {_fmt_num(result.power)}, "
            + (
                "suggesting adequate sample size."
                if result.power >= 0.8
                else "which is below 0.80 — the experiment may be underpowered."
            )
        )
    else:
        lines.append(
            f"No statistically significant difference was detected "
            f"(p={_fmt_num(result.pvalue)})."
        )
        n = min(result.control_n, result.treatment_n)
        lines.append(
            f"With n={n} per group, the observed effect of "
            f"{_fmt_num(result.lift)} was not large enough to reach significance."
        )
        ci_width = result.ci_upper - result.ci_lower
        if ci_width > abs(result.lift) * 4 and result.lift != 0:
            lines.append(
                "The confidence interval is wide relative to the observed "
                "effect — sample size may be insufficient."
            )
        lines.append(
            "Consider running longer or using variance reduction (CUPED)."
        )

    # Suggestions
    suggestions = _experiment_suggestions(result)
    if suggestions:
        lines.append("")
        lines.append("Suggestions:")
        for s in suggestions:
            lines.append(f"  - {s}")

    return " ".join(lines[:_first_blank(lines)]) + (
        "\n" + "\n".join(lines[_first_blank(lines) :])
        if _first_blank(lines) < len(lines)
        else ""
    )


def _first_blank(lines: list[str]) -> int:
    for i, line in enumerate(lines):
        if line == "":
            return i
    return len(lines)


def _experiment_suggestions(result: Any) -> list[str]:
    suggestions: list[str] = []
    if result.power < 0.8:
        suggestions.append(
            "Experiment appears underpowered. Consider using CUPED for "
            "variance reduction or increasing sample size."
        )
    if not result.significant:
        ci_width = result.ci_upper - result.ci_lower
        if ci_width > 0 and abs(result.lift) > 0 and ci_width > abs(result.lift) * 4:
            suggestions.append(
                "Wide confidence interval relative to observed effect — "
                "consider increasing sample size."
            )
    return suggestions


def _explain_srm(result: Any) -> str:
    """Interpret an SRMResult."""
    if not result.passed:
        worst_dev = result.deviations_pct[result.worst_variant]
        return (
            f"WARNING: Sample Ratio Mismatch detected (p={_fmt_num(result.pvalue)}). "
            f"The traffic split deviates significantly from expected "
            f"(variant {result.worst_variant} is off by {worst_dev:+.1f}%). "
            f"All experiment results should be considered invalid until the "
            f"cause is identified. Common causes: randomization bugs, bot "
            f"traffic, tracking errors.\n\n"
            f"Suggestions:\n"
            f"  - Check randomization logic and hash function\n"
            f"  - Filter bot traffic and verify tracking pipeline\n"
            f"  - Compare pre-experiment vs. in-experiment traffic ratios"
        )
    return (
        f"No Sample Ratio Mismatch detected (p={_fmt_num(result.pvalue)}). "
        f"Traffic split is consistent with the expected allocation."
    )


def _explain_bayesian(result: Any) -> str:
    """Interpret a BayesianResult."""
    prob_pct = _fmt_pct(result.prob_b_beats_a)

    if result.prob_b_beats_a > 0.5:
        better = "treatment"
        loss = result.expected_loss_b
    else:
        better = "control"
        loss = result.expected_loss_a

    loss_str = _fmt_pct(loss) if abs(loss) < 1 else _fmt_num(loss)

    lines: list[str] = [
        f"There is a {prob_pct} probability that the treatment is better "
        f"than control."
    ]

    lines.append(
        f"The expected loss from choosing the {better} is {loss_str}."
    )

    if result.prob_b_beats_a >= 0.95:
        lines.append("Recommendation: ship the treatment.")
    elif result.prob_b_beats_a <= 0.05:
        lines.append("Recommendation: keep the control.")
    else:
        lines.append(
            "The evidence is not yet decisive. Consider collecting more data."
        )

    if result.prob_in_rope is not None:
        lines.append(
            f"Probability that the effect is practically negligible "
            f"(within ROPE [{_fmt_num(result.rope[0])}, "
            f"{_fmt_num(result.rope[1])}]): {_fmt_pct(result.prob_in_rope)}."
        )

    return " ".join(lines)


def _explain_sample_size(result: Any) -> str:
    """Interpret a SampleSizeResult."""
    lines: list[str] = [
        f"You need {result.n_per_variant:,} users per variant "
        f"({result.n_total:,} total) to detect a "
    ]

    if result.relative_mde is not None:
        lines[0] += (
            f"{_fmt_pct(result.relative_mde)} relative lift "
            f"(absolute MDE = {_fmt_num(result.mde)}) "
        )
    else:
        lines[0] += f"{_fmt_num(result.mde)} absolute lift "

    lines[0] += (
        f"from a {_fmt_num(result.baseline)} baseline with "
        f"{_fmt_pct(result.power)} power at alpha={result.alpha}."
    )

    if result.days_needed is not None:
        lines.append(
            f"At the given traffic rate, this will take approximately "
            f"{result.days_needed} days."
        )

    return " ".join(lines)


# ─── Registry of explainable types ──────────────────────────────────

_EXPLAINERS: dict[str, Any] = {
    "ExperimentResult": _explain_experiment,
    "SRMResult": _explain_srm,
    "BayesianResult": _explain_bayesian,
    "SampleSizeResult": _explain_sample_size,
}


def explain(result: Any) -> str:
    """Return a plain-English interpretation of any splita result.

    Parameters
    ----------
    result : dataclass
        A splita result object (e.g. ``ExperimentResult``,
        ``BayesianResult``, ``SRMResult``, ``SampleSizeResult``).

    Returns
    -------
    str
        Human-readable interpretation with actionable suggestions.

    Raises
    ------
    TypeError
        If the result type is not supported.

    Examples
    --------
    >>> from splita._types import ExperimentResult
    >>> r = ExperimentResult(
    ...     control_mean=0.10, treatment_mean=0.12,
    ...     lift=0.02, relative_lift=0.2, pvalue=0.003,
    ...     statistic=2.1, ci_lower=0.007, ci_upper=0.033,
    ...     significant=True, alpha=0.05, method="ztest",
    ...     metric="conversion", control_n=5000,
    ...     treatment_n=5000, power=0.82, effect_size=0.15,
    ... )
    >>> from splita.explain import explain
    >>> text = explain(r)
    >>> "significant" in text
    True
    """
    type_name = type(result).__name__
    explainer = _EXPLAINERS.get(type_name)
    if explainer is None:
        raise TypeError(
            f"`explain()` does not support {type_name}.\n"
            f"  Detail: supported types are {', '.join(sorted(_EXPLAINERS))}.\n"
            f"  Hint: pass an ExperimentResult, BayesianResult, SRMResult, "
            f"or SampleSizeResult."
        )
    return explainer(result)
