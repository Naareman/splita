"""Structured next-steps diagnosis for experiment results.

Different from :func:`~splita.explain` which gives a narrative
interpretation -- :func:`diagnose` gives numbered action items and
a structured status assessment.

Examples
--------
>>> from splita._types import ExperimentResult
>>> r = ExperimentResult(
...     control_mean=0.10, treatment_mean=0.12, lift=0.02,
...     relative_lift=0.20, pvalue=0.003, statistic=2.97,
...     ci_lower=0.007, ci_upper=0.033, significant=True,
...     alpha=0.05, method="ztest", metric="conversion",
...     control_n=5000, treatment_n=5000, power=0.82,
...     effect_size=0.15,
... )
>>> result = diagnose(r)
>>> result.status
'healthy'
"""

from __future__ import annotations

from splita._types import DiagnosisResult, ExperimentResult


def diagnose(result: ExperimentResult) -> DiagnosisResult:
    """Get a structured actionable checklist for any experiment result.

    Analyses the experiment result and produces a status assessment,
    numbered action items, next steps, and a confidence level rating.

    Parameters
    ----------
    result : ExperimentResult
        An experiment result to diagnose.

    Returns
    -------
    DiagnosisResult
        Structured diagnosis with status, action items, next steps,
        and confidence level.

    Raises
    ------
    TypeError
        If result is not an ExperimentResult.
    """
    if not isinstance(result, ExperimentResult):
        raise TypeError(f"`result` must be an ExperimentResult, got {type(result).__name__}.")

    action_items: list[str] = []
    next_steps: list[str] = []
    issues: list[str] = []

    # ── Power Assessment ─────────────────────────────────────────
    if result.power < 0.5:
        issues.append("critically_underpowered")
        action_items.append(
            f"Experiment is severely underpowered (power={result.power:.1%}). "
            "Do not make decisions based on this result."
        )
        action_items.append(
            "Increase sample size or apply variance reduction (CUPED) before re-running."
        )
    elif result.power < 0.8:
        issues.append("underpowered")
        action_items.append(
            f"Experiment is underpowered (power={result.power:.1%}). "
            "A non-significant result does not mean no effect."
        )

    # ── Sample Size Assessment ───────────────────────────────────
    n_min = min(result.control_n, result.treatment_n)
    if n_min < 100:
        issues.append("small_sample")
        action_items.append(
            f"Very small sample size (n={n_min}). Results may be unreliable. "
            "Consider running longer."
        )
    elif n_min < 500:
        issues.append("moderate_sample")
        action_items.append(
            f"Moderate sample size (n={n_min}). Adequate for large effects "
            "but may miss smaller ones."
        )

    # ── Effect Size Assessment ───────────────────────────────────
    abs_effect = abs(result.effect_size)
    if result.significant:
        if abs_effect < 0.2:
            action_items.append(
                "Statistically significant but the effect size is negligible "
                f"(d={result.effect_size:.3f}). Consider whether this is "
                "practically meaningful."
            )
        elif abs_effect >= 0.8:
            action_items.append(
                f"Large effect size (d={result.effect_size:.3f}). Verify the "
                "result is not due to a bug or data quality issue."
            )

    # ── CI Width Assessment ──────────────────────────────────────
    ci_width = result.ci_upper - result.ci_lower
    if result.lift != 0 and ci_width > abs(result.lift) * 4:
        issues.append("wide_ci")
        action_items.append(
            "Confidence interval is very wide relative to the observed effect. "
            "The true effect could plausibly be zero or opposite in sign."
        )

    # ── Significance Assessment ──────────────────────────────────
    if result.significant:
        if result.pvalue > 0.01:
            action_items.append(
                f"Marginally significant (p={result.pvalue:.4f}). Consider "
                "replicating before shipping."
            )
    else:
        if result.pvalue < 0.10:
            action_items.append(
                f"Close to significant (p={result.pvalue:.4f}). Consider "
                "running longer before concluding no effect."
            )

    # ── Sample Ratio Check Reminder ──────────────────────────────
    ratio = result.control_n / result.treatment_n if result.treatment_n > 0 else 0
    if abs(ratio - 1.0) > 0.1:
        issues.append("unbalanced")
        action_items.append(
            f"Sample sizes are imbalanced (control={result.control_n}, "
            f"treatment={result.treatment_n}). Check for SRM."
        )

    # ── Determine Status ─────────────────────────────────────────
    if "critically_underpowered" in issues or "small_sample" in issues:
        status = "critical"
    elif issues:
        status = "warning"
    else:
        status = "healthy"

    # ── Confidence Level ─────────────────────────────────────────
    if result.power >= 0.8 and n_min >= 1000 and not issues:
        confidence_level = "high"
    elif result.power >= 0.5 and n_min >= 200:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    # ── Next Steps ───────────────────────────────────────────────
    if result.significant and status == "healthy":
        next_steps.append("Result is significant and healthy. Consider shipping.")
        next_steps.append("Run SRM check (splita.SRMCheck) if not already done.")
        next_steps.append("Check for novelty effects using splita.NoveltyCurve.")
    elif result.significant and status != "healthy":
        next_steps.append("Result is significant but has issues. Address action items first.")
        next_steps.append("Consider replicating the experiment before making decisions.")
    else:
        next_steps.append("Result is not significant. This does not prove no effect exists.")
        if result.power < 0.8:
            next_steps.append("Increase sample size or use CUPED to improve power.")
        next_steps.append(
            "Consider whether the MDE was realistic for your sample size. "
            "Use splita.SampleSize to plan the next experiment."
        )

    # Default action item if none
    if not action_items:
        action_items.append("No issues detected. The experiment result appears reliable.")

    return DiagnosisResult(
        status=status,
        action_items=action_items,
        next_steps=next_steps,
        confidence_level=confidence_level,
    )
