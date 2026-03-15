"""Zero-config complete A/B test analysis.

Automatically detects metric type, checks SRM, handles outliers,
applies CUPED if pre-data is provided, runs the right test, and
applies multiple correction if multiple metrics are given.

Examples
--------
>>> import numpy as np
>>> rng = np.random.default_rng(42)
>>> ctrl = rng.binomial(1, 0.10, 5000)
>>> trt = rng.binomial(1, 0.12, 5000)
>>> result = auto(ctrl, trt)
>>> result.primary_result.metric
'conversion'
"""

from __future__ import annotations

import numpy as np

from splita._types import AutoResult
from splita._validation import check_array_like
from splita.core.experiment import Experiment
from splita.core.srm import SRMCheck

ArrayLike = list | tuple | np.ndarray


def auto(
    control: ArrayLike,
    treatment: ArrayLike,
    *,
    control_pre: ArrayLike | None = None,
    treatment_pre: ArrayLike | None = None,
    control_denominator: ArrayLike | None = None,
    treatment_denominator: ArrayLike | None = None,
    metrics: dict | None = None,
    alpha: float = 0.05,
) -> AutoResult:
    """Complete A/B test analysis in one function call.

    Automatically: detects metric type, checks SRM, handles outliers,
    applies CUPED if pre-data provided, runs the right test, applies
    multiple correction if multiple metrics.

    Parameters
    ----------
    control : array-like
        Observations from the control group.
    treatment : array-like
        Observations from the treatment group.
    control_pre : array-like or None, default None
        Pre-experiment control data (enables CUPED variance reduction).
    treatment_pre : array-like or None, default None
        Pre-experiment treatment data.
    control_denominator : array-like or None, default None
        Denominator for ratio metrics.
    treatment_denominator : array-like or None, default None
        Denominator for ratio metrics.
    metrics : dict or None, default None
        Dictionary mapping metric names to ``(control, treatment)`` tuples
        for multi-metric analysis with multiple testing correction.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    AutoResult
        Complete analysis result with pipeline trace and reasoning chain.
    """
    pipeline_steps: list[str] = []
    recommendations: list[str] = []
    reasoning: list[str] = []

    ctrl = check_array_like(control, "control", min_length=2)
    trt = check_array_like(treatment, "treatment", min_length=2)

    # ── Step 1: Metric Detection ──────────────────────────────────
    from splita._utils import auto_detect_metric

    combined = np.concatenate([ctrl, trt])
    detected_metric = auto_detect_metric(combined)

    if detected_metric == "conversion":
        reasoning.append("Detected metric type: conversion (all values are 0 or 1)")
    else:
        min_val = float(np.min(combined))
        max_val = float(np.max(combined))
        reasoning.append(
            f"Detected metric type: continuous (values range from {min_val:.1f} to {max_val:.1f})"
        )

    # ── Step 2: SRM Check ────────────────────────────────────────
    pipeline_steps.append("1. Checked sample ratio mismatch (SRM)")
    srm_result = SRMCheck([len(ctrl), len(trt)], alpha=0.01).run()
    if srm_result.passed:
        reasoning.append(
            f"SRM check: PASSED (p={srm_result.pvalue:.3f}, traffic split is balanced)"
        )
    else:
        reasoning.append(
            f"SRM check: FAILED (p={srm_result.pvalue:.6f}). Results may not be trustworthy."
        )
        recommendations.append(
            "WARNING: Sample ratio mismatch detected. All results should "
            "be treated with extreme caution."
        )

    # ── Step 3: Outlier Detection & Handling ─────────────────────
    if detected_metric == "continuous":
        q1, q3 = float(np.percentile(combined, 1)), float(np.percentile(combined, 99))
        ctrl_clipped = np.clip(ctrl, q1, q3)
        trt_clipped = np.clip(trt, q1, q3)
        n_clipped = int(np.sum(ctrl != ctrl_clipped) + np.sum(trt != trt_clipped))
        if n_clipped > 0:
            total_n = len(ctrl) + len(trt)
            pct = 100.0 * n_clipped / total_n
            pipeline_steps.append(
                f"2. Winsorized {n_clipped} outlier values at 1st/99th percentile"
            )
            reasoning.append(
                f"Applied outlier capping at 1st/99th percentile — "
                f"{n_clipped} values capped ({pct:.1f}%)"
            )
            ctrl = ctrl_clipped
            trt = trt_clipped
        else:
            pipeline_steps.append("2. No outliers detected (skipped winsorization)")
            reasoning.append("No outliers detected — winsorization not needed")
    else:
        pipeline_steps.append("2. Skipped outlier handling (conversion metric)")
        reasoning.append("Skipped outlier handling (not applicable for conversion metrics)")

    # ── Step 4: CUPED Variance Reduction ─────────────────────────
    variance_reduction: float | None = None
    if control_pre is not None and treatment_pre is not None:
        from splita.variance.cuped import CUPED

        ctrl_pre = check_array_like(control_pre, "control_pre", min_length=2)
        trt_pre = check_array_like(treatment_pre, "treatment_pre", min_length=2)

        try:
            cuped = CUPED()
            ctrl, trt = cuped.fit_transform(ctrl, trt, ctrl_pre, trt_pre)
            variance_reduction = float(cuped.variance_reduction_)
            corr = float(cuped.correlation_)
            pipeline_steps.append(
                f"3. Applied CUPED variance reduction ({variance_reduction:.1%} "
                f"variance removed, r={corr:.3f})"
            )
            reasoning.append(
                f"Applied CUPED variance reduction "
                f"(correlation={corr:.2f}, variance reduced by {variance_reduction:.0%})"
            )
        except Exception:
            pipeline_steps.append("3. CUPED failed (insufficient correlation); skipped")
            # Try to compute correlation for reasoning
            try:
                min_len = min(len(ctrl), len(ctrl_pre))
                corr = float(np.corrcoef(ctrl[:min_len], ctrl_pre[:min_len])[0, 1])
                reasoning.append(f"CUPED skipped — correlation with pre-data too low ({corr:.2f})")
            except Exception:
                reasoning.append("CUPED skipped — insufficient correlation with pre-data")
            variance_reduction = None
    else:
        pipeline_steps.append("3. No pre-experiment data provided (skipped CUPED)")
        if control_pre is not None or treatment_pre is not None:
            reasoning.append("CUPED skipped — both control_pre and treatment_pre are required")
        else:
            reasoning.append("No pre-experiment data provided (CUPED not applied)")
        if detected_metric == "continuous":
            recommendations.append(
                "Consider providing pre-experiment data for CUPED variance "
                "reduction — it can reduce required sample size by 20-65%."
            )

    # ── Step 5: Run Experiment ───────────────────────────────────
    exp = Experiment(
        ctrl,
        trt,
        alpha=alpha,
        control_denominator=control_denominator,
        treatment_denominator=treatment_denominator,
    )
    primary_result = exp.run()
    pipeline_steps.append(
        f"4. Ran {primary_result.method} test (metric={primary_result.metric}, alpha={alpha})"
    )

    # Method selection reasoning
    method = primary_result.method
    metric = primary_result.metric
    if metric == "conversion":
        reasoning.append(f"Selected {method} test (standard for conversion/binary metrics)")
    elif metric == "ratio":
        reasoning.append(f"Selected {method} test (appropriate for ratio metrics)")
    else:
        reasoning.append(f"Selected {method} test (standard for continuous metrics)")

    # ── Step 6: Multiple Metrics Correction ──────────────────────
    corrected_pvalues: list[float] | None = None
    if metrics is not None and len(metrics) >= 1:
        from splita.core.correction import MultipleCorrection

        all_pvalues = [primary_result.pvalue]
        metric_labels = ["primary"]

        for name, (mc, mt) in metrics.items():
            mc_arr = check_array_like(mc, f"metrics[{name!r}][0]", min_length=2)
            mt_arr = check_array_like(mt, f"metrics[{name!r}][1]", min_length=2)
            r = Experiment(mc_arr, mt_arr, alpha=alpha).run()
            all_pvalues.append(r.pvalue)
            metric_labels.append(name)

        correction = MultipleCorrection(
            all_pvalues, method="bh", alpha=alpha, labels=metric_labels
        ).run()
        corrected_pvalues = correction.adjusted_pvalues
        pipeline_steps.append(
            f"5. Applied Benjamini-Hochberg correction across {len(all_pvalues)} metrics"
        )
        reasoning.append(
            f"Applied Benjamini-Hochberg correction for {len(all_pvalues)} metrics "
            f"to control false discovery rate"
        )
    else:
        pipeline_steps.append("5. Single metric (no multiple testing correction needed)")
        reasoning.append("Single metric — no multiple testing correction needed")

    # ── Result Interpretation ─────────────────────────────────────
    if primary_result.significant:
        direction = "increased" if primary_result.lift > 0 else "decreased"
        reasoning.append(
            f"Result: SIGNIFICANT (p={primary_result.pvalue:.4f}). "
            f"Treatment {direction} by {primary_result.relative_lift:+.2%}"
        )
        reasoning.append("Recommendation: SHIP — effect is statistically significant")
    else:
        reasoning.append(f"Result: NOT significant (p={primary_result.pvalue:.3f})")
        reasoning.append("Recommendation: DO NOT SHIP — insufficient evidence of an effect")
        recommendations.append(
            "Result is not statistically significant. Consider running "
            "longer or increasing sample size."
        )

    # ── Power Assessment ──────────────────────────────────────────
    if primary_result.power < 0.8:
        reasoning.append(
            f"Warning: experiment may be underpowered "
            f"(power={primary_result.power:.2f}). Consider running longer."
        )
        recommendations.append(
            f"Post-hoc power is {primary_result.power:.1%}, below 80%. "
            "The experiment may be underpowered."
        )

    return AutoResult(
        primary_result=primary_result,
        srm_result=srm_result,
        variance_reduction=variance_reduction,
        corrected_pvalues=corrected_pvalues,
        pipeline_steps=pipeline_steps,
        recommendations=recommendations,
        reasoning=reasoning,
    )
