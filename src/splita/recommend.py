"""Recommend which splita methods to use for an experiment.

Provides step-by-step reasoning about test selection, variance reduction,
corrections, and sequential monitoring — with a copy-pasteable code example.

Examples
--------
>>> import numpy as np
>>> rng = np.random.default_rng(42)
>>> data = rng.binomial(1, 0.10, 1000)
>>> result = recommend(data)
>>> result.recommended_test
'Z-test (standard for conversion metrics)'
"""

from __future__ import annotations

import numpy as np

from splita._types import RecommendationResult
from splita._validation import check_array_like

ArrayLike = list | tuple | np.ndarray


def recommend(
    control: ArrayLike,
    treatment: ArrayLike | None = None,
    *,
    metric: str | None = None,
    has_pre_data: bool = False,
    has_clusters: bool = False,
    is_sequential: bool = False,
    n_metrics: int = 1,
) -> RecommendationResult:
    """Recommend which splita methods to use for your experiment.

    Answers: "What test should I run? What variance reduction? What corrections?"

    Parameters
    ----------
    control : array-like
        Control group data (or just sample data to analyze).
    treatment : array-like or None, default None
        Treatment group data (optional — for pre-experiment planning).
    metric : str or None, default None
        Known metric type (``"conversion"``, ``"continuous"``, ``"ratio"``),
        or ``None`` for auto-detection.
    has_pre_data : bool, default False
        Whether pre-experiment data is available for variance reduction.
    has_clusters : bool, default False
        Whether data has cluster structure (users with multiple observations).
    is_sequential : bool, default False
        Whether you will peek at results over time (sequential testing).
    n_metrics : int, default 1
        Number of metrics being tested simultaneously.

    Returns
    -------
    RecommendationResult
        Recommendations with reasoning chain and code example.
    """
    reasoning: list[str] = []
    warnings: list[str] = []

    ctrl = check_array_like(control, "control", min_length=2)
    n_ctrl = len(ctrl)

    if treatment is not None:
        trt = check_array_like(treatment, "treatment", min_length=2)
        n_trt = len(trt)
        combined = np.concatenate([ctrl, trt])
        total_n = n_ctrl + n_trt
    else:
        combined = ctrl
        total_n = n_ctrl

    # ── Metric Detection ──────────────────────────────────────────
    if metric is not None:
        detected_metric = metric
        reasoning.append(f"Metric type provided: {metric}")
    else:
        from splita._utils import auto_detect_metric

        detected_metric = auto_detect_metric(combined)
        if detected_metric == "conversion":
            reasoning.append(
                "Detected metric type: conversion (all values are 0 or 1)"
            )
        else:
            reasoning.append(
                f"Detected metric type: continuous "
                f"(values range from {float(np.min(combined)):.1f} "
                f"to {float(np.max(combined)):.1f})"
            )

    # ── Sample Size Assessment ────────────────────────────────────
    is_small_n = total_n < 100
    if is_small_n:
        reasoning.append(
            f"Small sample size detected (n={total_n}). "
            f"Non-parametric or bootstrap methods recommended."
        )
        warnings.append(
            f"Sample size is very small (n={total_n}). "
            f"Power may be insufficient to detect realistic effects."
        )

    # ── Skewness Assessment (continuous only) ─────────────────────
    is_skewed = False
    if detected_metric == "continuous":
        from scipy.stats import skew

        skewness = float(skew(combined))
        if abs(skewness) > 1.0:
            is_skewed = True
            reasoning.append(
                f"Data is heavily skewed (skewness={skewness:.2f}). "
                f"Non-parametric tests may be more appropriate."
            )
        else:
            reasoning.append(
                f"Data is approximately symmetric (skewness={skewness:.2f}). "
                f"Parametric tests are appropriate."
            )

    # ── Test Selection ────────────────────────────────────────────
    if has_clusters:
        recommended_test = "ClusterExperiment (accounts for within-cluster correlation)"
        reasoning.append(
            "Data has cluster structure — standard tests would underestimate "
            "variance and inflate false positives. ClusterExperiment handles this."
        )
    elif detected_metric == "conversion":
        if is_small_n:
            recommended_test = "PermutationTest (exact test for small binary samples)"
            reasoning.append(
                "Binary data with small sample — permutation test provides exact "
                "p-values without distributional assumptions."
            )
        else:
            recommended_test = "Z-test (standard for conversion metrics)"
            reasoning.append(
                "Binary data with sufficient sample size — Z-test is the standard "
                "and most powerful option."
            )
    elif detected_metric == "ratio":
        recommended_test = "Delta method (standard for ratio metrics)"
        reasoning.append(
            "Ratio metric detected — delta method correctly handles the "
            "variance of ratios."
        )
    elif is_small_n:
        recommended_test = (
            "PermutationTest or bootstrap (small sample, no distributional assumptions)"
        )
        reasoning.append(
            "Small sample — permutation test or bootstrap avoid normality "
            "assumptions that may not hold."
        )
    elif is_skewed:
        recommended_test = "Mann-Whitney U or bootstrap (data is heavily skewed)"
        reasoning.append(
            "Heavily skewed continuous data — Mann-Whitney or bootstrap are robust "
            "to non-normal distributions."
        )
    else:
        recommended_test = "Welch's t-test (standard for continuous metrics)"
        reasoning.append(
            "Normal continuous data — Welch's t-test is the standard, most powerful "
            "test and does not assume equal variances."
        )

    # ── Variance Reduction ────────────────────────────────────────
    variance_parts: list[str] = []

    if detected_metric == "continuous" and not is_small_n:
        variance_parts.append("OutlierHandler")
        reasoning.append(
            "OutlierHandler recommended for continuous data — "
            "caps extreme values that inflate variance."
        )

    if has_pre_data:
        variance_parts.append("CUPED")
        reasoning.append(
            "CUPED recommended — pre-experiment data enables variance reduction "
            "of 20-65%, equivalent to running the experiment 1.5-3x longer."
        )

    recommended_variance = " + ".join(variance_parts) if variance_parts else None
    if recommended_variance is None and detected_metric == "continuous":
        reasoning.append(
            "No pre-data available for CUPED. Consider collecting "
            "pre-experiment data to reduce variance."
        )

    # ── Multiple Testing Correction ───────────────────────────────
    if n_metrics > 1:
        recommended_correction = "Benjamini-Hochberg (BH)"
        reasoning.append(
            f"Testing {n_metrics} metrics simultaneously — "
            f"Benjamini-Hochberg correction controls the false discovery rate."
        )
        warnings.append(
            f"Without correction, testing {n_metrics} metrics at alpha=0.05 "
            f"gives a ~{min(100, n_metrics * 5)}% chance of at least one false positive."
        )
    else:
        recommended_correction = None
        reasoning.append("Single metric — no multiple testing correction needed.")

    # ── Sequential Testing ────────────────────────────────────────
    if is_sequential:
        recommended_sequential = "mSPRT (always-valid sequential monitoring)"
        reasoning.append(
            "Sequential monitoring requested — mSPRT provides always-valid "
            "p-values that you can check at any time without inflating "
            "false positive rate."
        )
        warnings.append(
            "Do not use fixed-horizon tests for sequential monitoring — "
            "they inflate the false positive rate with repeated peeking."
        )
    else:
        recommended_sequential = None

    # ── Generate Code Example ─────────────────────────────────────
    code_example = _build_code_example(
        recommended_test=recommended_test,
        recommended_variance=recommended_variance,
        recommended_correction=recommended_correction,
        recommended_sequential=recommended_sequential,
        has_pre_data=has_pre_data,
        has_clusters=has_clusters,
        is_sequential=is_sequential,
        n_metrics=n_metrics,
    )

    return RecommendationResult(
        recommended_test=recommended_test,
        recommended_variance=recommended_variance,
        recommended_correction=recommended_correction,
        recommended_sequential=recommended_sequential,
        reasoning=reasoning,
        code_example=code_example,
        warnings=warnings,
    )


def _build_code_example(
    *,
    recommended_test: str,
    recommended_variance: str | None,
    recommended_correction: str | None,
    recommended_sequential: str | None,
    has_pre_data: bool,
    has_clusters: bool,
    is_sequential: bool,
    n_metrics: int,
) -> str:
    """Build a copy-pasteable Python code example."""
    imports: list[str] = ["from splita import Experiment"]
    steps: list[str] = []
    step_num = 1

    # Variance reduction imports and steps
    if recommended_variance and "OutlierHandler" in recommended_variance:
        imports.append("from splita import OutlierHandler")
        steps.append(f"# Step {step_num}: Handle outliers")
        steps.append("handler = OutlierHandler()")
        steps.append("ctrl, trt = handler.fit_transform(control, treatment)")
        step_num += 1

    if recommended_variance and "CUPED" in recommended_variance:
        imports.append("from splita import CUPED")
        steps.append(f"# Step {step_num}: CUPED variance reduction")
        steps.append("cuped = CUPED()")
        steps.append("ctrl, trt = cuped.fit_transform(ctrl, trt, ctrl_pre, trt_pre)")
        steps.append('print(f"Variance reduced by {cuped.variance_reduction_:.1%}")')
        step_num += 1

    # Test selection
    if has_clusters:
        imports[0] = "from splita import ClusterExperiment"
        steps.append(f"# Step {step_num}: Run cluster-aware test")
        steps.append(
            "result = ClusterExperiment(ctrl, trt, cluster_ids_ctrl, cluster_ids_trt).run()"
        )
    elif is_sequential:
        imports.append("from splita import mSPRT")
        steps.append(f"# Step {step_num}: Set up sequential monitoring")
        steps.append("monitor = mSPRT(alpha=0.05)")
        steps.append("for batch in data_batches:")
        steps.append("    monitor.update(batch)")
        steps.append("    if monitor.result().reject:")
        steps.append('        print("Significant! Can stop early.")')
        steps.append("        break")
    else:
        data_var = "ctrl" if recommended_variance else "control"
        trt_var = "trt" if recommended_variance else "treatment"
        steps.append(f"# Step {step_num}: Run the test")
        steps.append(f"result = Experiment({data_var}, {trt_var}).run()")
    step_num += 1

    # Correction
    if recommended_correction and n_metrics > 1:
        imports.append("from splita import MultipleCorrection")
        steps.append(f"# Step {step_num}: Correct for multiple comparisons")
        steps.append("pvalues = [result.pvalue, result2.pvalue]  # all metric p-values")
        steps.append('correction = MultipleCorrection(pvalues, method="bh").run()')
        steps.append("print(correction.adjusted_pvalues)")
        step_num += 1

    # Print result
    steps.append("")
    steps.append("print(result)")

    lines = "\n".join(imports) + "\n\n" + "\n".join(steps)
    return lines
