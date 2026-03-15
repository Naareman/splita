"""Pre-analysis health report for A/B tests.

Run all pre-analysis checks on your data in one call: SRM, flicker
detection, outlier detection, power assessment, and basic data quality.

Examples
--------
>>> import numpy as np
>>> rng = np.random.default_rng(42)
>>> ctrl = rng.binomial(1, 0.10, 5000)
>>> trt = rng.binomial(1, 0.12, 5000)
>>> result = check(ctrl, trt)
>>> result.srm_passed
True
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm, skew

from splita._types import CheckResult
from splita._validation import check_array_like

ArrayLike = list | tuple | np.ndarray


def check(
    control: ArrayLike,
    treatment: ArrayLike,
    *,
    control_pre: ArrayLike | None = None,
    treatment_pre: ArrayLike | None = None,
    segments: ArrayLike | None = None,
    user_ids: ArrayLike | None = None,
    variant_assignments: ArrayLike | None = None,
    alpha: float = 0.05,
) -> CheckResult:
    """Run all pre-analysis checks on your data in one call.

    Runs: SRM, covariate balance (if segments), flicker detection
    (if user_ids), outlier detection, metric sensitivity, basic data
    quality. Returns structured health report.

    Parameters
    ----------
    control : array-like
        Observations from the control group.
    treatment : array-like
        Observations from the treatment group.
    control_pre : array-like or None, default None
        Pre-experiment control data (enables power assessment with CUPED).
    treatment_pre : array-like or None, default None
        Pre-experiment treatment data.
    segments : array-like or None, default None
        Segment labels for covariate balance check.
    user_ids : array-like or None, default None
        User IDs for flicker detection.
    variant_assignments : array-like or None, default None
        Variant assignments for flicker detection (required if user_ids
        is provided).
    alpha : float, default 0.05
        Significance level for all checks.

    Returns
    -------
    CheckResult
        Structured health report with all check results.
    """
    ctrl = check_array_like(control, "control", min_length=2)
    trt = check_array_like(treatment, "treatment", min_length=2)

    checks: list[dict] = []
    recommendations: list[str] = []

    # ── 1. SRM Check ─────────────────────────────────────────────
    from splita.core.srm import SRMCheck

    srm_result = SRMCheck([len(ctrl), len(trt)], alpha=alpha).run()
    srm_passed = srm_result.passed
    checks.append(
        {
            "name": "srm",
            "passed": srm_passed,
            "detail": srm_result.message,
        }
    )
    if not srm_passed:
        recommendations.append(
            "Sample ratio mismatch detected. Investigate randomisation "
            "logic, bot traffic, or tracking errors before analysing results."
        )

    # ── 2. Flicker Detection ─────────────────────────────────────
    flicker_rate: float | None = None
    if user_ids is not None and variant_assignments is not None:
        from splita.diagnostics.flicker import FlickerDetector

        flicker_result = FlickerDetector().detect(user_ids, variant_assignments)
        flicker_rate = flicker_result.flicker_rate
        flicker_ok = not flicker_result.is_problematic
        checks.append(
            {
                "name": "flicker",
                "passed": flicker_ok,
                "detail": flicker_result.message,
            }
        )
        if not flicker_ok:
            recommendations.append(
                f"High flicker rate ({flicker_rate:.1%}). Users are switching "
                "variants, which contaminates intent-to-treat estimates."
            )
    elif user_ids is not None:
        checks.append(
            {
                "name": "flicker",
                "passed": False,
                "detail": "user_ids provided but variant_assignments missing.",
            }
        )
        recommendations.append(
            "Provide variant_assignments along with user_ids for flicker detection."
        )

    # ── 3. Outlier Detection ─────────────────────────────────────
    combined = np.concatenate([ctrl, trt])
    q1 = float(np.percentile(combined, 25))
    q3 = float(np.percentile(combined, 75))
    iqr_val = q3 - q1
    if iqr_val > 0:
        lower_fence = q1 - 3.0 * iqr_val
        upper_fence = q3 + 3.0 * iqr_val
        n_outliers = int(np.sum((combined < lower_fence) | (combined > upper_fence)))
        outlier_pct = n_outliers / len(combined)
        has_outliers = outlier_pct > 0.01  # >1% outliers
    else:
        n_outliers = 0
        outlier_pct = 0.0
        has_outliers = False

    checks.append(
        {
            "name": "outliers",
            "passed": not has_outliers,
            "detail": f"{n_outliers} outliers ({outlier_pct:.1%} of data) "
            f"detected using 3x IQR fences.",
        }
    )
    if has_outliers:
        recommendations.append(
            f"Found {n_outliers} outliers ({outlier_pct:.1%}). Consider "
            "using OutlierHandler(method='winsorize') before analysis."
        )

    # ── 4. Data Quality ──────────────────────────────────────────
    n_ctrl_nan = int(np.sum(np.isnan(ctrl)))
    n_trt_nan = int(np.sum(np.isnan(trt)))
    nan_ok = n_ctrl_nan == 0 and n_trt_nan == 0
    checks.append(
        {
            "name": "data_quality",
            "passed": nan_ok,
            "detail": f"NaNs: control={n_ctrl_nan}, treatment={n_trt_nan}.",
        }
    )
    if not nan_ok:  # pragma: no cover
        recommendations.append(
            "Data contains NaN values. These are automatically removed "
            "by Experiment, but investigate why they exist."
        )

    # ── 5. Skewness Check ────────────────────────────────────────
    ctrl_skew = float(skew(ctrl))
    trt_skew = float(skew(trt))
    skew_ok = abs(ctrl_skew) <= 2.0 and abs(trt_skew) <= 2.0
    checks.append(
        {
            "name": "skewness",
            "passed": skew_ok,
            "detail": f"Skewness: control={ctrl_skew:.2f}, treatment={trt_skew:.2f}.",
        }
    )
    if not skew_ok:
        recommendations.append(
            "High skewness detected. Consider using method='mannwhitney' "
            "or method='bootstrap', or apply outlier handling."
        )

    # ── 6. Power Assessment ──────────────────────────────────────
    ctrl_mean = float(np.mean(ctrl))
    trt_mean = float(np.mean(trt))
    diff = trt_mean - ctrl_mean
    pooled_std = float(np.std(combined, ddof=1))

    if pooled_std > 0:
        effect_d = abs(diff) / pooled_std
        n_harmonic = 2.0 * len(ctrl) * len(trt) / (len(ctrl) + len(trt))
        z_alpha = norm.ppf(1 - alpha / 2)
        power = float(norm.cdf(effect_d * math.sqrt(n_harmonic / 2) - z_alpha))
        is_powered = power >= 0.8
    else:
        power = 1.0 if diff != 0 else 0.0
        is_powered = True

    checks.append(
        {
            "name": "power",
            "passed": is_powered,
            "detail": f"Estimated power: {power:.2%}.",
        }
    )
    if not is_powered:
        recommendations.append(
            f"Estimated power is {power:.1%}, below the 80% threshold. "
            "Consider increasing sample size or using CUPED for variance reduction."
        )

    # ── 7. Covariate Balance (if segments provided) ──────────────
    if segments is not None:
        seg_arr = np.asarray(segments)
        if len(seg_arr) == len(ctrl) + len(trt):
            ctrl_segs = seg_arr[: len(ctrl)]
            trt_segs = seg_arr[len(ctrl) :]
            ctrl_unique, ctrl_counts = np.unique(ctrl_segs, return_counts=True)
            trt_unique, trt_counts = np.unique(trt_segs, return_counts=True)
            # Check proportional balance
            ctrl_props = ctrl_counts / len(ctrl)
            trt_props = trt_counts / len(trt)
            if set(ctrl_unique.tolist()) == set(trt_unique.tolist()):
                # Sort to align
                ctrl_order = np.argsort(ctrl_unique)
                trt_order = np.argsort(trt_unique)
                max_imbalance = float(np.max(np.abs(ctrl_props[ctrl_order] - trt_props[trt_order])))
                balance_ok = max_imbalance < 0.05
                checks.append(
                    {
                        "name": "covariate_balance",
                        "passed": balance_ok,
                        "detail": f"Max segment proportion difference: {max_imbalance:.3f}.",
                    }
                )
                if not balance_ok:
                    recommendations.append(
                        f"Covariate imbalance detected (max diff={max_imbalance:.3f}). "
                        "Consider stratified analysis."
                    )
            else:
                checks.append(
                    {
                        "name": "covariate_balance",
                        "passed": False,
                        "detail": "Segment categories differ between control and treatment.",
                    }
                )
                recommendations.append(
                    "Segment categories differ between groups. Check randomisation."
                )

    # ── Aggregate ─────────────────────────────────────────────────
    all_passed = all(c["passed"] for c in checks)

    if all_passed and not recommendations:
        recommendations.append("All checks passed. Data looks healthy for analysis.")

    return CheckResult(
        srm_passed=srm_passed,
        flicker_rate=flicker_rate,
        has_outliers=has_outliers,
        is_powered=is_powered,
        all_passed=all_passed,
        checks=checks,
        recommendations=recommendations,
    )
