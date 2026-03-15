"""Internal advisory system — generates contextual recommendations.

Every splita class can call these functions to check if the user's
choices are optimal for their data and emit helpful warnings.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np

_VERBOSE = False


def info(message: str) -> None:
    """Print info message if verbose mode is active."""
    if _VERBOSE:
        print(f"  [splita] {message}", file=sys.stderr)


def advise_method_choice(
    data: np.ndarray,
    chosen_method: str,
    metric_type: str,
    n: int,
) -> None:
    """Warn if the chosen method isn't ideal for this data."""
    from scipy.stats import skew as scipy_skew

    if metric_type == "continuous" and np.std(data) > 0:
        s = float(scipy_skew(data))

        # Skewed data + ttest
        if chosen_method == "ttest" and abs(s) > 2:
            warnings.warn(
                f"Your data has high skewness ({s:.1f}). The t-test assumes "
                f"approximate normality. Consider method='mannwhitney' or "
                f"method='bootstrap' for more reliable results.",
                RuntimeWarning,
                stacklevel=3,
            )

        # Large n + mannwhitney (unnecessarily conservative)
        if chosen_method == "mannwhitney" and n > 5000 and abs(s) < 1:
            warnings.warn(
                f"With n={n} and low skewness ({s:.1f}), the t-test would be "
                f"more powerful than Mann-Whitney. Consider method='ttest'.",
                RuntimeWarning,
                stacklevel=3,
            )

    # Small n + ztest
    if chosen_method == "ztest" and n < 30:
        warnings.warn(
            f"With n={n} per group, the z-test normal approximation may not "
            f"hold. Consider method='permutation' or method='bootstrap'.",
            RuntimeWarning,
            stacklevel=3,
        )


def advise_sample_size(n_ctrl: int, n_trt: int, metric_type: str) -> None:
    """Warn about sample size issues."""
    n_min = min(n_ctrl, n_trt)

    if n_min < 30:
        warnings.warn(
            f"Small sample size (n={n_min}). Statistical tests may be unreliable. "
            f"Consider using method='permutation' or method='bootstrap'.",
            RuntimeWarning,
            stacklevel=3,
        )

    # Imbalanced groups
    ratio = max(n_ctrl, n_trt) / max(min(n_ctrl, n_trt), 1)
    if ratio > 3:
        warnings.warn(
            f"Groups are imbalanced ({n_ctrl} vs {n_trt}, ratio {ratio:.1f}:1). "
            f"Consider whether this is intentional. Balanced groups maximize power.",
            RuntimeWarning,
            stacklevel=3,
        )


def advise_pre_analysis(n_ctrl: int, n_trt: int) -> None:
    """Suggest running SRM check before analysis."""
    # This is a gentle reminder, not a warning
    pass  # Could emit info-level message if we had a logging system


def advise_variance_reduction(
    variance_reduction: float,
    method_used: str,
    has_pre_data: bool,
) -> None:
    """Suggest better variance reduction if current is weak."""
    if variance_reduction < 0.05 and method_used == "CUPED":
        warnings.warn(
            f"CUPED reduced variance by only {variance_reduction:.1%}. "
            f"Consider CUPAC with ML features for better reduction, "
            f"or check that your pre-experiment data is from the right time period.",
            RuntimeWarning,
            stacklevel=3,
        )

    if not has_pre_data:
        # Could suggest CUPED if they have pre-data they didn't use
        pass


def advise_multiple_testing(n_metrics: int, corrected: bool) -> None:
    """Warn about multiple testing."""
    if n_metrics > 1 and not corrected:
        warnings.warn(
            f"You are testing {n_metrics} metrics without correction. "
            f"False positive rate inflated to ~{1 - (1 - 0.05) ** n_metrics:.1%}. "
            f"Use MultipleCorrection to adjust p-values.",
            RuntimeWarning,
            stacklevel=3,
        )


def advise_sequential(n_peeks: int, is_sequential: bool) -> None:
    """Warn about peeking without sequential methods."""
    if n_peeks > 1 and not is_sequential:
        warnings.warn(
            f"You've peeked at results {n_peeks} times. Without sequential "
            f"testing (mSPRT, GroupSequential), this inflates false positives. "
            f"Consider using mSPRT for always-valid inference.",
            RuntimeWarning,
            stacklevel=3,
        )
