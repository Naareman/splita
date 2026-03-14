"""SRMCheck — Sample Ratio Mismatch detector.

Tests whether the observed traffic split matches the configured split
using a chi-square goodness-of-fit test.  SRM invalidates all experiment
results — this should be the first check run on any experiment.
"""

from __future__ import annotations

import math
import warnings
from typing import Sequence

from scipy.stats import chi2

from splita._types import SRMResult
from splita._validation import (
    check_in_range,
    check_probabilities_sum_to_one,
    format_error,
)


class SRMCheck:
    """Detect Sample Ratio Mismatch via chi-square goodness-of-fit.

    Parameters
    ----------
    observed : Sequence[int]
        Observed user counts per variant (e.g., ``[4850, 5150]`` for A/B).
    expected_fractions : Sequence[float] or None, default None
        Expected traffic fraction per variant.  Must sum to 1.0.
        If ``None``, an equal split is assumed.
    alpha : float, default 0.01
        Significance level.  The default of 0.01 is stricter than the
        typical 0.05 because SRM is a data-quality check, not a
        scientific hypothesis test.
    min_expected_count : int, default 5
        Minimum expected count per cell for the chi-square approximation
        to be valid.  A ``RuntimeWarning`` is emitted when any expected
        count falls below this threshold.

    Examples
    --------
    >>> from splita.core.srm import SRMCheck
    >>> result = SRMCheck([4850, 5150]).run()
    >>> result.passed
    True
    """

    def __init__(
        self,
        observed: Sequence[int],
        *,
        expected_fractions: Sequence[float] | None = None,
        alpha: float = 0.01,
        min_expected_count: int = 5,
    ) -> None:
        # --- validate observed -------------------------------------------
        if len(observed) < 2:
            raise ValueError(
                format_error(
                    "`observed` must have at least 2 variants, "
                    f"got {len(observed)}.",
                    "a single variant cannot be checked for ratio mismatch.",
                    "pass counts for all variants, e.g. [5000, 5000].",
                )
            )

        for i, count in enumerate(observed):
            if count < 0:
                raise ValueError(
                    format_error(
                        f"`observed[{i}]` must be >= 0, got {count}.",
                        "observed counts cannot be negative.",
                        "check your data pipeline for sign errors.",
                    )
                )

        for i, count in enumerate(observed):
            if math.isnan(count) or math.isinf(count):
                raise ValueError(
                    format_error(
                        f"`observed` contains non-finite value at index {i}: {count}.",
                        hint="observed counts must be finite non-negative integers.",
                    )
                )

        # --- validate expected_fractions ---------------------------------
        k = len(observed)
        if expected_fractions is None:
            fractions = [1.0 / k] * k
        else:
            if len(expected_fractions) != k:
                raise ValueError(
                    format_error(
                        "`expected_fractions` must have the same length as "
                        f"`observed` ({k}), got {len(expected_fractions)}.",
                        "each variant needs exactly one fraction.",
                        "pass one fraction per variant.",
                    )
                )
            check_probabilities_sum_to_one(
                expected_fractions, "expected_fractions"
            )
            fractions = list(expected_fractions)

        # --- validate alpha ----------------------------------------------
        check_in_range(
            alpha,
            "alpha",
            0.0,
            1.0,
            hint="typical values are 0.01 or 0.05.",
        )

        # --- validate min_expected_count ---------------------------------
        if min_expected_count < 1:
            raise ValueError(
                format_error(
                    "`min_expected_count` must be >= 1, "
                    f"got {min_expected_count}.",
                    "the chi-square approximation requires at least 1 "
                    "expected observation per cell.",
                    "use the default value of 5.",
                )
            )

        self._observed = list(observed)
        self._fractions = fractions
        self._alpha = alpha
        self._min_expected_count = min_expected_count

    def run(self) -> SRMResult:
        """Execute the SRM check.

        Returns
        -------
        SRMResult
            Frozen dataclass with chi-square statistic, p-value,
            pass/fail flag, per-variant deviations, and a human-readable
            message.
        """
        observed = self._observed
        fractions = self._fractions
        alpha = self._alpha

        # 1. Total observations
        n = sum(observed)

        if n == 0:
            raise ValueError(
                format_error(
                    "`observed` must contain at least some observations, got all zeros.",
                    detail=f"total observations: {n}.",
                    hint="an experiment with zero observations cannot be checked for SRM.",
                )
            )

        # 2. Expected counts
        expected = [n * f for f in fractions]

        # 3. Check minimum expected count
        if any(e < self._min_expected_count for e in expected):
            warnings.warn(
                "One or more expected counts are below "
                f"{self._min_expected_count}. The chi-square approximation "
                "may not be reliable.",
                RuntimeWarning,
                stacklevel=2,
            )

        # 4. Chi-square statistic
        chi2_stat = sum(
            (o - e) ** 2 / e for o, e in zip(observed, expected) if e > 0
        )

        # 5. P-value  (k-1 degrees of freedom)
        k = len(observed)
        df = k - 1
        pvalue = float(chi2.sf(chi2_stat, df))

        # 6. Pass/fail
        passed = bool(pvalue >= alpha)

        # 7. Deviations
        deviations_pct = [
            ((o - e) / e * 100.0) if e > 0 else (float('inf') if o > 0 else 0.0)
            for o, e in zip(observed, expected)
        ]

        # 8. Worst variant (largest absolute deviation)
        worst_variant = max(
            range(k), key=lambda i: abs(deviations_pct[i])
        )

        # 9. Human-readable message
        if passed:
            message = (
                f"No sample ratio mismatch detected (p={pvalue:.4f})."
            )
        else:
            dev = deviations_pct[worst_variant]
            message = (
                f"Sample ratio mismatch detected (p={pvalue:.4f}). "
                f"Variant {worst_variant} has the largest deviation "
                f"({dev:.1f}%). Experiment results cannot be trusted "
                "until the cause is identified."
            )

        return SRMResult(
            observed=observed,
            expected_counts=expected,
            chi2_statistic=chi2_stat,
            pvalue=pvalue,
            passed=passed,
            alpha=alpha,
            deviations_pct=deviations_pct,
            worst_variant=worst_variant,
            message=message,
        )
