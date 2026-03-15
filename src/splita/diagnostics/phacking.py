"""P-hacking detection via p-curve analysis.

Analyses the distribution of significant p-values from multiple experiments
to detect evidence of selective reporting or p-hacking
(Simonsohn, Nelson & Simmons, 2014).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kstest

from splita._types import PHackingResult
from splita._validation import format_error

__all__ = ["PHackingDetector"]


class PHackingDetector:
    """Detect p-hacking via p-curve analysis.

    Under a true effect, significant p-values should be right-skewed
    (clustered near 0). Under the null (no effect), they are uniform.
    Under p-hacking, they bunch just below 0.05.

    Parameters
    ----------
    significance_threshold : float, default 0.05
        Threshold below which p-values are considered significant.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> honest_pvals = rng.beta(1, 10, 50) * 0.05  # right-skewed
    >>> detector = PHackingDetector()
    >>> r = detector.detect(honest_pvals.tolist())
    >>> r.suspicious
    False
    """

    def __init__(self, *, significance_threshold: float = 0.05) -> None:
        if not 0 < significance_threshold < 1:
            raise ValueError(
                format_error(
                    f"`significance_threshold` must be in (0, 1), got {significance_threshold}.",
                    "this is the threshold for selecting significant p-values.",
                    "the standard threshold is 0.05.",
                )
            )
        self._threshold = significance_threshold

    def detect(self, pvalues: list[float]) -> PHackingResult:
        """Analyse a collection of p-values for evidence of p-hacking.

        Parameters
        ----------
        pvalues : list of float
            p-values from multiple experiments / hypothesis tests.

        Returns
        -------
        PHackingResult
            Frozen dataclass with p-curve test results and diagnostics.

        Raises
        ------
        ValueError
            If fewer than 3 p-values are provided, or any p-value is outside [0, 1].
        """
        if not isinstance(pvalues, (list, tuple, np.ndarray)):
            raise TypeError(
                format_error(
                    "`pvalues` must be a list of floats.",
                    f"got type {type(pvalues).__name__}.",
                )
            )

        n_experiments = len(pvalues)
        if n_experiments < 3:
            raise ValueError(
                format_error(
                    f"`pvalues` must have at least 3 elements, got {n_experiments}.",
                    "p-curve analysis requires multiple experiments.",
                    "collect more p-values before running this test.",
                )
            )

        arr = np.asarray(pvalues, dtype=float)

        if np.any(arr < 0) or np.any(arr > 1):
            raise ValueError(
                format_error(
                    "All p-values must be in [0, 1].",
                    "found values outside this range.",
                    "check that your inputs are valid p-values.",
                )
            )

        if np.any(np.isnan(arr)):
            raise ValueError(
                format_error(
                    "p-values must not contain NaN.",
                    "found NaN values in input.",
                )
            )

        # Select significant p-values
        sig = arr[arr < self._threshold]
        n_sig = len(sig)

        if n_sig < 3:
            return PHackingResult(
                suspicious=False,
                p_curve_test_pvalue=1.0,
                bunching_near_05=False,
                n_experiments=n_experiments,
                message=(
                    f"Only {n_sig} significant p-value(s) found (need >= 3). "
                    "Insufficient data for p-curve analysis."
                ),
            )

        # Transform significant p-values to pp-values (uniform under null)
        # pp_i = p_i / threshold (uniform[0,1] under the null)
        pp = sig / self._threshold

        # Test for right-skewness using KS test against Uniform(0,1)
        # Under true effect, pp should be right-skewed (more mass near 0)
        _ks_stat, ks_pvalue = kstest(pp, "uniform", args=(0, 1))

        # Check if distribution is right-skewed (more small pp-values)
        # Median of pp < 0.5 under true effect
        median_pp = float(np.median(pp))
        is_right_skewed = median_pp < 0.5

        # Bunching near threshold (p-hacking signal)
        # Check if disproportionate mass is near the threshold
        near_threshold = np.sum((sig > self._threshold * 0.8) & (sig <= self._threshold))
        expected_near = n_sig * 0.2  # expect 20% in the top 20% band
        bunching_near_05 = bool(near_threshold > max(expected_near * 2, 3))

        # Determine if suspicious:
        # 1. Not right-skewed (flat or left-skewed) OR bunching near threshold
        suspicious = bool((not is_right_skewed and ks_pvalue < 0.1) or bunching_near_05)

        # Build message
        if suspicious and bunching_near_05:
            message = (
                "Suspicious: excess bunching of p-values just below the "
                f"significance threshold ({self._threshold}). "
                f"Found {int(near_threshold)} of {n_sig} significant p-values "
                f"in the top 20% band near the threshold."
            )
        elif suspicious:
            message = (
                "Suspicious: significant p-values are not right-skewed "
                f"(median pp-value = {median_pp:.3f}, expected < 0.5 "
                "under true effects). This pattern is consistent with "
                "p-hacking or selective reporting."
            )
        elif is_right_skewed:
            message = (
                "No evidence of p-hacking. Significant p-values show "
                "the right-skewed pattern expected under true effects "
                f"(median pp-value = {median_pp:.3f})."
            )
        else:
            message = (
                "No strong evidence of p-hacking, but p-curve is not "
                f"clearly right-skewed (median pp-value = {median_pp:.3f}). "
                "Effects may be weak or absent."
            )

        return PHackingResult(
            suspicious=suspicious,
            p_curve_test_pvalue=float(ks_pvalue),
            bunching_near_05=bunching_near_05,
            n_experiments=n_experiments,
            message=message,
        )
