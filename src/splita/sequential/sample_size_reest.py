"""SampleSizeReestimation — mid-experiment sample size adjustment.

Based on conditional power at an interim analysis.

References
----------
.. [1] Mehta, C. R. & Pocock, S. J. (2011).  "Adaptive increase in
       sample size when interim results are promising."
"""

from __future__ import annotations

import math

from scipy.stats import norm

from splita._types import ReestimationResult
from splita._validation import (
    check_in_range,
    check_positive,
    format_error,
)


class SampleSizeReestimation:
    """Mid-experiment sample size adjustment based on conditional power.

    Given interim results, computes the conditional power at the current
    sample size and re-estimates the required sample size to achieve the
    target power.

    Examples
    --------
    >>> result = SampleSizeReestimation.reestimate(
    ...     current_n=500, interim_effect=0.05, interim_se=0.04,
    ...     target_power=0.80, alpha=0.05,
    ... )
    >>> result.new_n_per_variant >= 500
    True
    """

    @staticmethod
    def reestimate(
        current_n: int,
        interim_effect: float,
        interim_se: float,
        target_power: float = 0.80,
        alpha: float = 0.05,
    ) -> ReestimationResult:
        """Re-estimate sample size from interim data.

        Parameters
        ----------
        current_n : int
            Current sample size per variant.
        interim_effect : float
            Observed treatment effect at the interim analysis.
        interim_se : float
            Standard error of the effect estimate at the interim.
        target_power : float, default 0.80
            Desired statistical power.
        alpha : float, default 0.05
            Significance level.

        Returns
        -------
        ReestimationResult
            New sample size with conditional power analysis.

        Raises
        ------
        ValueError
            If parameters are out of valid ranges.
        """
        if not isinstance(current_n, (int, float)) or current_n < 2:
            raise ValueError(
                format_error(
                    f"`current_n` must be an integer >= 2, got {current_n}.",
                    hint="pass the current sample size per variant.",
                )
            )
        current_n = int(current_n)

        check_positive(interim_se, "interim_se")
        check_in_range(
            target_power,
            "target_power",
            0.0,
            1.0,
            hint="typical values are 0.80 or 0.90.",
        )
        check_in_range(
            alpha,
            "alpha",
            0.0,
            1.0,
            hint="typical values are 0.05, 0.01, or 0.10.",
        )

        z_alpha = float(norm.ppf(1.0 - alpha / 2.0))

        # Conditional power at current sample size
        # B_n = effect / se is the current Z-statistic
        z_current = abs(interim_effect) / interim_se
        # Conditional power: P(reject H0 | interim data)
        # = Phi(z_current - z_alpha) for large samples
        cond_power_current = float(norm.cdf(z_current - z_alpha))
        cond_power_current = max(0.0, min(1.0, cond_power_current))

        # Required sample size for target power
        # SE scales as 1/sqrt(n), so effect/SE scales as sqrt(n)
        # We need z_new = z_alpha + z_beta where z_beta = Phi^{-1}(power)
        z_beta = float(norm.ppf(target_power))
        z_needed = z_alpha + z_beta

        if abs(interim_effect) < 1e-15:
            # Effect is essentially zero — can't achieve power
            new_n = current_n * 10  # large increase as signal
            cond_power_new = 0.0
        else:
            # SE at current_n: interim_se
            # SE at new_n: interim_se * sqrt(current_n / new_n)
            # z_new = |effect| / (interim_se * sqrt(current_n / new_n))
            #       = z_current * sqrt(new_n / current_n)
            # Solve: z_needed = z_current * sqrt(new_n / current_n)
            if z_current > 0:
                ratio = (z_needed / z_current) ** 2
                new_n = max(current_n, int(math.ceil(current_n * ratio)))
            else:
                new_n = current_n * 10

            # Conditional power at new_n
            z_new = z_current * math.sqrt(new_n / current_n)
            cond_power_new = float(norm.cdf(z_new - z_alpha))
            cond_power_new = max(0.0, min(1.0, cond_power_new))

        increase_ratio = new_n / current_n

        return ReestimationResult(
            new_n_per_variant=new_n,
            conditional_power_current=cond_power_current,
            conditional_power_new=cond_power_new,
            increase_ratio=increase_ratio,
        )
