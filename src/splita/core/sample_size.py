"""Sample-size calculators for A/B tests.

Provides :class:`SampleSize` — a namespace of static methods for computing
required sample sizes (and minimum detectable effects) for proportion,
continuous, and ratio metrics.
"""

from __future__ import annotations

import math
import warnings
from typing import Literal

from scipy.optimize import brentq
from scipy.stats import norm

from splita._types import SampleSizeResult
from splita._validation import (
    check_in_range,
    check_is_integer,
    check_positive,
    format_error,
)

# ─── Internal helpers ──────────────────────────────────────────────────


def _z_alpha(alpha: float, alternative: str) -> float:
    """Return the critical z-value for the given alpha and sidedness."""
    if alternative == "two-sided":
        return float(norm.ppf(1.0 - alpha / 2.0))
    return float(norm.ppf(1.0 - alpha))


def _n_for_proportion(
    baseline: float,
    mde: float,
    alpha: float,
    power: float,
    alternative: str,
) -> int:
    """Core per-variant sample size for a proportion test (Farrington-Manning)."""
    p1 = baseline
    p2 = baseline + mde
    p_bar = (p1 + p2) / 2.0
    se0 = math.sqrt(2.0 * p_bar * (1.0 - p_bar))
    se1 = math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    za = _z_alpha(alpha, alternative)
    zb = float(norm.ppf(power))
    numerator = (za * se0 + zb * se1) ** 2
    return math.ceil(numerator / (mde**2))


# ─── Public API ────────────────────────────────────────────────────────


class SampleSize:
    """Namespace for sample-size and MDE calculations.

    All methods are static — no instance state is needed.  Import and call
    directly::

        from splita.core.sample_size import SampleSize

        result = SampleSize.for_proportion(baseline=0.10, mde=0.02)
    """

    # ── proportion ────────────────────────────────────────────────

    @staticmethod
    def for_proportion(
        baseline: float,
        mde: float | None = None,
        *,
        alpha: float = 0.05,
        power: float = 0.80,
        alternative: Literal["two-sided", "one-sided"] = "two-sided",
        n_variants: int = 2,
        traffic_fraction: float = 1.0,
        relative_mde: float | None = None,
    ) -> SampleSizeResult:
        """Required sample size for a conversion-rate (proportion) test.

        Uses the Farrington-Manning formula with pooled standard error
        under H0.

        Parameters
        ----------
        baseline : float
            Current conversion rate, in (0, 1).
        mde : float or None, default None
            Minimum detectable effect (absolute change in proportion).
            Required unless *relative_mde* is provided.  Mutually
            exclusive with *relative_mde*.
            Positive values detect improvement, negative values detect degradation.
            For example, ``mde=-0.01`` detects a 1pp drop in conversion.
        alpha : float, default 0.05
            Significance level (false-positive rate).
        power : float, default 0.80
            Statistical power (1 - false-negative rate).
        alternative : {"two-sided", "one-sided"}, default "two-sided"
            Whether the test is two-sided or one-sided.
        n_variants : int, default 2
            Number of variants (including control).
        traffic_fraction : float, default 1.0
            Fraction of total traffic allocated to the experiment.
            Stored in the result but does **not** inflate ``n_per_variant``.
        relative_mde : float or None, default None
            If provided, the absolute MDE is computed as
            ``baseline * relative_mde``.  Mutually exclusive with *mde*.

        Returns
        -------
        SampleSizeResult
            Frozen dataclass with ``n_per_variant``, ``n_total``, and
            supporting fields.

        Raises
        ------
        ValueError
            If any parameter is out of its valid range, or if both/neither
            of *mde* and *relative_mde* are provided.

        Examples
        --------
        >>> res = SampleSize.for_proportion(0.10, 0.02)
        >>> res.n_per_variant  # ~3843
        3843
        """
        # --- resolve MDE ---
        if mde is not None and relative_mde is not None:
            raise ValueError(
                format_error(
                    "provide either `mde` or `relative_mde`, not both.",
                    f"got mde={mde} and relative_mde={relative_mde}.",
                    "use one or the other to specify the minimum detectable effect.",
                )
            )
        if mde is None and relative_mde is None:
            raise ValueError(
                format_error(
                    "either `mde` or `relative_mde` must be provided.",
                    "both are None.",
                    "pass mde=0.02 for an absolute effect or "
                    "relative_mde=0.10 for a 10% relative effect.",
                )
            )
        if relative_mde is not None:
            mde = baseline * relative_mde

        # --- validation ---
        check_in_range(baseline, "baseline", 0.0, 1.0, hint="pass a proportion between 0 and 1.")
        p2 = baseline + mde
        if not (0.0 < p2 < 1.0):
            raise ValueError(
                format_error(
                    f"`baseline + mde` must be in (0, 1), got {p2}.",
                    f"baseline={baseline}, mde={mde}.",
                    "reduce the MDE or adjust the baseline.",
                )
            )
        check_in_range(alpha, "alpha", 0.0, 1.0, hint="typical values are 0.05, 0.01, or 0.10.")
        check_in_range(power, "power", 0.0, 1.0, hint="typical values are 0.80 or 0.90.")
        check_is_integer(
            n_variants,
            "n_variants",
            min_value=2,
            hint="include at least a control and one treatment.",
        )
        check_in_range(
            traffic_fraction,
            "traffic_fraction",
            0.0,
            1.0,
            high_inclusive=True,
            hint="typical values are 0.1, 0.5, or 1.0.",
        )

        if baseline < 0.01:
            warnings.warn(
                f"baseline={baseline} is very small (<1%). "
                "The normal approximation may not be reliable. "
                "Consider an exact test or simulation.",
                RuntimeWarning,
                stacklevel=2,
            )

        # --- compute ---
        n_per_variant = _n_for_proportion(baseline, mde, alpha, power, alternative)
        n_total = n_per_variant * int(n_variants)

        return SampleSizeResult(
            n_per_variant=n_per_variant,
            n_total=n_total,
            alpha=alpha,
            power=power,
            mde=mde,
            relative_mde=relative_mde,
            baseline=baseline,
            metric="proportion",
            effect_size=None,
            days_needed=None,
        )

    # ── continuous (mean) ─────────────────────────────────────────

    @staticmethod
    def for_mean(
        baseline_mean: float,
        baseline_std: float,
        mde: float | None = None,
        *,
        alpha: float = 0.05,
        power: float = 0.80,
        alternative: Literal["two-sided", "one-sided"] = "two-sided",
        n_variants: int = 2,
        relative_mde: float | None = None,
    ) -> SampleSizeResult:
        """Required sample size for a continuous-metric test.

        Uses the two-sample *t*-test power formula with equal group sizes.

        Parameters
        ----------
        baseline_mean : float
            Expected mean of the control group.
        baseline_std : float
            Expected standard deviation (must be > 0).
        mde : float or None, default None
            Minimum detectable effect (absolute change in mean).
            Required unless *relative_mde* is provided.  Mutually
            exclusive with *relative_mde*.
        alpha : float, default 0.05
            Significance level.
        power : float, default 0.80
            Statistical power.
        alternative : {"two-sided", "one-sided"}, default "two-sided"
            Sidedness of the test.
        n_variants : int, default 2
            Number of variants (including control).
        relative_mde : float or None, default None
            If provided, ``mde = baseline_mean * relative_mde``.
            Mutually exclusive with *mde*.

        Returns
        -------
        SampleSizeResult

        Raises
        ------
        ValueError
            If parameters are out of range, or if both/neither of
            *mde* and *relative_mde* are provided.

        Examples
        --------
        >>> res = SampleSize.for_mean(25.0, 40.0, 2.0)
        >>> res.n_per_variant > 0
        True
        """
        # --- resolve MDE ---
        if mde is not None and relative_mde is not None:
            raise ValueError(
                format_error(
                    "provide either `mde` or `relative_mde`, not both.",
                    f"got mde={mde} and relative_mde={relative_mde}.",
                    "use one or the other to specify the minimum detectable effect.",
                )
            )
        if mde is None and relative_mde is None:
            raise ValueError(
                format_error(
                    "either `mde` or `relative_mde` must be provided.",
                    "both are None.",
                    "pass mde=2.0 for an absolute effect or "
                    "relative_mde=0.10 for a 10% relative effect.",
                )
            )
        if relative_mde is not None:
            mde = baseline_mean * relative_mde

        # --- validation ---
        check_positive(
            baseline_std,
            "baseline_std",
            hint="standard deviation must be strictly positive.",
        )
        if mde == 0.0:
            raise ValueError(
                format_error(
                    "`mde` must not be zero.",
                    "an effect size of zero means no difference to detect.",
                    "pass a nonzero minimum detectable effect.",
                )
            )
        check_in_range(alpha, "alpha", 0.0, 1.0)
        check_in_range(power, "power", 0.0, 1.0)
        check_is_integer(n_variants, "n_variants", min_value=2)

        # --- compute ---
        effect_size_d = mde / baseline_std
        za = _z_alpha(alpha, alternative)
        zb = float(norm.ppf(power))
        n_per_variant = math.ceil(2.0 * ((za + zb) / effect_size_d) ** 2)
        n_total = n_per_variant * int(n_variants)

        return SampleSizeResult(
            n_per_variant=n_per_variant,
            n_total=n_total,
            alpha=alpha,
            power=power,
            mde=mde,
            relative_mde=relative_mde,
            baseline=baseline_mean,
            metric="mean",
            effect_size=abs(effect_size_d),
            days_needed=None,
        )

    # ── ratio (delta method) ──────────────────────────────────────

    @staticmethod
    def for_ratio(
        baseline_num_mean: float,
        baseline_den_mean: float,
        baseline_num_std: float,
        baseline_den_std: float,
        baseline_covariance: float,
        mde: float,
        *,
        alpha: float = 0.05,
        power: float = 0.80,
        alternative: Literal["two-sided", "one-sided"] = "two-sided",
        n_variants: int = 2,
    ) -> SampleSizeResult:
        """Required sample size for a ratio metric using the delta method.

        Implements the variance estimator from Deng et al. (2018).

        Parameters
        ----------
        baseline_num_mean : float
            Mean of the numerator metric.
        baseline_den_mean : float
            Mean of the denominator metric (must be > 0).
        baseline_num_std : float
            Standard deviation of the numerator metric.
        baseline_den_std : float
            Standard deviation of the denominator metric.
        baseline_covariance : float
            Covariance between numerator and denominator.
        mde : float
            Minimum detectable effect (absolute change in the ratio).
        alpha : float, default 0.05
            Significance level.
        power : float, default 0.80
            Statistical power.
        alternative : {"two-sided", "one-sided"}, default "two-sided"
            Sidedness of the test.
        n_variants : int, default 2
            Number of variants.

        Returns
        -------
        SampleSizeResult

        Raises
        ------
        ValueError
            If ``baseline_den_mean <= 0`` or other parameters are invalid.
        """
        # --- validation ---
        check_positive(
            baseline_den_mean,
            "baseline_den_mean",
            hint="the denominator mean must be strictly positive.",
        )
        if mde == 0.0:
            raise ValueError(
                format_error(
                    "`mde` must not be zero.",
                    "an effect size of zero means no difference to detect.",
                    "pass a nonzero minimum detectable effect.",
                )
            )
        check_in_range(alpha, "alpha", 0.0, 1.0)
        check_in_range(power, "power", 0.0, 1.0)
        check_is_integer(n_variants, "n_variants", min_value=2)

        # --- compute (delta method variance) ---
        r = baseline_num_mean / baseline_den_mean
        d2 = baseline_den_mean**2
        delta_var = (
            baseline_num_std**2 / d2
            - 2.0 * r * baseline_covariance / d2
            + r**2 * baseline_den_std**2 / d2
        )

        za = _z_alpha(alpha, alternative)
        zb = float(norm.ppf(power))
        n_per_variant = math.ceil(2 * delta_var * ((za + zb) / mde) ** 2)
        n_total = n_per_variant * int(n_variants)

        return SampleSizeResult(
            n_per_variant=n_per_variant,
            n_total=n_total,
            alpha=alpha,
            power=power,
            mde=mde,
            relative_mde=None,
            baseline=r,
            metric="ratio",
            effect_size=None,
            days_needed=None,
        )

    # ── inverse: MDE from sample size ─────────────────────────────

    @staticmethod
    def mde_for_proportion(
        baseline: float,
        n: int,
        *,
        alpha: float = 0.05,
        power: float = 0.80,
        alternative: Literal["two-sided", "one-sided"] = "two-sided",
    ) -> float:
        """Minimum detectable effect for a proportion test given a fixed n.

        Numerically inverts :meth:`for_proportion` using Brent's method.

        Parameters
        ----------
        baseline : float
            Current conversion rate, in (0, 1).
        n : int
            Per-variant sample size.
        alpha : float, default 0.05
            Significance level.
        power : float, default 0.80
            Statistical power.
        alternative : {"two-sided", "one-sided"}, default "two-sided"
            Sidedness of the test.

        Returns
        -------
        float
            The smallest absolute MDE detectable with the given *n*.

        Raises
        ------
        ValueError
            If parameters are out of range.

        Examples
        --------
        >>> mde = SampleSize.mde_for_proportion(0.10, 3843)
        >>> round(mde, 4)  # ~0.02
        0.02
        """
        # --- validation ---
        check_in_range(baseline, "baseline", 0.0, 1.0)
        check_is_integer(n, "n", min_value=1)
        check_in_range(alpha, "alpha", 0.0, 1.0)
        check_in_range(power, "power", 0.0, 1.0)

        n = int(n)

        # Search range: mde in (epsilon, max_mde)
        max_mde = min(baseline, 1.0 - baseline) - 1e-9

        if max_mde <= 0:
            raise ValueError(
                format_error(
                    f"Cannot compute MDE for baseline={baseline}.",
                    "baseline is too close to 0 or 1.",
                )
            )

        def objective(mde: float) -> float:
            n_req = _n_for_proportion(
                baseline,
                mde,
                alpha,
                power,
                alternative,
            )
            return float(n_req - n)

        # Check whether even the largest possible MDE requires more samples
        # than n. If so, brentq would fail because both endpoints are positive.
        if objective(max_mde) > 0:
            raise ValueError(
                format_error(
                    "`n` is too small to detect any effect at this baseline.",
                    f"with n={n} per variant and baseline={baseline}, even the "
                    f"largest possible MDE ({max_mde:.6g}) requires more samples.",
                    "increase n or use a less extreme baseline.",
                )
            )

        # brentq needs f(a) and f(b) to have opposite signs.
        # At very small mde, n_required >> n  (positive).
        # At large mde, n_required << n  (negative).
        result = brentq(objective, 1e-10, max_mde, xtol=1e-10, maxiter=200)
        return float(result)

    # ── duration ──────────────────────────────────────────────────

    @staticmethod
    def duration(
        n_required: int,
        daily_users: int,
        *,
        traffic_fraction: float = 1.0,
        ramp_days: int = 0,
    ) -> int:
        """Estimate experiment duration in days.

        Parameters
        ----------
        n_required : int
            Total required sample size across all variants.
        daily_users : int
            Expected daily users (or daily eligible traffic).
        traffic_fraction : float, default 1.0
            Fraction of daily traffic allocated to the experiment.
        ramp_days : int, default 0
            Additional ramp-up days to add.

        Returns
        -------
        int
            Estimated number of calendar days.

        Raises
        ------
        ValueError
            If parameters are out of range.

        Examples
        --------
        >>> SampleSize.duration(10000, 1000)
        10
        """
        # --- validation ---
        check_positive(n_required, "n_required", hint="pass the total required sample size.")
        check_positive(daily_users, "daily_users", hint="pass the expected number of daily users.")
        check_in_range(
            traffic_fraction,
            "traffic_fraction",
            0.0,
            1.0,
            high_inclusive=True,
            hint="typical values are 0.1, 0.5, or 1.0.",
        )
        if ramp_days < 0:
            raise ValueError(
                format_error(
                    f"`ramp_days` must be >= 0, got {ramp_days}.",
                    "ramp-up days cannot be negative.",
                )
            )

        return math.ceil(n_required / (daily_users * traffic_fraction)) + ramp_days
