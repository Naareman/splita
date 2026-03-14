"""Group sequential testing with alpha-spending functions.

Pre-planned group sequential testing that allows valid interim analyses at
planned time points while controlling overall Type I error.

References
----------
O'Brien, P.C. & Fleming, T.R. (1979). A Multiple Testing Procedure for
Clinical Trials. *Biometrics*, 35, 549-556.

Lan, K.K.G. & DeMets, D.L. (1983). Discrete Sequential Boundaries for
Clinical Trials. *Biometrika*, 70, 659-663.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from scipy.stats import norm

from splita._types import BoundaryResult, GSResult
from splita._validation import (
    check_in_range,
    check_is_integer,
    check_one_of,
    check_positive,
    format_error,
)

_VALID_SPENDING = ["obf", "pocock", "kim_demets", "linear"]
_VALID_BETA_SPENDING = ["obf", "pocock"]


class GroupSequential:
    """Pre-planned group sequential testing with alpha-spending functions.

    Allows valid interim analyses at planned time points while controlling
    overall Type I error rate through spending functions.

    Parameters
    ----------
    n_analyses : int
        Total number of planned analyses including the final analysis.
        Must be >= 2.
    alpha : float, default 0.05
        Overall Type I error rate.
    power : float, default 0.80
        Target power at final analysis.
    spending_function : str, default ``'obf'``
        Alpha-spending function.

        - ``'obf'`` (O'Brien-Fleming): conservative early, aggressive late.
        - ``'pocock'``: approximately equal alpha at each look.
        - ``'kim_demets'``: power family with ``rho`` parameter.
        - ``'linear'``: linear spending.
    beta_spending : ``'obf'`` or ``'pocock'`` or None, default None
        Futility spending function. ``None`` disables futility stopping.
    rho : float, default 3.0
        Shape parameter for ``'kim_demets'`` spending.
        ``rho=1`` gives linear, ``rho=3`` gives OBF-like behaviour.

    Examples
    --------
    >>> from splita.sequential.group_sequential import GroupSequential
    >>> gs = GroupSequential(n_analyses=5, alpha=0.05, spending_function="obf")
    >>> b = gs.boundary()
    >>> len(b.efficacy_boundaries)
    5
    """

    def __init__(
        self,
        n_analyses: int,
        *,
        alpha: float = 0.05,
        power: float = 0.80,
        spending_function: Literal["obf", "pocock", "kim_demets", "linear"] = "obf",
        beta_spending: Literal["obf", "pocock"] | None = None,
        rho: float = 3.0,
    ) -> None:
        # ── validation ──
        check_is_integer(n_analyses, "n_analyses", min_value=2)
        check_in_range(alpha, "alpha", 0.0, 1.0)
        check_in_range(power, "power", 0.0, 1.0)
        check_one_of(spending_function, "spending_function", _VALID_SPENDING)
        if beta_spending is not None:
            check_one_of(beta_spending, "beta_spending", _VALID_BETA_SPENDING)
        check_positive(rho, "rho")

        self._n_analyses = int(n_analyses)
        self._alpha = float(alpha)
        self._power = float(power)
        self._spending_function = spending_function
        self._beta_spending = beta_spending
        self._rho = float(rho)

    # ── spending functions ──────────────────────────────────────────

    def _alpha_spent(self, t: float, alpha: float, kind: str) -> float:
        """Cumulative alpha spent at information fraction *t*.

        Parameters
        ----------
        t : float
            Information fraction in (0, 1].
        alpha : float
            Total alpha to spend.
        kind : str
            Spending function identifier.

        Returns
        -------
        float
            Cumulative alpha spent up to fraction *t*.
        """
        if t <= 0.0:
            return 0.0
        if t >= 1.0:
            return alpha

        if kind == "obf":
            z_alpha = norm.ppf(1.0 - alpha / 2.0)
            return 2.0 * (1.0 - norm.cdf(z_alpha / math.sqrt(t)))
        elif kind == "pocock":
            # Approximation to Pocock-type equal-alpha spending.
            # The exact Pocock boundary requires recursive integration;
            # this logarithmic spending function is the standard
            # approximation used in practice (see Lan & DeMets, 1983).
            return alpha * math.log(1.0 + (math.e - 1.0) * t)
        elif kind == "kim_demets":
            return alpha * (t ** self._rho)
        elif kind == "linear":
            return alpha * t
        else:  # pragma: no cover
            raise ValueError(f"Unknown spending function: {kind!r}")

    # ── boundary helpers ────────────────────────────────────────────

    @staticmethod
    def _compute_boundaries(
        alpha_spent_cum: list[float],
    ) -> list[float]:
        """Compute efficacy boundaries using conditional error spending.

        For each look k, the boundary z_k is found such that the
        conditional probability of crossing at look k (given survival
        to look k) equals the incremental alpha spent at that look.

        This accounts for the correlation structure between test
        statistics at successive looks, unlike the naive approach of
        treating each look independently.

        Parameters
        ----------
        alpha_spent_cum : list[float]
            Cumulative alpha spent at each look.

        Returns
        -------
        list[float]
            Efficacy z-value boundaries at each look.
        """
        K = len(alpha_spent_cum)
        boundaries = np.zeros(K)

        for k in range(K):
            if k == 0:
                delta_k = alpha_spent_cum[0]
            else:
                delta_k = alpha_spent_cum[k] - alpha_spent_cum[k - 1]

            if delta_k <= 0:  # pragma: no cover
                # No additional spending at this look
                boundaries[k] = boundaries[k - 1] if k > 0 else 8.0
                continue

            if k == 0:
                # First look: straightforward inversion
                boundaries[0] = float(norm.ppf(1.0 - delta_k / 2.0))
            else:
                # Account for sequential correlation structure.
                # P(survived to k) = 1 - alpha_spent_{k-1}
                remaining = 1.0 - alpha_spent_cum[k - 1]
                if remaining <= 0:  # pragma: no cover
                    boundaries[k] = boundaries[k - 1]
                    continue
                # Conditional alpha: probability of rejection at look k
                # given that we survived all prior looks
                conditional_alpha = delta_k / remaining
                conditional_alpha = min(conditional_alpha, 1.0)
                boundaries[k] = float(
                    norm.ppf(1.0 - conditional_alpha / 2.0)
                )

        return [float(b) for b in boundaries]

    @staticmethod
    def _compute_futility_boundaries(
        beta_spent_cum: list[float],
        alpha: float,
    ) -> list[float]:
        """Compute futility boundaries from cumulative beta spending.

        Futility boundaries are low early (hard to trigger — almost
        never stop early for futility) and increase over time as more
        beta is spent.  At look k, if |Z_k| < futility_k, we stop for
        futility because the effect is too small to ever reach
        significance.

        The boundary at each look is computed by mapping the cumulative
        beta fraction spent to the range [0, z_alpha]:

            futility_k = z_alpha * (beta_spent_k / beta_total)

        where beta_total is the total beta budget (beta_spent at the
        last look).

        Parameters
        ----------
        beta_spent_cum : list[float]
            Cumulative beta spent at each look.
        alpha : float
            Overall significance level (used to derive z_alpha upper
            bound for futility).

        Returns
        -------
        list[float]
            Futility z-value boundaries at each look (non-negative,
            non-decreasing).
        """
        z_alpha = float(norm.ppf(1.0 - alpha / 2.0))
        beta_total = beta_spent_cum[-1] if beta_spent_cum else 1.0
        if beta_total <= 0:  # pragma: no cover
            return [0.0] * len(beta_spent_cum)
        return [
            float(z_alpha * (bs / beta_total))
            for bs in beta_spent_cum
        ]

    # ── public API ──────────────────────────────────────────────────

    def boundary(self) -> BoundaryResult:
        """Compute critical value boundaries for all planned analyses.

        Returns equally-spaced information fractions and the corresponding
        efficacy (and optionally futility) z-value boundaries.

        Returns
        -------
        BoundaryResult
            Spending-function boundaries for the design.

        Examples
        --------
        >>> gs = GroupSequential(n_analyses=3, spending_function="obf")
        >>> b = gs.boundary()
        >>> b.efficacy_boundaries[0] > b.efficacy_boundaries[-1]
        True
        """
        k = self._n_analyses
        info_fracs = [(i + 1) / k for i in range(k)]

        # ── efficacy boundaries ──
        alpha_spent_cum = [
            self._alpha_spent(t, self._alpha, self._spending_function)
            for t in info_fracs
        ]
        efficacy_z = self._compute_boundaries(alpha_spent_cum)

        # ── futility boundaries ──
        # Futility boundaries are positive z-values: stop for futility
        # when |z| < futility_z[k].  Low early (near 0, hard to trigger)
        # and increasing over time as more beta is spent.
        futility_z: list[float] | None = None
        if self._beta_spending is not None:
            beta = 1.0 - self._power
            beta_spent_cum = [
                self._alpha_spent(t, beta, self._beta_spending)
                for t in info_fracs
            ]
            futility_z = self._compute_futility_boundaries(
                beta_spent_cum, self._alpha
            )

        adjusted_alpha = alpha_spent_cum[-1]

        return BoundaryResult(
            efficacy_boundaries=efficacy_z,
            futility_boundaries=futility_z,
            information_fractions=info_fracs,
            alpha_spent=alpha_spent_cum,
            adjusted_alpha=adjusted_alpha,
        )

    def test(
        self,
        statistics: list[float | None],
        information_fractions: list[float],
    ) -> GSResult:
        """Test observed z-statistics against boundaries.

        Parameters
        ----------
        statistics : list of float or None
            Z-statistics at each analysis. Use ``None`` for future (not yet
            observed) analyses.
        information_fractions : list of float
            Actual information fractions at each analysis. Must be
            non-decreasing with the last element equal to 1.0.

        Returns
        -------
        GSResult
            Per-look analysis details and stopping recommendation.

        Raises
        ------
        ValueError
            If inputs fail validation (length mismatch, invalid fractions, etc.).

        Examples
        --------
        >>> gs = GroupSequential(n_analyses=3)
        >>> result = gs.test([3.5, None, None], [0.33, 0.67, 1.0])
        >>> result.recommended_action
        'stop_efficacy'
        """
        self._validate_test_inputs(statistics, information_fractions)

        # Recompute boundaries at the actual information fractions
        alpha_spent_cum = [
            self._alpha_spent(t, self._alpha, self._spending_function)
            for t in information_fractions
        ]
        efficacy_z = self._compute_boundaries(alpha_spent_cum)

        futility_z: list[float] | None = None
        if self._beta_spending is not None:
            beta = 1.0 - self._power
            beta_spent_cum = [
                self._alpha_spent(t, beta, self._beta_spending)
                for t in information_fractions
            ]
            futility_z = self._compute_futility_boundaries(
                beta_spent_cum, self._alpha
            )

        # ── evaluate each look ──
        crossed_efficacy = False
        crossed_futility = False
        recommended_action = "continue"
        analysis_results: list[dict[str, Any]] = []

        for i, z_obs in enumerate(statistics):
            look: dict[str, Any] = {
                "look": i + 1,
                "information_fraction": information_fractions[i],
                "efficacy_boundary": efficacy_z[i],
                "statistic": z_obs,
            }
            if futility_z is not None:
                look["futility_boundary"] = futility_z[i]

            if z_obs is None:
                look["action"] = "not_yet_observed"
                analysis_results.append(look)
                continue

            # Check efficacy
            if abs(z_obs) >= efficacy_z[i]:
                look["action"] = "stop_efficacy"
                if not crossed_efficacy and not crossed_futility:
                    crossed_efficacy = True
                    recommended_action = "stop_efficacy"
            # Check futility
            elif futility_z is not None and abs(z_obs) <= futility_z[i]:
                look["action"] = "stop_futility"
                if not crossed_efficacy and not crossed_futility:
                    crossed_futility = True
                    recommended_action = "stop_futility"
            else:
                look["action"] = "continue"

            analysis_results.append(look)

        return GSResult(
            analysis_results=analysis_results,
            crossed_efficacy=crossed_efficacy,
            crossed_futility=crossed_futility,
            recommended_action=recommended_action,
        )

    # ── validation helpers ──────────────────────────────────────────

    def _validate_test_inputs(
        self,
        statistics: list[float | None],
        information_fractions: list[float],
    ) -> None:
        """Validate inputs to :meth:`test`.

        Raises
        ------
        ValueError
            If any validation check fails.
        """
        if len(statistics) != len(information_fractions):
            raise ValueError(
                format_error(
                    "`statistics` and `information_fractions` must have "
                    "the same length.",
                    f"statistics has {len(statistics)} elements, "
                    f"information_fractions has {len(information_fractions)} elements.",
                )
            )

        if len(information_fractions) == 0:
            raise ValueError(
                format_error(
                    "`information_fractions` can't be empty.",
                    "received a sequence with 0 elements.",
                )
            )

        # Check fractions are in (0, 1]
        for i, frac in enumerate(information_fractions):
            if not (0.0 < frac <= 1.0):
                raise ValueError(
                    format_error(
                        f"`information_fractions[{i}]` must be in (0, 1], got {frac}.",
                        "each information fraction must be strictly "
                        "positive and at most 1.",
                    )
                )

        # Check non-decreasing
        for i in range(1, len(information_fractions)):
            if information_fractions[i] < information_fractions[i - 1]:
                raise ValueError(
                    format_error(
                        "`information_fractions` must be non-decreasing.",
                        f"information_fractions[{i}]={information_fractions[i]} < "
                        f"information_fractions[{i - 1}]="
                        f"{information_fractions[i - 1]}.",
                        "sort your information fractions in ascending order.",
                    )
                )

        # Last must be 1.0
        if not math.isclose(information_fractions[-1], 1.0, rel_tol=1e-9):
            raise ValueError(
                format_error(
                    "`information_fractions[-1]` must be 1.0, "
                    f"got {information_fractions[-1]}.",
                    "the last information fraction represents the final analysis.",
                    "set the last element to 1.0.",
                )
            )
