"""PowerSimulation — Monte Carlo power analysis for complex experiment designs.

Useful when closed-form formulas don't exist: CUPED-adjusted metrics,
stratified experiments, ratio metrics, or custom data-generating processes.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from splita._types import PowerSimulationResult
from splita._utils import ensure_rng
from splita._validation import check_in_range, check_is_integer, format_error
from splita.core.experiment import Experiment


def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - spread), min(1.0, centre + spread)


class PowerSimulation:
    """Monte Carlo power analysis via simulation.

    Parameters
    ----------
    n_simulations : int, default 1000
        Number of Monte Carlo replications.
    alpha : float, default 0.05
        Significance level for each simulated test.
    random_state : int, Generator, or None, default None
        Seed for reproducibility.

    Examples
    --------
    >>> import numpy as np
    >>> from splita import PowerSimulation
    >>> result = PowerSimulation.for_proportion(
    ...     0.10, 0.02, 4000, n_simulations=500, random_state=42
    ... )
    >>> 0.5 < result.power < 1.0
    True
    """

    def __init__(
        self,
        *,
        n_simulations: int = 1000,
        alpha: float = 0.05,
        random_state: int | np.random.Generator | None = None,
    ):
        check_is_integer(
            n_simulations,
            "n_simulations",
            min_value=10,
            hint="use at least 100 simulations for reliable estimates.",
        )
        check_in_range(
            alpha,
            "alpha",
            0.0,
            1.0,
            hint="typical values are 0.05, 0.01, or 0.10.",
        )

        self._n_simulations = int(n_simulations)
        self._alpha = alpha
        self._rng = ensure_rng(random_state)

    def run(
        self,
        dgp: Callable[[np.random.Generator], tuple[np.ndarray, np.ndarray]],
        n_per_variant: int,
        method: str = "auto",
    ) -> PowerSimulationResult:
        """Run the Monte Carlo power simulation.

        Parameters
        ----------
        dgp : callable
            Data-generating process. Takes an ``np.random.Generator`` and
            returns ``(control, treatment)`` arrays of size *n_per_variant*.
        n_per_variant : int
            Sample size per variant (passed to the DGP for documentation;
            the DGP itself controls actual array sizes).
        method : str, default 'auto'
            Statistical method passed to :class:`~splita.Experiment`.

        Returns
        -------
        PowerSimulationResult
            Frozen dataclass with power estimate, confidence interval, and
            summary statistics.

        Raises
        ------
        ValueError
            If *n_per_variant* is not a positive integer.
        """
        check_is_integer(
            n_per_variant,
            "n_per_variant",
            min_value=2,
            hint="need at least 2 observations per variant.",
        )

        significant_count = 0
        pvalues: list[float] = []
        lifts: list[float] = []

        for _ in range(self._n_simulations):
            ctrl, trt = dgp(self._rng)
            result = Experiment(
                ctrl,
                trt,
                method=method,
                alpha=self._alpha,
            ).run()
            significant_count += int(result.significant)
            pvalues.append(result.pvalue)
            lifts.append(result.lift)

        power = significant_count / self._n_simulations
        ci_lower, ci_upper = _wilson_ci(power, self._n_simulations)

        return PowerSimulationResult(
            power=power,
            rejection_rate=power,
            n_simulations=self._n_simulations,
            n_per_variant=int(n_per_variant),
            alpha=self._alpha,
            mean_effect=float(np.mean(lifts)),
            mean_pvalue=float(np.mean(pvalues)),
            ci_power_lower=ci_lower,
            ci_power_upper=ci_upper,
        )

    # ── convenience class methods ────────────────────────────────────

    @classmethod
    def for_proportion(
        cls,
        baseline: float,
        mde: float,
        n_per_variant: int,
        **kwargs,
    ) -> PowerSimulationResult:
        """Simulate power for a proportion (conversion rate) test.

        Parameters
        ----------
        baseline : float
            Baseline conversion rate, in (0, 1).
        mde : float
            Minimum detectable effect (absolute), e.g. 0.02 for +2pp.
        n_per_variant : int
            Sample size per variant.
        **kwargs
            Passed to :class:`PowerSimulation` constructor (e.g.
            ``n_simulations``, ``alpha``, ``random_state``).

        Returns
        -------
        PowerSimulationResult
        """
        check_in_range(baseline, "baseline", 0.0, 1.0)
        treatment_rate = baseline + mde
        if treatment_rate <= 0.0 or treatment_rate >= 1.0:
            raise ValueError(
                format_error(
                    f"`baseline + mde` must be in (0, 1), got {treatment_rate}.",
                    f"baseline={baseline}, mde={mde}.",
                    "ensure the treatment rate is a valid probability.",
                )
            )

        def dgp(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
            ctrl = rng.binomial(1, baseline, n_per_variant).astype(float)
            trt = rng.binomial(1, treatment_rate, n_per_variant).astype(float)
            return ctrl, trt

        return cls(**kwargs).run(dgp, n_per_variant)

    @classmethod
    def for_mean(
        cls,
        baseline_mean: float,
        baseline_std: float,
        mde: float,
        n_per_variant: int,
        **kwargs,
    ) -> PowerSimulationResult:
        """Simulate power for a continuous metric test.

        Parameters
        ----------
        baseline_mean : float
            Mean of the control group.
        baseline_std : float
            Standard deviation of the control group.
        mde : float
            Minimum detectable effect (absolute difference in means).
        n_per_variant : int
            Sample size per variant.
        **kwargs
            Passed to :class:`PowerSimulation` constructor.

        Returns
        -------
        PowerSimulationResult
        """
        if baseline_std <= 0:
            raise ValueError(
                format_error(
                    f"`baseline_std` must be > 0, got {baseline_std}.",
                    "standard deviation must be strictly positive.",
                )
            )

        def dgp(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
            ctrl = rng.normal(baseline_mean, baseline_std, n_per_variant)
            trt = rng.normal(baseline_mean + mde, baseline_std, n_per_variant)
            return ctrl, trt

        return cls(**kwargs).run(dgp, n_per_variant)
