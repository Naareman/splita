"""Dilution analysis for triggered experiments.

When only a fraction of users are triggered (exposed to the treatment),
the triggered effect can be diluted back to the full ITT population
(Deng et al. 2015).
"""

from __future__ import annotations

from scipy.stats import norm

from splita._types import DilutionResult
from splita._validation import format_error

__all__ = ["DilutionAnalysis"]


class DilutionAnalysis:
    """Dilute a triggered treatment effect back to the full population.

    Given an effect measured among triggered users and the trigger rate,
    compute the intent-to-treat (ITT) effect for the full experiment
    population.

    Parameters
    ----------
    alpha : float, default 0.05
        Significance level for inference.

    Examples
    --------
    >>> da = DilutionAnalysis()
    >>> r = da.dilute(triggered_effect=2.0, triggered_se=0.5, trigger_rate=0.3)
    >>> abs(r.diluted_ate - 0.6) < 0.01
    True
    """

    def __init__(self, *, alpha: float = 0.05) -> None:
        if not 0 < alpha < 1:
            raise ValueError(
                format_error(
                    "`alpha` must be in (0, 1), got {}.".format(alpha),
                    "alpha represents the significance level.",
                    "typical values are 0.05, 0.01, or 0.10.",
                )
            )
        self._alpha = alpha

    def dilute(
        self,
        triggered_effect: float,
        triggered_se: float,
        trigger_rate: float,
    ) -> DilutionResult:
        """Dilute a triggered effect to the full population.

        Parameters
        ----------
        triggered_effect : float
            Estimated treatment effect among triggered (exposed) users.
        triggered_se : float
            Standard error of the triggered effect estimate.
        trigger_rate : float
            Fraction of users who were triggered (in (0, 1]).

        Returns
        -------
        DilutionResult
            Frozen dataclass with diluted and triggered estimates.

        Raises
        ------
        ValueError
            If ``trigger_rate`` is not in (0, 1] or ``triggered_se`` is negative.
        """
        if not 0 < trigger_rate <= 1:
            raise ValueError(
                format_error(
                    "`trigger_rate` must be in (0, 1], got {}.".format(trigger_rate),
                    "trigger_rate is the fraction of users who were exposed.",
                    "use the ratio of triggered to total assigned users.",
                )
            )

        if triggered_se < 0:
            raise ValueError(
                format_error(
                    "`triggered_se` must be >= 0, got {}.".format(triggered_se),
                    "standard errors cannot be negative.",
                )
            )

        # ITT = trigger_rate * triggered_effect
        diluted_ate = trigger_rate * triggered_effect
        diluted_se = trigger_rate * triggered_se

        if diluted_se > 0:
            z = diluted_ate / diluted_se
            diluted_pvalue = float(2.0 * norm.sf(abs(z)))
        else:
            diluted_pvalue = 1.0 if diluted_ate == 0 else 0.0

        return DilutionResult(
            diluted_ate=diluted_ate,
            diluted_se=diluted_se,
            diluted_pvalue=diluted_pvalue,
            trigger_rate=trigger_rate,
            triggered_ate=triggered_effect,
        )
