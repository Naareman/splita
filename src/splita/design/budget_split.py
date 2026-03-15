"""Budget-split experiment design (Liu et al. LinkedIn KDD 2021).

Creates two independent sub-marketplaces by splitting budgets, eliminating
cannibalization bias that occurs when treatment and control compete for the
same resources.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ttest_ind

from splita._types import BudgetSplitResult
from splita._validation import (
    check_array_like,
    check_in_range,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


class BudgetSplitDesign:
    """Design and analyse a budget-split experiment.

    Splits available budgets into treatment and control sub-marketplaces
    so that they cannot cannibalize each other.  This eliminates the bias
    that arises when treatment and control units compete for the same
    limited resources (e.g., ad impressions, job listings).

    Parameters
    ----------
    alpha : float, default 0.05
        Significance level for hypothesis testing.

    Examples
    --------
    >>> import numpy as np
    >>> design = BudgetSplitDesign()
    >>> budgets = np.array([100, 200, 150, 300, 250])
    >>> result = design.design(budgets, split_fraction=0.6)
    >>> result.treatment_budget > 0
    True
    """

    def __init__(self, *, alpha: float = 0.05) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(
                format_error(
                    f"`alpha` must be in (0, 1), got {alpha}.",
                    "alpha controls the significance level.",
                    "typical values are 0.05, 0.01, or 0.10.",
                )
            )
        self._alpha = alpha
        self._split_fraction: float | None = None
        self._treatment_budget: float = 0.0
        self._control_budget: float = 0.0

    def design(
        self,
        budgets: ArrayLike,
        split_fraction: float = 0.5,
    ) -> BudgetSplitResult:
        """Create a budget split design.

        Parameters
        ----------
        budgets : array-like
            Available budgets for each unit (advertiser, campaign, etc.).
        split_fraction : float, default 0.5
            Fraction of total budget allocated to the treatment market.
            Must be in (0, 1).

        Returns
        -------
        BudgetSplitResult
            Design with budget allocations.  ``ate`` and ``pvalue`` are
            set to 0.0 and 1.0 respectively (no data yet).

        Raises
        ------
        ValueError
            If *budgets* is empty or *split_fraction* is out of range.
        """
        b = check_array_like(budgets, "budgets", min_length=1)

        check_in_range(
            split_fraction,
            "split_fraction",
            0.0,
            1.0,
            hint="typical values are 0.3, 0.5, or 0.7.",
        )

        if np.any(b < 0):
            raise ValueError(
                format_error(
                    "`budgets` must contain only non-negative values.",
                    f"found {int(np.sum(b < 0))} negative budget(s).",
                    "budgets represent available resources and cannot be negative.",
                )
            )

        total_budget = float(np.sum(b))
        self._treatment_budget = total_budget * split_fraction
        self._control_budget = total_budget * (1.0 - split_fraction)
        self._split_fraction = split_fraction

        return BudgetSplitResult(
            ate=0.0,
            pvalue=1.0,
            significant=False,
            treatment_budget=self._treatment_budget,
            control_budget=self._control_budget,
            cannibalization_free=True,
        )

    def analyze(
        self,
        outcomes_treatment_market: ArrayLike,
        outcomes_control_market: ArrayLike,
    ) -> BudgetSplitResult:
        """Analyse outcomes from the two sub-marketplaces.

        Parameters
        ----------
        outcomes_treatment_market : array-like
            Outcomes from the treatment sub-marketplace.
        outcomes_control_market : array-like
            Outcomes from the control sub-marketplace.

        Returns
        -------
        BudgetSplitResult
            Treatment effect estimate with significance test.

        Raises
        ------
        ValueError
            If either outcome array is too short.
        """
        y_t = check_array_like(
            outcomes_treatment_market,
            "outcomes_treatment_market",
            min_length=2,
        )
        y_c = check_array_like(
            outcomes_control_market,
            "outcomes_control_market",
            min_length=2,
        )

        ate = float(np.mean(y_t) - np.mean(y_c))

        # Welch's t-test
        _stat, pvalue = ttest_ind(y_t, y_c, equal_var=False)
        pvalue = float(pvalue)

        treatment_budget = (
            self._treatment_budget if self._treatment_budget > 0 else float(np.sum(y_t))
        )
        control_budget = self._control_budget if self._control_budget > 0 else float(np.sum(y_c))

        return BudgetSplitResult(
            ate=ate,
            pvalue=pvalue,
            significant=pvalue < self._alpha,
            treatment_budget=treatment_budget,
            control_budget=control_budget,
            cannibalization_free=True,
        )
