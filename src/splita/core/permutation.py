"""PermutationTest — exact distribution-free hypothesis testing.

Permutes group labels to build a null distribution of the test
statistic, then computes an exact (Monte Carlo) p-value.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from splita._types import PermutationResult
from splita._validation import (
    check_array_like,
    check_in_range,
    check_is_integer,
    check_one_of,
    format_error,
)

ArrayLike = list | tuple | np.ndarray

_VALID_STATISTICS = ["mean_diff", "median_diff"]
_VALID_ALTERNATIVES = ["two-sided", "greater", "less"]


class PermutationTest:
    """Run a permutation (randomisation) test on two groups.

    Computes the observed test statistic, then permutes group labels
    ``n_permutations`` times to build a null distribution. The p-value
    is the fraction of permuted statistics at least as extreme as the
    observed one.

    Parameters
    ----------
    control : array-like
        Observations from the control group.
    treatment : array-like
        Observations from the treatment group.
    n_permutations : int, default 10000
        Number of random permutations.
    statistic : {'mean_diff', 'median_diff'}, default 'mean_diff'
        Test statistic to compute.
    alternative : {'two-sided', 'greater', 'less'}, default 'two-sided'
        Direction of the test.
    random_state : int, Generator, or None, default None
        Seed or RNG for reproducibility.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> ctrl = rng.normal(0, 1, 100)
    >>> trt = rng.normal(0.5, 1, 100)
    >>> result = PermutationTest(ctrl, trt, random_state=42).run()
    >>> result.significant
    True
    """

    def __init__(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
        *,
        n_permutations: int = 10000,
        statistic: Literal["mean_diff", "median_diff"] = "mean_diff",
        alternative: Literal["two-sided", "greater", "less"] = "two-sided",
        random_state: int | np.random.Generator | None = None,
    ):
        check_one_of(statistic, "statistic", _VALID_STATISTICS)
        check_one_of(alternative, "alternative", _VALID_ALTERNATIVES)
        check_is_integer(n_permutations, "n_permutations", min_value=100)

        self._control = check_array_like(control, "control", min_length=2)
        self._treatment = check_array_like(treatment, "treatment", min_length=2)
        self._n_permutations = int(n_permutations)
        self._statistic = statistic
        self._alternative = alternative
        self._alpha = 0.05  # default; could be made configurable

        if isinstance(random_state, np.random.Generator):
            self._rng = random_state
        else:
            self._rng = np.random.default_rng(random_state)

    def _compute_statistic(
        self, group_a: np.ndarray, group_b: np.ndarray
    ) -> float:
        """Compute test statistic: treatment - control."""
        if self._statistic == "mean_diff":
            return float(np.mean(group_b) - np.mean(group_a))
        else:  # median_diff
            return float(np.median(group_b) - np.median(group_a))

    def run(self) -> PermutationResult:
        """Execute the permutation test.

        Returns
        -------
        PermutationResult
            Test result including observed statistic, p-value, and
            null distribution summary.
        """
        n_c = len(self._control)
        pooled = np.concatenate([self._control, self._treatment])
        n_total = len(pooled)

        observed = self._compute_statistic(self._control, self._treatment)

        null_stats = np.empty(self._n_permutations)
        for i in range(self._n_permutations):
            perm = self._rng.permutation(n_total)
            perm_ctrl = pooled[perm[:n_c]]
            perm_trt = pooled[perm[n_c:]]
            null_stats[i] = self._compute_statistic(perm_ctrl, perm_trt)

        # Compute p-value based on alternative
        if self._alternative == "two-sided":
            pvalue = float(np.mean(np.abs(null_stats) >= np.abs(observed)))
        elif self._alternative == "greater":
            pvalue = float(np.mean(null_stats >= observed))
        else:  # less
            pvalue = float(np.mean(null_stats <= observed))

        # Ensure p-value is at least 1 / (n_permutations + 1) to avoid exact 0
        pvalue = max(pvalue, 1.0 / (self._n_permutations + 1))

        significant = pvalue < self._alpha

        return PermutationResult(
            observed_statistic=observed,
            pvalue=pvalue,
            significant=significant,
            n_permutations=self._n_permutations,
            alpha=self._alpha,
            null_distribution_mean=float(np.mean(null_stats)),
            null_distribution_std=float(np.std(null_stats)),
        )
