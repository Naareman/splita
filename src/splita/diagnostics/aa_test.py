"""AATest — validate randomisation and metrics before running the real experiment."""

from __future__ import annotations

import math

import numpy as np

from splita._types import AATestResult
from splita._utils import ensure_rng
from splita._validation import (
    check_array_like,
    check_in_range,
    check_is_integer,
)
from splita.core.experiment import Experiment

ArrayLike = list | tuple | np.ndarray


class AATest:
    """Validate that randomisation and metrics are working correctly.

    Randomly splits ``data`` into two equal groups ``n_simulations`` times,
    runs :class:`~splita.Experiment` on each split, and checks that the
    false positive rate is consistent with the chosen ``alpha``.

    Parameters
    ----------
    n_simulations : int, default 1000
        Number of random A/A splits to run.
    alpha : float, default 0.05
        Significance level for each per-split test.
    random_state : int, Generator, or None, default None
        Seed for reproducibility.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.default_rng(0).binomial(1, 0.10, size=2000)
    >>> result = AATest(n_simulations=200, random_state=42).run(data)
    >>> result.passed
    True
    """

    def __init__(
        self,
        *,
        n_simulations: int = 1000,
        alpha: float = 0.05,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        check_is_integer(n_simulations, "n_simulations", min_value=10)
        check_in_range(
            alpha,
            "alpha",
            0.0,
            1.0,
            hint="typical values are 0.05, 0.01, or 0.10",
        )
        self._n_simulations = int(n_simulations)
        self._alpha = alpha
        self._rng = ensure_rng(random_state)

    def run(
        self,
        data: ArrayLike,
        metric: str = "auto",
    ) -> AATestResult:
        """Run the A/A test.

        Parameters
        ----------
        data : array-like
            The full dataset to validate.  Will be randomly split in half
            for each simulation.
        metric : {'auto', 'conversion', 'continuous'}, default 'auto'
            Metric type passed to :class:`~splita.Experiment`.

        Returns
        -------
        AATestResult
            Frozen dataclass with false positive rate and diagnostics.

        Raises
        ------
        ValueError
            If *data* has fewer than 4 elements (need at least 2 per group).
        """
        arr = check_array_like(data, "data", min_length=4)
        n = len(arr)
        half = n // 2

        p_values: list[float] = []
        n_significant = 0

        for _ in range(self._n_simulations):
            shuffled = self._rng.permutation(arr)
            group_a = shuffled[:half]
            group_b = shuffled[half : half * 2]

            try:
                exp = Experiment(group_a, group_b, metric=metric, alpha=self._alpha)
                res = exp.run()
                p_values.append(res.pvalue)
                if res.significant:
                    n_significant += 1
            except (ValueError, RuntimeError):
                # Skip failed runs (e.g. zero-variance splits)
                continue

        if len(p_values) == 0:
            return AATestResult(
                false_positive_rate=0.0,
                expected_rate=self._alpha,
                passed=False,
                p_values=[],
                n_simulations=self._n_simulations,
                message="All simulations failed. Check your data.",
            )

        fp_rate = n_significant / len(p_values)

        # Check if FP rate is within expected bounds: alpha +/- 2*SE
        se = math.sqrt(self._alpha * (1 - self._alpha) / len(p_values))
        lower_bound = self._alpha - 2 * se
        upper_bound = self._alpha + 2 * se
        passed = lower_bound <= fp_rate <= upper_bound

        if passed:
            message = (
                f"A/A test passed. False positive rate {fp_rate:.3f} is within "
                f"expected bounds [{lower_bound:.3f}, {upper_bound:.3f}]."
            )
        else:
            message = (
                f"A/A test failed. False positive rate {fp_rate:.3f} is outside "
                f"expected bounds [{lower_bound:.3f}, {upper_bound:.3f}]. "
                "This may indicate a problem with your randomisation or metric."
            )

        return AATestResult(
            false_positive_rate=fp_rate,
            expected_rate=self._alpha,
            passed=passed,
            p_values=p_values,
            n_simulations=len(p_values),
            message=message,
        )
