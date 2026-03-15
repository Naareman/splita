"""Response-Adaptive Randomization (RAR) for multi-arm experiments.

Updates allocation probabilities based on accumulating reward data,
directing more traffic to better-performing arms while maintaining
a minimum allocation floor.
"""

from __future__ import annotations

import numpy as np

from splita._types import RARResult
from splita._utils import ensure_rng
from splita._validation import (
    check_in_range,
    check_is_integer,
    check_one_of,
    format_error,
)


class ResponseAdaptiveRandomization:
    """Response-Adaptive Randomization for multi-arm experiments.

    Dynamically adjusts allocation probabilities to favour
    better-performing arms while maintaining ethical minimums.

    Parameters
    ----------
    n_arms : int, default 2
        Number of treatment arms.
    method : ``"bayesian"`` or ``"urn"``, default ``"bayesian"``
        Adaptation method.
    min_allocation : float, default 0.1
        Minimum allocation probability per arm.
    random_state : int, np.random.Generator, or None, default None
        Random state for reproducibility.

    Examples
    --------
    >>> rar = ResponseAdaptiveRandomization(n_arms=2, random_state=42)
    >>> rar.update(0, 1.0)
    >>> rar.update(1, 0.0)
    >>> rar.get_allocation()[0] > rar.get_allocation()[1]
    True
    """

    def __init__(
        self,
        n_arms: int = 2,
        *,
        method: str = "bayesian",
        min_allocation: float = 0.1,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        check_is_integer(n_arms, "n_arms", min_value=2)
        check_one_of(method, "method", ["bayesian", "urn"])
        check_in_range(
            min_allocation,
            "min_allocation",
            0.0,
            1.0,
            low_inclusive=True,
            hint="use 0.0 for no minimum or up to 1/n_arms for equal allocation.",
        )
        n_arms = int(n_arms)
        if min_allocation * n_arms > 1.0:
            raise ValueError(
                format_error(
                    "`min_allocation` * `n_arms` must be <= 1.0.",
                    f"min_allocation={min_allocation} * n_arms={n_arms} = "
                    f"{min_allocation * n_arms:.2f} > 1.0.",
                    hint="reduce min_allocation or n_arms.",
                )
            )

        self._n_arms = n_arms
        self._method = method
        self._min_allocation = min_allocation
        self._rng = ensure_rng(random_state)

        # Bayesian: Beta(alpha, beta) prior for each arm
        self._alpha_params = np.ones(n_arms)  # successes + 1
        self._beta_params = np.ones(n_arms)  # failures + 1

        # Urn: counts per arm
        self._urn_counts = np.ones(n_arms, dtype=float)

        # Track observations
        self._rewards: list[list[float]] = [[] for _ in range(n_arms)]

    def update(self, arm: int, reward: float) -> None:
        """Record an observation for an arm.

        Parameters
        ----------
        arm : int
            Index of the arm (0-indexed).
        reward : float
            Observed reward value.

        Raises
        ------
        ValueError
            If arm index is out of range.
        """
        if not (0 <= arm < self._n_arms):
            raise ValueError(
                format_error(
                    f"`arm` must be in [0, {self._n_arms - 1}], got {arm}.",
                    hint=f"valid arms are 0 through {self._n_arms - 1}.",
                )
            )

        self._rewards[arm].append(float(reward))

        if self._method == "bayesian":
            # Update Beta distribution parameters
            # Treat reward as success probability
            if reward > 0:
                self._alpha_params[arm] += reward
            else:
                self._beta_params[arm] += abs(reward) if reward < 0 else 1.0
        else:  # urn
            # Add balls proportional to reward
            self._urn_counts[arm] += max(reward, 0.0) + 0.01  # small constant to avoid zero

    def get_allocation(self) -> list[float]:
        """Get current allocation probabilities.

        Returns
        -------
        list of float
            Allocation probability for each arm, summing to 1.0.
        """
        if self._method == "bayesian":
            return self._bayesian_allocation()
        else:
            return self._urn_allocation()

    def _bayesian_allocation(self) -> list[float]:
        """Compute allocation proportional to probability of being best."""
        n_samples = 10000
        samples = np.zeros((n_samples, self._n_arms))
        for arm in range(self._n_arms):
            samples[:, arm] = self._rng.beta(
                self._alpha_params[arm], self._beta_params[arm], n_samples
            )

        # Probability each arm is best
        best_arm = np.argmax(samples, axis=1)
        probs = np.zeros(self._n_arms)
        for arm in range(self._n_arms):
            probs[arm] = float(np.mean(best_arm == arm))

        return self._apply_min_allocation(probs)

    def _urn_allocation(self) -> list[float]:
        """Compute allocation proportional to urn counts."""
        total = float(np.sum(self._urn_counts))
        if total == 0:
            probs = np.ones(self._n_arms) / self._n_arms
        else:
            probs = self._urn_counts / total
        return self._apply_min_allocation(probs)

    def _apply_min_allocation(self, probs: np.ndarray) -> list[float]:
        """Enforce minimum allocation and normalize."""
        probs = np.maximum(probs, self._min_allocation)
        probs = probs / np.sum(probs)
        return [float(p) for p in probs]

    def recommend(self) -> int:
        """Recommend which arm to assign next.

        Returns
        -------
        int
            Index of the recommended arm.
        """
        alloc = self.get_allocation()
        return int(self._rng.choice(self._n_arms, p=alloc))

    def result(self) -> RARResult:
        """Get current state summary.

        Returns
        -------
        RARResult
            Current allocations and observation counts.
        """
        allocations = self.get_allocation()
        n_per_arm = [len(r) for r in self._rewards]
        total_observations = sum(n_per_arm)
        best_arm = int(np.argmax(allocations))

        return RARResult(
            allocations=allocations,
            n_per_arm=n_per_arm,
            best_arm=best_arm,
            total_observations=total_observations,
        )
