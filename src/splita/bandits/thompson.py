"""Thompson Sampling for multi-armed bandits.

Balances exploration and exploitation by sampling from posterior
distributions.  Supports Bernoulli, Gaussian (Normal-Inverse-Gamma),
and Poisson (Gamma-Poisson) reward models.

Reference
---------
Russo, Van Roy, Kazerouni, Osband & Wen (2018).
"A Tutorial on Thompson Sampling." *Foundations and Trends in ML*.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from splita._types import BanditResult
from splita._utils import ensure_rng
from splita._validation import check_is_integer, check_one_of, format_error

_LIKELIHOODS = ["bernoulli", "gaussian", "poisson"]
_STOPPING_RULES = ["expected_loss", "prob_best", "n_samples"]

_DEFAULT_PRIORS: dict[str, dict[str, float]] = {
    "bernoulli": {"alpha": 1.0, "beta": 1.0},
    "gaussian": {"mu": 0.0, "kappa": 1.0, "alpha": 1.0, "beta": 1.0},
    "poisson": {"alpha": 1.0, "beta": 1.0},
}


class ThompsonSampler:
    """Thompson Sampling multi-armed bandit.

    Parameters
    ----------
    n_arms : int
        Number of arms (>= 2).
    likelihood : {'bernoulli', 'gaussian', 'poisson'}, default 'bernoulli'
        Reward model.

        - ``'bernoulli'``: Beta-Binomial conjugate (binary rewards).
        - ``'gaussian'``: Normal-Inverse-Gamma conjugate (continuous rewards).
        - ``'poisson'``: Gamma-Poisson conjugate (count rewards).
    prior : dict or None, default None
        Prior hyperparameters applied to every arm.  ``None`` uses
        non-informative defaults for the chosen likelihood.
    stopping_rule : {'expected_loss', 'prob_best', 'n_samples'}, default 'expected_loss'
        Criterion for recommending early stopping.
    stopping_threshold : float, default 0.01
        Threshold for the stopping rule.
    min_samples : int, default 100
        Minimum total observations before evaluating the stopping rule.
    random_state : int, Generator, or None, default None
        Seed for reproducibility.

    Examples
    --------
    >>> ts = ThompsonSampler(2, random_state=42)
    >>> for _ in range(200):
    ...     arm = ts.recommend()
    ...     reward = 1 if (arm == 0 and np.random.random() < 0.8) else 0
    ...     ts.update(arm, reward)
    >>> ts.result().current_best_arm
    0
    """

    def __init__(
        self,
        n_arms: int,
        *,
        likelihood: Literal["bernoulli", "gaussian", "poisson"] = "bernoulli",
        prior: dict | None = None,
        stopping_rule: Literal["expected_loss", "prob_best", "n_samples"] = "expected_loss",
        stopping_threshold: float = 0.01,
        min_samples: int = 100,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        # ── validate ──
        check_is_integer(n_arms, "n_arms", min_value=2)
        check_one_of(likelihood, "likelihood", _LIKELIHOODS)
        check_one_of(stopping_rule, "stopping_rule", _STOPPING_RULES)
        check_is_integer(min_samples, "min_samples", min_value=0)
        self._validate_stopping_threshold(stopping_rule, stopping_threshold)

        self._n_arms = int(n_arms)
        self._likelihood = likelihood
        self._stopping_rule = stopping_rule
        self._stopping_threshold = float(stopping_threshold)
        self._min_samples = int(min_samples)
        self._rng = ensure_rng(random_state)

        # ── per-arm state ──
        self._n_pulls = np.zeros(self._n_arms, dtype=int)
        self._total_reward = 0.0

        prior = prior if prior is not None else _DEFAULT_PRIORS[likelihood]

        if likelihood == "bernoulli":
            self._alpha = np.full(self._n_arms, prior["alpha"], dtype=float)
            self._beta = np.full(self._n_arms, prior["beta"], dtype=float)
        elif likelihood == "gaussian":
            self._mu = np.full(self._n_arms, prior["mu"], dtype=float)
            self._kappa = np.full(self._n_arms, prior["kappa"], dtype=float)
            self._alpha = np.full(self._n_arms, prior["alpha"], dtype=float)
            self._beta = np.full(self._n_arms, prior["beta"], dtype=float)
        else:  # poisson
            self._alpha = np.full(self._n_arms, prior["alpha"], dtype=float)
            self._beta = np.full(self._n_arms, prior["beta"], dtype=float)

    # ─── public API ───────────────────────────────────────────────

    def update(self, arm: int, reward: float) -> None:
        """Update the posterior for *arm* with an observed *reward*.

        Parameters
        ----------
        arm : int
            Arm index in ``[0, n_arms)``.
        reward : float
            Observed reward value.

        Raises
        ------
        ValueError
            If *arm* is out of range or *reward* is invalid for the
            configured likelihood.
        """
        self._validate_arm(arm)
        self._validate_reward(reward)

        if self._likelihood == "bernoulli":
            self._alpha[arm] += reward
            self._beta[arm] += 1.0 - reward
        elif self._likelihood == "gaussian":
            mu0 = self._mu[arm]
            kappa0 = self._kappa[arm]
            alpha0 = self._alpha[arm]
            beta0 = self._beta[arm]

            kappa_n = kappa0 + 1.0
            self._mu[arm] = (kappa0 * mu0 + reward) / kappa_n
            self._alpha[arm] = alpha0 + 0.5
            self._beta[arm] = beta0 + 0.5 * kappa0 * (reward - mu0) ** 2 / kappa_n
            self._kappa[arm] = kappa_n
        else:  # poisson
            self._alpha[arm] += reward
            self._beta[arm] += 1.0

        self._n_pulls[arm] += 1
        self._total_reward += reward

    def recommend(self) -> int:
        """Sample from each arm's posterior and return the winning arm.

        Returns
        -------
        int
            Index of the arm with the highest posterior sample.
        """
        samples = self._sample_posteriors(1)  # shape (1, n_arms)
        return int(np.argmax(samples[0]))

    def result(self) -> BanditResult:
        """Return the current bandit state as a :class:`BanditResult`.

        Returns
        -------
        BanditResult
            Frozen dataclass with posterior summaries and stopping info.
        """
        n_mc = 10_000
        samples = self._sample_posteriors(n_mc)  # (n_mc, n_arms)

        # prob_best: fraction of MC draws where each arm wins
        winners = np.argmax(samples, axis=1)
        prob_best = [float(np.mean(winners == i)) for i in range(self._n_arms)]

        # expected_loss: E[max(others) - this | choosing this]
        best_per_draw = np.max(samples, axis=1, keepdims=True)  # (n_mc, 1)
        losses = best_per_draw - samples  # (n_mc, n_arms)
        expected_loss = [float(np.mean(losses[:, i])) for i in range(self._n_arms)]

        # arm means & credible intervals
        arm_means = self._posterior_means()
        arm_ci = self._posterior_credible_intervals(samples)

        # stopping
        should_stop = self._evaluate_stopping(prob_best, expected_loss)

        # best arm = lowest expected loss
        current_best = int(np.argmin(expected_loss))

        return BanditResult(
            n_pulls_per_arm=self._n_pulls.tolist(),
            prob_best=prob_best,
            expected_loss=expected_loss,
            current_best_arm=current_best,
            should_stop=should_stop,
            total_reward=self._total_reward,
            cumulative_regret=None,
            arm_means=arm_means,
            arm_credible_intervals=arm_ci,
        )

    # ─── posterior sampling ───────────────────────────────────────

    def _sample_posteriors(self, n: int) -> np.ndarray:
        """Draw *n* samples from each arm's posterior.  Returns (n, n_arms)."""
        samples = np.empty((n, self._n_arms))

        if self._likelihood == "bernoulli":
            for i in range(self._n_arms):
                samples[:, i] = self._rng.beta(self._alpha[i], self._beta[i], size=n)

        elif self._likelihood == "gaussian":
            for i in range(self._n_arms):
                # NIG posterior: sample variance from Inv-Gamma, then mean
                precision = self._rng.gamma(self._alpha[i], 1.0 / self._beta[i], size=n)
                # precision = 1/sigma^2; avoid zero
                precision = np.maximum(precision, 1e-12)
                sigma = 1.0 / np.sqrt(precision)
                samples[:, i] = self._rng.normal(self._mu[i], sigma / np.sqrt(self._kappa[i]))

        else:  # poisson
            for i in range(self._n_arms):
                samples[:, i] = self._rng.gamma(self._alpha[i], 1.0 / self._beta[i], size=n)

        return samples

    def _posterior_means(self) -> list[float]:
        """Compute posterior means for each arm."""
        if self._likelihood == "bernoulli":
            return [
                float(self._alpha[i] / (self._alpha[i] + self._beta[i]))
                for i in range(self._n_arms)
            ]
        elif self._likelihood == "gaussian":
            return [float(self._mu[i]) for i in range(self._n_arms)]
        else:  # poisson
            return [float(self._alpha[i] / self._beta[i]) for i in range(self._n_arms)]

    def _posterior_credible_intervals(self, samples: np.ndarray) -> list[tuple[float, float]]:
        """95% credible intervals from MC samples."""
        return [
            (
                float(np.percentile(samples[:, i], 2.5)),
                float(np.percentile(samples[:, i], 97.5)),
            )
            for i in range(self._n_arms)
        ]

    # ─── stopping ─────────────────────────────────────────────────

    def _evaluate_stopping(
        self,
        prob_best: list[float],
        expected_loss: list[float],
    ) -> bool:
        """Evaluate the configured stopping rule."""
        total_samples = int(np.sum(self._n_pulls))

        if total_samples < self._min_samples:
            return False

        if self._stopping_rule == "expected_loss":
            return min(expected_loss) < self._stopping_threshold
        elif self._stopping_rule == "prob_best":
            return max(prob_best) > self._stopping_threshold
        else:  # n_samples
            return total_samples >= self._stopping_threshold

    # ─── validation ───────────────────────────────────────────────

    def _validate_arm(self, arm: int) -> None:
        """Raise if *arm* is not a valid index."""
        if not isinstance(arm, (int, np.integer)):
            raise TypeError(
                format_error(
                    f"`arm` must be an integer, got type {type(arm).__name__}.",
                )
            )
        if arm < 0 or arm >= self._n_arms:
            raise ValueError(
                format_error(
                    f"`arm` must be in [0, {self._n_arms}), got {arm}.",
                    f"valid arm indices are 0..{self._n_arms - 1}.",
                    "check that arm index matches your n_arms configuration.",
                )
            )

    def _validate_reward(self, reward: float) -> None:
        """Raise if *reward* is invalid for the configured likelihood."""
        if self._likelihood == "bernoulli":
            if reward not in (0, 1, 0.0, 1.0):
                raise ValueError(
                    format_error(
                        f"`reward` must be 0 or 1 for Bernoulli likelihood, got {reward}.",
                        "Bernoulli rewards are binary (success/failure).",
                        "use 1 for success and 0 for failure.",
                    )
                )
        elif self._likelihood == "gaussian":
            if math.isnan(reward) or math.isinf(reward):
                raise ValueError(
                    format_error(
                        f"`reward` must be finite, got {reward}.",
                        hint="check for missing or infinite values.",
                    )
                )
        elif self._likelihood == "poisson":
            if reward < 0:
                raise ValueError(
                    format_error(
                        f"`reward` must be >= 0 for Poisson likelihood, got {reward}.",
                        "Poisson rewards are non-negative counts.",
                        "ensure reward values are non-negative integers.",
                    )
                )
            if reward != int(reward):
                raise ValueError(
                    format_error(
                        f"`reward` must be integer-valued for Poisson likelihood, got {reward}.",
                        "Poisson rewards represent counts.",
                        "round or truncate to the nearest integer.",
                    )
                )

    @staticmethod
    def _validate_stopping_threshold(stopping_rule: str, stopping_threshold: float) -> None:
        """Validate that *stopping_threshold* is sensible for the rule."""
        if stopping_rule == "expected_loss":
            if stopping_threshold <= 0:
                raise ValueError(
                    format_error(
                        "`stopping_threshold` must be > 0 for 'expected_loss' "
                        f"rule, got {stopping_threshold}.",
                        hint="typical values are 0.001 to 0.05.",
                    )
                )
        elif stopping_rule == "prob_best":
            if not (0 < stopping_threshold < 1):
                raise ValueError(
                    format_error(
                        "`stopping_threshold` must be in (0, 1) for "
                        f"'prob_best' rule, got {stopping_threshold}.",
                        hint="typical values are 0.90 to 0.99.",
                    )
                )
        elif stopping_rule == "n_samples" and stopping_threshold <= 0:
            raise ValueError(
                format_error(
                    "`stopping_threshold` must be > 0 for 'n_samples' "
                    f"rule, got {stopping_threshold}.",
                    hint="pass a positive sample count.",
                )
            )
