"""Linear Thompson Sampling for contextual bandits.

Assigns personalized treatments using user context features to estimate
reward per arm via Bayesian linear regression.

Reference
---------
Agrawal & Goyal (2013).
"Thompson Sampling for Contextual Bandits with Linear Payoffs." *ICML*.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from splita._types import BanditResult
from splita._utils import ensure_rng
from splita._validation import check_is_integer, check_positive, format_error


class LinTS:
    """Linear Thompson Sampling contextual bandit.

    Maintains a Bayesian linear regression model per arm. At each step,
    posterior parameters are sampled via Thompson Sampling and the arm
    with the highest predicted reward for the given context is chosen.

    Parameters
    ----------
    n_arms : int
        Number of arms (>= 2).
    n_features : int
        Dimensionality of context features (>= 1).
    lambda_ : float, default 1.0
        L2 regularisation (prior precision). Larger values encourage
        more exploration early on.
    noise_var : float, default 1.0
        Assumed observation noise variance. Exploration scales with
        ``sqrt(noise_var)``.
    random_state : int, Generator, or None, default None
        Seed for reproducibility.

    Examples
    --------
    >>> import numpy as np
    >>> lints = LinTS(2, n_features=3, random_state=42)
    >>> ctx = np.array([1.0, 0.0, 0.5])
    >>> arm = lints.recommend(ctx)
    >>> lints.update(arm, ctx, reward=1.0)
    """

    def __init__(
        self,
        n_arms: int,
        n_features: int,
        *,
        lambda_: float = 1.0,
        noise_var: float = 1.0,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        # ── Validation ──────────────────────────────────────────────
        check_is_integer(n_arms, "n_arms", min_value=2, hint="need at least 2 arms.")
        check_is_integer(
            n_features,
            "n_features",
            min_value=1,
            hint="need at least 1 feature.",
        )
        check_positive(lambda_, "lambda_", hint="regularisation must be > 0.")
        check_positive(noise_var, "noise_var", hint="noise variance must be > 0.")

        self._n_arms: int = int(n_arms)
        self._n_features: int = int(n_features)
        self._lambda: float = float(lambda_)
        self._noise_var: float = float(noise_var)
        self._rng: np.random.Generator = ensure_rng(random_state)

        # Per-arm posterior state
        d = self._n_features
        self._B: list[np.ndarray] = [
            self._lambda * np.eye(d) for _ in range(self._n_arms)
        ]
        self._f: list[np.ndarray] = [np.zeros(d) for _ in range(self._n_arms)]
        self._mu_hat: list[np.ndarray] = [
            np.zeros(d) for _ in range(self._n_arms)
        ]

        # Tracking state for result()
        self._n_pulls = np.zeros(self._n_arms, dtype=int)
        self._total_reward = 0.0
        self._reward_sums = np.zeros(self._n_arms)

    # ── Public read-only properties ─────────────────────────────────

    @property
    def n_arms(self) -> int:
        """Number of arms."""
        return self._n_arms

    @property
    def n_features(self) -> int:
        """Number of context features."""
        return self._n_features

    # ── Public API ──────────────────────────────────────────────────

    def update(
        self, arm: int, context: np.ndarray | list | tuple, reward: float
    ) -> None:
        """Update the posterior for *arm* given a context-reward pair.

        Parameters
        ----------
        arm : int
            Arm index in ``[0, n_arms)``.
        context : array-like, shape (n_features,)
            Context feature vector.
        reward : float
            Observed reward.

        Raises
        ------
        ValueError
            If *arm* is out of range or *context* has the wrong shape.
        """
        self._validate_arm(arm)
        x = self._validate_context(context)

        self._B[arm] += np.outer(x, x)
        self._f[arm] += reward * x
        self._mu_hat[arm] = np.linalg.solve(self._B[arm], self._f[arm])

        self._n_pulls[arm] += 1
        self._total_reward += reward
        self._reward_sums[arm] += reward

    def recommend(self, context: np.ndarray | list | tuple) -> int:
        """Choose the best arm for the given context via Thompson Sampling.

        For each arm, a parameter vector is sampled from the posterior
        and the arm with the highest predicted reward is returned.

        Parameters
        ----------
        context : array-like, shape (n_features,)
            Context feature vector.

        Returns
        -------
        int
            Index of the recommended arm.

        Raises
        ------
        ValueError
            If *context* has the wrong shape.
        """
        x = self._validate_context(context)

        best_arm = 0
        best_reward = -np.inf

        for arm in range(self._n_arms):
            cf = cho_factor(self._B[arm])
            cov = self._noise_var * cho_solve(cf, np.eye(self._n_features))
            theta = self._rng.multivariate_normal(self._mu_hat[arm], cov)
            r_hat = float(theta @ x)
            if r_hat > best_reward:
                best_reward = r_hat
                best_arm = arm

        return best_arm

    def result(self) -> BanditResult:
        """Return the current bandit state as a :class:`BanditResult`.

        Returns
        -------
        BanditResult
            Frozen dataclass with posterior summaries and stopping info.

        Notes
        -----
        LinTS is a contextual bandit, so ``prob_best`` and
        ``expected_loss`` are not directly applicable and are set to
        uniform / zero respectively.  ``arm_means`` reflect the average
        observed reward per arm, and ``arm_credible_intervals`` are
        approximated from mu_hat norms.
        """
        n_pulls_list = self._n_pulls.tolist()

        # Arm means: average observed reward per arm
        arm_means: list[float] = []
        for i in range(self._n_arms):
            if self._n_pulls[i] > 0:
                arm_means.append(float(self._reward_sums[i] / self._n_pulls[i]))
            else:
                arm_means.append(0.0)

        # Current best arm: arm with highest average reward
        current_best = int(np.argmax(arm_means)) if sum(n_pulls_list) > 0 else 0

        # prob_best: uniform (not directly applicable for contextual)
        prob_best = [1.0 / self._n_arms] * self._n_arms

        # expected_loss: zeros (not directly applicable for contextual)
        expected_loss = [0.0] * self._n_arms

        # Credible intervals: approximate from posterior uncertainty
        arm_ci: list[tuple[float, float]] = []
        for i in range(self._n_arms):
            if self._n_pulls[i] > 0:
                cf = cho_factor(self._B[i])
                cov = self._noise_var * cho_solve(
                    cf, np.eye(self._n_features)
                )
                # Use norm of diagonal as a scalar uncertainty measure
                std = float(np.sqrt(np.mean(np.diag(cov))))
                mean = arm_means[i]
                arm_ci.append((mean - 1.96 * std, mean + 1.96 * std))
            else:
                arm_ci.append((0.0, 0.0))

        return BanditResult(
            n_pulls_per_arm=n_pulls_list,
            prob_best=prob_best,
            expected_loss=expected_loss,
            current_best_arm=current_best,
            should_stop=False,
            total_reward=self._total_reward,
            cumulative_regret=None,
            arm_means=arm_means,
            arm_credible_intervals=arm_ci,
        )

    # ── Validation helpers ──────────────────────────────────────────

    def _validate_arm(self, arm: int) -> None:
        """Check that *arm* is a valid arm index."""
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
                    "check that arm index matches the number of arms.",
                )
            )

    def _validate_context(self, context: np.ndarray | list | tuple) -> np.ndarray:
        """Validate and convert *context* to a 1-D float64 array."""
        x = np.asarray(context, dtype=np.float64)
        if x.shape != (self._n_features,):
            raise ValueError(
                format_error(
                    f"`context` must have shape ({self._n_features},), got {x.shape}.",
                    f"expected {self._n_features} features, received "
                    f"array with shape {x.shape}.",
                    "ensure context is a 1-D array with n_features elements.",
                )
            )
        return x
