"""LinUCB contextual bandit.

Upper Confidence Bound approach for contextual bandits.  Uses the
posterior precision matrix to construct confidence intervals and
selects the arm with the highest upper confidence bound.

Reference
---------
Li, Chu, Langford & Schapire (2010).
"A Contextual-Bandit Approach to Personalized News Article Recommendation."
*WWW*.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from splita._types import BanditResult
from splita._utils import ensure_rng
from splita._validation import check_is_integer, check_positive, format_error


class LinUCB:
    """LinUCB contextual bandit algorithm.

    Maintains a regularised linear model per arm and selects the arm
    with the highest upper confidence bound for the given context.

    Parameters
    ----------
    n_arms : int
        Number of arms (>= 2).
    n_features : int
        Dimensionality of context features (>= 1).
    alpha : float, default 1.0
        Exploration parameter controlling the width of the confidence
        bound.  Larger values encourage more exploration.
    random_state : int, Generator, or None, default None
        Seed for reproducibility (used only for tie-breaking).

    Examples
    --------
    >>> import numpy as np
    >>> ucb = LinUCB(2, n_features=3, alpha=1.0, random_state=42)
    >>> ctx = np.array([1.0, 0.0, 0.5])
    >>> arm = ucb.recommend(ctx)
    >>> ucb.update(arm, ctx, reward=1.0)
    """

    def __init__(
        self,
        n_arms: int,
        n_features: int,
        *,
        alpha: float = 1.0,
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
        check_positive(alpha, "alpha", hint="exploration parameter must be > 0.")

        self._n_arms: int = int(n_arms)
        self._n_features: int = int(n_features)
        self._alpha: float = float(alpha)
        self._rng: np.random.Generator = ensure_rng(random_state)

        # Per-arm state: B = regularised design matrix, f = reward-weighted features
        d = self._n_features
        self._B: list[np.ndarray] = [np.eye(d) for _ in range(self._n_arms)]
        self._f: list[np.ndarray] = [np.zeros(d) for _ in range(self._n_arms)]
        self._mu_hat: list[np.ndarray] = [np.zeros(d) for _ in range(self._n_arms)]

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

    def update(self, arm: int, context: np.ndarray | list | tuple, reward: float) -> None:
        """Update the model for *arm* given a context-reward pair.

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

        # Solve via Cholesky for numerical stability
        cf = cho_factor(self._B[arm])
        self._mu_hat[arm] = cho_solve(cf, self._f[arm])

        self._n_pulls[arm] += 1
        self._total_reward += reward
        self._reward_sums[arm] += reward

    def recommend(self, context: np.ndarray | list | tuple) -> int:
        """Choose the best arm for the given context via UCB.

        For each arm, the upper confidence bound is computed as
        ``mu_hat @ x + alpha * sqrt(x^T B^{-1} x)`` and the arm with the
        highest score is returned.

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
        best_score = -np.inf

        for arm in range(self._n_arms):
            cf = cho_factor(self._B[arm])
            # x^T B^{-1} x
            B_inv_x = cho_solve(cf, x)
            exploration = self._alpha * np.sqrt(float(x @ B_inv_x))
            exploitation = float(self._mu_hat[arm] @ x)
            score = exploitation + exploration

            if score > best_score:
                best_score = score
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
        LinUCB is a contextual bandit, so ``prob_best`` and
        ``expected_loss`` are not directly applicable and are set to
        uniform / zero respectively.
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
                B_inv = cho_solve(cf, np.eye(self._n_features))
                std = float(np.sqrt(np.mean(np.diag(B_inv))))
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
                    f"expected {self._n_features} features, received array with shape {x.shape}.",
                    "ensure context is a 1-D array with n_features elements.",
                )
            )
        return x
