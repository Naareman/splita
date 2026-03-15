"""Offline policy evaluation — evaluate policies from logged data.

Implements the inverse propensity scoring (IPS) estimator and the
doubly robust (DR) estimator for unbiased offline evaluation of
new policies using historical logged data (Li et al. 2011).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from splita._types import OfflineResult
from splita._validation import (
    check_array_like,
    check_in_range,
    check_same_length,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


class OfflineEvaluator:
    """Evaluate a target policy offline using logged bandit data.

    Uses importance weighting to correct for the mismatch between
    the logging policy and the target policy (Li et al. 2011).

    Parameters
    ----------
    method : {'ips', 'doubly_robust'}, default 'ips'
        Estimation method.

        - ``'ips'``: Inverse Propensity Scoring.
        - ``'doubly_robust'``: Doubly robust estimator combining
          IPS with a reward model.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    clip : float or None, default 100.0
        Maximum importance weight ratio to clip for stability.
        Set to None to disable clipping.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n = 500
    >>> rewards = rng.binomial(1, 0.3, n).astype(float)
    >>> actions = rng.choice(3, n)
    >>> contexts = rng.normal(0, 1, (n, 2))
    >>> log_probs = np.full(n, 1.0 / 3)
    >>> target_probs = np.full(n, 1.0 / 3)
    >>> evaluator = OfflineEvaluator()
    >>> evaluator.evaluate(rewards, actions, contexts, log_probs, target_probs)
    ... # doctest: +ELLIPSIS
    OfflineResult(...)
    """

    def __init__(
        self,
        *,
        method: str = "ips",
        alpha: float = 0.05,
        clip: float | None = 100.0,
    ) -> None:
        valid_methods = ["ips", "doubly_robust"]
        if method not in valid_methods:
            raise ValueError(
                format_error(
                    f"`method` must be one of {tuple(valid_methods)}, got {method!r}.",
                    hint="use 'ips' for inverse propensity scoring "
                    "or 'doubly_robust' for the DR estimator.",
                )
            )

        check_in_range(
            alpha,
            "alpha",
            0.0,
            1.0,
            hint="typical values are 0.05, 0.01, or 0.10",
        )

        if clip is not None and clip <= 0:
            raise ValueError(
                format_error(
                    "`clip` must be positive or None, got {}.".format(clip),
                    "the clip parameter bounds the maximum importance weight.",
                )
            )

        self._method = method
        self._alpha = alpha
        self._clip = clip

    def evaluate(
        self,
        rewards: ArrayLike,
        actions: ArrayLike,
        contexts: ArrayLike,
        logging_policy_probs: ArrayLike,
        target_policy_probs: ArrayLike,
    ) -> OfflineResult:
        """Evaluate the target policy using logged data.

        Parameters
        ----------
        rewards : array-like
            Observed rewards for each logged interaction.
        actions : array-like
            Actions taken by the logging policy.
        contexts : array-like
            Context features for each interaction. Can be 1-D or 2-D.
        logging_policy_probs : array-like
            Probability of the logged action under the logging policy,
            i.e. ``mu(a_i | x_i)`` for each observation.
        target_policy_probs : array-like
            Probability of the logged action under the target policy,
            i.e. ``pi(a_i | x_i)`` for each observation.

        Returns
        -------
        OfflineResult
            Estimated policy value with uncertainty.

        Raises
        ------
        ValueError
            If arrays have mismatched lengths or logging probabilities
            contain zeros.
        """
        r = check_array_like(rewards, "rewards", min_length=2)
        a = check_array_like(actions, "actions", min_length=2)

        # contexts can be multi-dimensional, handle separately
        ctx = np.asarray(contexts, dtype="float64")

        mu = check_array_like(logging_policy_probs, "logging_policy_probs", min_length=2)
        pi = check_array_like(target_policy_probs, "target_policy_probs", min_length=2)

        check_same_length(r, a, "rewards", "actions")
        check_same_length(r, mu, "rewards", "logging_policy_probs")
        check_same_length(r, pi, "rewards", "target_policy_probs")

        n = len(r)

        # Validate probabilities
        if np.any(mu <= 0):
            raise ValueError(
                format_error(
                    "`logging_policy_probs` must be strictly positive.",
                    "found values <= 0. Zero-probability actions cannot be used for IPS.",
                    "ensure the logging policy has full support over all observed actions.",
                )
            )

        if np.any(pi < 0):
            raise ValueError(
                format_error(
                    "`target_policy_probs` must be non-negative.",
                    "found negative probability values.",
                )
            )

        # Importance weights
        w = pi / mu

        # Clip weights for stability
        if self._clip is not None:
            w = np.clip(w, 0, self._clip)

        if self._method == "ips":
            # IPS estimator: V(pi) = 1/n * sum(r_i * w_i)
            weighted_rewards = r * w
            estimated_value = float(np.mean(weighted_rewards))

            # Standard error
            se = float(np.std(weighted_rewards, ddof=1) / np.sqrt(n))

        else:
            # Doubly robust estimator
            # Use simple reward model: mean reward per action
            unique_actions = np.unique(a)
            reward_model = {}
            for act in unique_actions:
                mask = a == act
                reward_model[act] = float(np.mean(r[mask])) if np.any(mask) else 0.0

            # DR estimator:
            # V(pi) = 1/n * sum(w_i * (r_i - r_hat(x_i, a_i)) + r_hat_pi(x_i))
            # where r_hat_pi(x_i) = sum_a pi(a|x_i) * r_hat(x_i, a)
            r_hat = np.array([reward_model.get(int(ai), 0.0) for ai in a])

            # For the direct method term, we approximate by the weighted average
            # of reward model predictions
            r_hat_pi = np.zeros(n)
            for act in unique_actions:
                # Use the mean target probability for this action as approx
                act_mask = a == act
                if np.any(act_mask):
                    r_hat_pi += np.mean(pi[act_mask]) * reward_model.get(int(act), 0.0)

            dr_scores = w * (r - r_hat) + r_hat_pi
            estimated_value = float(np.mean(dr_scores))
            se = float(np.std(dr_scores, ddof=1) / np.sqrt(n))

        # Confidence interval
        z_crit = float(norm.ppf(1 - self._alpha / 2))
        ci_lower = estimated_value - z_crit * se
        ci_upper = estimated_value + z_crit * se

        # Effective sample size: n / (1 + Var(w))
        var_w = float(np.var(w, ddof=1)) if n > 1 else 0.0
        ess = n / (1 + var_w) if (1 + var_w) > 0 else float(n)

        return OfflineResult(
            estimated_value=estimated_value,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            effective_sample_size=float(ess),
            method=self._method,
            n=n,
        )
