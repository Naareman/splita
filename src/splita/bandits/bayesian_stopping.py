"""Bayesian stopping rules for bandits.

Standalone stopping rule evaluator that can be applied to any
:class:`~splita.BanditResult`.  Supports three criteria: expected loss,
probability of being best, and precision (credible interval width).
"""

from __future__ import annotations

from typing import Literal

from splita._types import BanditResult, BayesianStoppingResult
from splita._validation import (
    check_is_integer,
    check_one_of,
    check_positive,
    format_error,
)

_VALID_RULES = ["expected_loss", "prob_best", "precision"]


class BayesianStopping:
    """Bayesian stopping rules for multi-armed bandits.

    Evaluates whether a bandit experiment should stop based on the
    posterior summaries in a :class:`BanditResult`.

    Parameters
    ----------
    rule : ``'expected_loss'``, ``'prob_best'``, or ``'precision'``
        Stopping criterion.

        - ``'expected_loss'``: stop when ``min(expected_loss) < threshold``.
        - ``'prob_best'``: stop when ``max(prob_best) > threshold``.
        - ``'precision'``: stop when ``max(CI_width) < threshold``.
    threshold : float, default 0.01
        Threshold for the stopping rule.
    min_samples : int, default 100
        Minimum total observations before evaluating the stopping rule.

    Examples
    --------
    >>> from splita import ThompsonSampler
    >>> import numpy as np
    >>> ts = ThompsonSampler(2, random_state=42)
    >>> for _ in range(200):
    ...     arm = ts.recommend()
    ...     reward = 1 if (arm == 0 and np.random.random() < 0.8) else 0
    ...     ts.update(arm, reward)
    >>> stopper = BayesianStopping(rule="expected_loss", threshold=0.05)
    >>> result = stopper.evaluate(ts.result())
    >>> isinstance(result.should_stop, bool)
    True
    """

    def __init__(
        self,
        *,
        rule: Literal["expected_loss", "prob_best", "precision"] = "expected_loss",
        threshold: float = 0.01,
        min_samples: int = 100,
    ) -> None:
        check_one_of(rule, "rule", _VALID_RULES)
        self._validate_threshold(rule, threshold)
        check_is_integer(min_samples, "min_samples", min_value=0)

        self._rule = rule
        self._threshold = float(threshold)
        self._min_samples = int(min_samples)

    # ─── public API ─────────────────────────────────────────────────

    def should_stop(self, bandit_result: BanditResult) -> bool:
        """Check whether the stopping criterion is met.

        Parameters
        ----------
        bandit_result : BanditResult
            Current bandit state to evaluate.

        Returns
        -------
        bool
            ``True`` if the stopping criterion is satisfied.
        """
        return self.evaluate(bandit_result).should_stop

    def evaluate(self, bandit_result: BanditResult) -> BayesianStoppingResult:
        """Evaluate the stopping rule against a bandit result.

        Parameters
        ----------
        bandit_result : BanditResult
            Current bandit state to evaluate.

        Returns
        -------
        BayesianStoppingResult
            Detailed evaluation result.

        Raises
        ------
        TypeError
            If *bandit_result* is not a :class:`BanditResult`.
        """
        if not isinstance(bandit_result, BanditResult):
            raise TypeError(
                format_error(
                    f"`bandit_result` must be a BanditResult, got "
                    f"type {type(bandit_result).__name__}.",
                )
            )

        total_samples = sum(bandit_result.n_pulls_per_arm)

        if total_samples < self._min_samples:
            return BayesianStoppingResult(
                should_stop=False,
                rule=self._rule,
                threshold=self._threshold,
                current_value=float("nan"),
                message=(
                    f"Not enough samples ({total_samples} < {self._min_samples}). "
                    f"Waiting for minimum sample size."
                ),
            )

        if self._rule == "expected_loss":
            current_value = min(bandit_result.expected_loss)
            stop = current_value < self._threshold
            message = (
                f"min(expected_loss) = {current_value:.6f} "
                f"{'<' if stop else '>='} threshold {self._threshold}."
            )
        elif self._rule == "prob_best":
            current_value = max(bandit_result.prob_best)
            stop = current_value > self._threshold
            message = (
                f"max(prob_best) = {current_value:.4f} "
                f"{'>' if stop else '<='} threshold {self._threshold}."
            )
        else:  # precision
            ci_widths = [ci[1] - ci[0] for ci in bandit_result.arm_credible_intervals]
            current_value = max(ci_widths)
            stop = current_value < self._threshold
            message = (
                f"max(CI_width) = {current_value:.6f} "
                f"{'<' if stop else '>='} threshold {self._threshold}."
            )

        return BayesianStoppingResult(
            should_stop=stop,
            rule=self._rule,
            threshold=self._threshold,
            current_value=float(current_value),
            message=message,
        )

    # ─── private helpers ────────────────────────────────────────────

    @staticmethod
    def _validate_threshold(rule: str, threshold: float) -> None:
        """Validate that *threshold* is sensible for the rule."""
        if rule == "expected_loss":
            check_positive(threshold, "threshold", hint="typical values are 0.001 to 0.05.")
        elif rule == "prob_best":
            if not (0 < threshold < 1):
                raise ValueError(
                    format_error(
                        f"`threshold` must be in (0, 1) for 'prob_best' rule, got {threshold}.",
                        hint="typical values are 0.90 to 0.99.",
                    )
                )
        elif rule == "precision":
            check_positive(threshold, "threshold", hint="typical values are 0.01 to 0.1.")
