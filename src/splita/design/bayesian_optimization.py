"""Bayesian experiment optimization (Meta 2025 "Experimenting, Fast and Slow").

Combines short-term and long-term experimental outcomes via a surrogate
model and uses acquisition-function-based optimization to suggest the
next treatment parameters to try.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from splita._types import BayesOptResult
from splita._validation import format_error


class BayesianExperimentOptimizer:
    """Optimize experiment parameters using a surrogate model.

    Builds a surrogate model that maps short-term outcomes to long-term
    outcomes.  Uses this model with an acquisition function to recommend
    the next set of treatment parameters to try.

    Parameters
    ----------
    param_bounds : dict[str, tuple[float, float]]
        Bounds for each treatment parameter.  Keys are parameter names
        and values are ``(lower, upper)`` tuples.
    exploration_weight : float, default 1.0
        Weight for the exploration term in the acquisition function.
        Higher values encourage more exploration.

    Examples
    --------
    >>> opt = BayesianExperimentOptimizer(
    ...     param_bounds={"price": (5.0, 50.0), "discount": (0.0, 0.5)}
    ... )
    >>> opt.add_experiment({"price": 10.0, "discount": 0.1}, 5.0, 8.0)
    >>> opt.add_experiment({"price": 20.0, "discount": 0.2}, 7.0, 12.0)
    >>> opt.add_experiment({"price": 30.0, "discount": 0.3}, 4.0, 6.0)
    >>> result = opt.result()
    >>> result.n_experiments
    3
    """

    def __init__(
        self,
        param_bounds: dict[str, tuple[float, float]],
        *,
        exploration_weight: float = 1.0,
    ) -> None:
        if not isinstance(param_bounds, dict) or len(param_bounds) == 0:
            raise ValueError(
                format_error(
                    "`param_bounds` must be a non-empty dict.",
                    "each key should be a parameter name with (lower, upper) bounds.",
                    "example: {'price': (5.0, 50.0)}.",
                )
            )

        for name, bounds in param_bounds.items():
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:  # pragma: no cover
                raise ValueError(
                    format_error(
                        f"`param_bounds['{name}']` must be a (lower, upper) tuple.",
                        f"got {bounds!r}.",
                    )
                )
            lo, hi = bounds
            if lo >= hi:
                raise ValueError(
                    format_error(
                        f"`param_bounds['{name}']` lower bound must be < upper bound.",
                        f"got lower={lo}, upper={hi}.",
                    )
                )

        if exploration_weight < 0:
            raise ValueError(
                format_error(
                    f"`exploration_weight` must be >= 0, got {exploration_weight}.",
                    "exploration_weight controls exploration vs exploitation.",
                    "set to 0 for pure exploitation, higher for more exploration.",
                )
            )

        self._param_bounds = param_bounds
        self._param_names = sorted(param_bounds.keys())
        self._exploration_weight = exploration_weight

        # Storage for experiments
        self._param_vectors: list[np.ndarray] = []
        self._short_term: list[float] = []
        self._long_term: list[float | None] = []

    def add_experiment(
        self,
        treatment_params: dict,
        short_term_outcome: float,
        long_term_outcome: float | None = None,
    ) -> None:
        """Record the result of an experiment.

        Parameters
        ----------
        treatment_params : dict
            Treatment parameters used in the experiment.  Must contain
            all keys defined in ``param_bounds``.
        short_term_outcome : float
            Observed short-term outcome metric.
        long_term_outcome : float or None
            Observed long-term outcome metric, if available.

        Raises
        ------
        ValueError
            If *treatment_params* is missing keys or values are out of bounds.
        """
        if not isinstance(treatment_params, dict):  # pragma: no cover
            raise TypeError(
                format_error(
                    "`treatment_params` must be a dict.",
                    f"got type {type(treatment_params).__name__}.",
                )
            )

        for name in self._param_names:
            if name not in treatment_params:
                raise ValueError(
                    format_error(
                        f"`treatment_params` must contain key '{name}'.",
                        f"missing from provided params: {list(treatment_params.keys())}.",
                    )
                )

        vec = np.array([float(treatment_params[n]) for n in self._param_names])
        self._param_vectors.append(vec)
        self._short_term.append(float(short_term_outcome))
        self._long_term.append(float(long_term_outcome) if long_term_outcome is not None else None)

    def _build_surrogate(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Build a linear surrogate model from short-term to long-term.

        Returns
        -------
        coefficients : np.ndarray
            Linear regression coefficients (including intercept as last element).
        predictions : np.ndarray
            Predicted long-term outcomes for all experiments.
        r2 : float
            R-squared of the surrogate model.
        """
        # Use experiments with both short-term and long-term data
        X_list: list[np.ndarray] = []
        y_list: list[float] = []

        for i, lt in enumerate(self._long_term):
            if lt is not None:
                features = np.concatenate([self._param_vectors[i], [self._short_term[i]]])
                X_list.append(features)
                y_list.append(lt)

        if len(X_list) < 2:
            raise ValueError(
                format_error(
                    "Need at least 2 experiments with long-term outcomes to build surrogate.",
                    f"got {len(X_list)}.",
                    "call add_experiment() with long_term_outcome for more experiments.",
                )
            )

        X = np.array(X_list)
        y = np.array(y_list)

        # Add intercept column
        X_aug = np.column_stack([X, np.ones(len(X))])

        # OLS: coeffs = (X'X)^{-1} X'y
        try:
            coeffs = np.linalg.lstsq(X_aug, y, rcond=None)[0]
        except np.linalg.LinAlgError:  # pragma: no cover
            coeffs = np.zeros(X_aug.shape[1])

        # Predictions for all experiments
        all_preds = []
        for i in range(len(self._param_vectors)):
            features = np.concatenate([self._param_vectors[i], [self._short_term[i], 1.0]])
            all_preds.append(float(features @ coeffs))

        # R-squared on training data
        y_pred_train = X_aug @ coeffs
        ss_res = float(np.sum((y - y_pred_train) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r2 = max(0.0, r2)

        return coeffs, np.array(all_preds), r2

    def suggest_next(self) -> dict:
        """Suggest the next treatment parameters to try.

        Uses the surrogate model and an upper-confidence-bound acquisition
        function to balance exploration and exploitation.

        Returns
        -------
        dict
            Suggested treatment parameters.

        Raises
        ------
        ValueError
            If fewer than 2 experiments with long-term outcomes have been added.
        """
        coeffs, _predictions, _r2 = self._build_surrogate()

        # Uncertainty estimate: distance from existing points
        param_matrix = np.array(self._param_vectors)

        def neg_acquisition(x: np.ndarray) -> float:
            features = np.concatenate([x, [np.mean(self._short_term), 1.0]])
            predicted = float(features @ coeffs)

            # Distance-based exploration bonus
            if len(param_matrix) > 0:
                dists = np.linalg.norm(param_matrix - x, axis=1)
                min_dist = float(np.min(dists))
            else:  # pragma: no cover
                min_dist = 1.0

            return -(predicted + self._exploration_weight * min_dist)

        bounds = [self._param_bounds[n] for n in self._param_names]

        # Multi-start optimization
        best_val = float("inf")
        best_x = None
        rng = np.random.default_rng(42)

        for _ in range(10):
            x0 = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
            res = minimize(neg_acquisition, x0, bounds=bounds, method="L-BFGS-B")
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x

        if best_x is None:  # pragma: no cover
            # Fallback: center of bounds
            best_x = np.array([(lo + hi) / 2 for lo, hi in bounds])

        return {name: float(best_x[i]) for i, name in enumerate(self._param_names)}

    def result(self) -> BayesOptResult:
        """Return the current optimization result.

        Returns
        -------
        BayesOptResult
            Best parameters found so far and surrogate model quality.

        Raises
        ------
        ValueError
            If fewer than 2 experiments with long-term outcomes have been added.
        """
        _coeffs, predictions, r2 = self._build_surrogate()

        # Find best experiment by predicted long-term outcome
        best_idx = int(np.argmax(predictions))
        best_params = {
            name: float(self._param_vectors[best_idx][i])
            for i, name in enumerate(self._param_names)
        }

        return BayesOptResult(
            best_params=best_params,
            predicted_long_term=float(predictions[best_idx]),
            n_experiments=len(self._param_vectors),
            surrogate_r2=r2,
        )
