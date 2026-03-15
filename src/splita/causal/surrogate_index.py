"""Surrogate index estimation (Athey et al. 2019).

Predicts long-term treatment effects from short-term surrogate outcomes
using cross-fitted models.
"""

from __future__ import annotations

import warnings

import numpy as np

from splita._types import SurrogateIndexResult
from splita._utils import ensure_rng
from splita._validation import (
    check_array_like,
    check_is_integer,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


class SurrogateIndex:
    """Estimate long-term treatment effects via surrogate outcomes.

    Fits a cross-validated model mapping short-term outcomes to the
    long-term outcome, then uses this surrogate index to predict the
    treatment effect on the long-term metric.

    Parameters
    ----------
    estimator : sklearn-compatible estimator or None, default None
        Any object with ``.fit(X, y)`` and ``.predict(X)`` methods.
        If ``None``, uses ``sklearn.linear_model.Ridge(alpha=1.0)``.
    cv : int, default 5
        Number of cross-validation folds.
    random_state : int, Generator, or None, default None
        Seed for reproducibility.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n = 200
    >>> short_ctrl = rng.normal(0, 1, (n, 3))
    >>> short_trt = rng.normal(0.5, 1, (n, 3))
    >>> short_all = np.vstack([short_ctrl, short_trt])
    >>> long_outcome = short_all[:, 0] * 2 + rng.normal(0, 0.5, 2 * n)
    >>> treatment = np.array([0] * n + [1] * n)
    >>> si = SurrogateIndex(random_state=42)
    >>> si = si.fit(short_all, long_outcome, treatment)
    >>> result = si.predict(short_ctrl, short_trt)
    >>> result.n_surrogates
    3
    """

    def __init__(
        self,
        *,
        estimator: object | None = None,
        cv: int = 5,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        check_is_integer(cv, "cv", min_value=2)
        self._estimator = estimator
        self._cv = int(cv)
        self._rng = ensure_rng(random_state)
        self._random_state = random_state
        self._fitted = False
        self._models: list = []
        self._fold_indices: list = []
        self._r2: float = 0.0
        self._n_surrogates: int = 0

    def _make_estimator(self) -> object:
        """Create the default estimator (lazy sklearn import)."""
        if self._estimator is not None:
            try:
                from sklearn.base import clone

                return clone(self._estimator)
            except ImportError:  # pragma: no cover
                raise ImportError(
                    format_error(
                        "scikit-learn is required for SurrogateIndex.",
                        "install it with: pip install scikit-learn",
                    )
                ) from None
        try:
            from sklearn.linear_model import Ridge

            return Ridge(alpha=1.0)
        except ImportError:  # pragma: no cover
            raise ImportError(
                format_error(
                    "scikit-learn is required for SurrogateIndex.",
                    "install it with: pip install scikit-learn",
                )
            ) from None

    def fit(
        self,
        short_term_outcomes: np.ndarray,
        long_term_outcome: ArrayLike,
        treatment: ArrayLike,
    ) -> SurrogateIndex:
        """Fit the surrogate model using cross-validation.

        Parameters
        ----------
        short_term_outcomes : np.ndarray
            Matrix of short-term outcomes (n_samples, n_surrogates).
        long_term_outcome : array-like
            Long-term outcome for each unit.
        treatment : array-like
            Binary treatment assignment (0 or 1).

        Returns
        -------
        SurrogateIndex
            The fitted estimator (self).

        Raises
        ------
        ValueError
            If arrays are mismatched or too short.
        """
        X = np.asarray(short_term_outcomes, dtype="float64")
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(
                format_error(
                    "`short_term_outcomes` must be a 1-D or 2-D array.",
                    f"got {X.ndim}-D array with shape {X.shape}.",
                )
            )

        y_long = check_array_like(long_term_outcome, "long_term_outcome", min_length=2)
        t = check_array_like(treatment, "treatment", min_length=2)

        n = X.shape[0]
        if len(y_long) != n:
            raise ValueError(
                format_error(
                    "`short_term_outcomes` and `long_term_outcome` must have the "
                    "same number of rows.",
                    f"short_term_outcomes has {n} rows, "
                    f"long_term_outcome has {len(y_long)} elements.",
                )
            )
        if len(t) != n:
            raise ValueError(
                format_error(
                    "`short_term_outcomes` and `treatment` must have the same number of rows.",
                    f"short_term_outcomes has {n} rows, treatment has {len(t)} elements.",
                )
            )

        self._n_surrogates = X.shape[1]

        # Cross-fitted model: use control data to fit, predict on all
        # We fit on the full dataset for surrogate model (treatment-agnostic)
        indices = np.arange(n)
        self._rng.shuffle(indices)
        folds = np.array_split(indices, self._cv)

        self._models = []
        self._fold_indices = folds

        # Cross-validated predictions for R2 calculation
        cv_predictions = np.full(n, np.nan)

        for i in range(self._cv):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(self._cv) if j != i])

            model = self._make_estimator()
            model.fit(X[train_idx], y_long[train_idx])
            self._models.append(model)

            cv_predictions[test_idx] = model.predict(X[test_idx])

        # Compute cross-validated R2
        ss_res = float(np.sum((y_long - cv_predictions) ** 2))
        ss_tot = float(np.sum((y_long - np.mean(y_long)) ** 2))
        self._r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        if self._r2 < 0.3:
            warnings.warn(
                f"Surrogate R2 = {self._r2:.3f} is below 0.3. "
                "The surrogate index may not be a reliable predictor of the "
                "long-term outcome.",
                RuntimeWarning,
                stacklevel=2,
            )

        self._fitted = True
        return self

    def predict(
        self,
        short_term_control: np.ndarray,
        short_term_treatment: np.ndarray,
    ) -> SurrogateIndexResult:
        """Predict the long-term treatment effect via the surrogate index.

        Parameters
        ----------
        short_term_control : np.ndarray
            Short-term outcomes for control units (n_control, n_surrogates).
        short_term_treatment : np.ndarray
            Short-term outcomes for treatment units (n_treatment, n_surrogates).

        Returns
        -------
        SurrogateIndexResult
            Predicted treatment effect with confidence interval.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError(
                format_error(
                    "SurrogateIndex must be fitted before calling predict().",
                    "call .fit() first.",
                )
            )

        X_ctrl = np.asarray(short_term_control, dtype="float64")
        X_trt = np.asarray(short_term_treatment, dtype="float64")

        if X_ctrl.ndim == 1:
            X_ctrl = X_ctrl.reshape(-1, 1)
        if X_trt.ndim == 1:
            X_trt = X_trt.reshape(-1, 1)

        if X_ctrl.shape[1] != self._n_surrogates:
            raise ValueError(
                format_error(
                    "`short_term_control` must have the same number of features "
                    "as the training data.",
                    f"expected {self._n_surrogates} features, got {X_ctrl.shape[1]}.",
                )
            )
        if X_trt.shape[1] != self._n_surrogates:
            raise ValueError(
                format_error(
                    "`short_term_treatment` must have the same number of features "
                    "as the training data.",
                    f"expected {self._n_surrogates} features, got {X_trt.shape[1]}.",
                )
            )

        # Average predictions across cross-fitted models
        n_ctrl = X_ctrl.shape[0]
        n_trt = X_trt.shape[0]

        s_ctrl_all = np.zeros(n_ctrl)
        s_trt_all = np.zeros(n_trt)

        for model in self._models:
            s_ctrl_all += model.predict(X_ctrl)
            s_trt_all += model.predict(X_trt)

        s_ctrl_all /= len(self._models)
        s_trt_all /= len(self._models)

        # Treatment effect on the surrogate index
        mean_s_ctrl = float(np.mean(s_ctrl_all))
        mean_s_trt = float(np.mean(s_trt_all))
        predicted_effect = mean_s_trt - mean_s_ctrl

        # Delta method SE
        var_ctrl = float(np.var(s_ctrl_all, ddof=1))
        var_trt = float(np.var(s_trt_all, ddof=1))
        se = float(np.sqrt(var_ctrl / n_ctrl + var_trt / n_trt))

        # Confidence interval
        from scipy.stats import norm

        z = float(norm.ppf(0.975))
        ci_lower = predicted_effect - z * se
        ci_upper = predicted_effect + z * se

        return SurrogateIndexResult(
            predicted_effect=predicted_effect,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            surrogate_r2=self._r2,
            is_valid=self._r2 > 0.3,
            n_surrogates=self._n_surrogates,
        )
