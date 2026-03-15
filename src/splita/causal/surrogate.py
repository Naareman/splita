"""Surrogate estimator for long-run effect estimation.

Uses a short-term outcome as a surrogate for the long-term outcome,
fitting a model on observed (short, long) pairs and then predicting the
long-term treatment effect from short-term data alone.
"""

from __future__ import annotations

import warnings

import numpy as np

from splita._types import SurrogateResult
from splita._validation import check_array_like, check_same_length, format_error

ArrayLike = list | tuple | np.ndarray


class SurrogateEstimator:
    """Estimate long-run treatment effects via a surrogate (short-term) metric.

    Fits a regression model ``long_term = f(short_term, treatment)`` on
    historical data where both metrics are observed, then uses the fitted
    model to predict the long-term effect from short-term observations only.

    Parameters
    ----------
    estimator : object, optional
        A scikit-learn-compatible regressor with ``fit`` and ``predict``.
        If *None*, uses ``sklearn.linear_model.Ridge(alpha=1.0)``.
    random_state : int or None, default None
        Random seed passed to the default estimator.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> short = rng.normal(5, 1, 200)
    >>> treatment = np.array([0]*100 + [1]*100)
    >>> long = 2 * short + 3 * treatment + rng.normal(0, 0.5, 200)
    >>> est = SurrogateEstimator(random_state=42)
    >>> est.fit(short, long, treatment)  # doctest: +ELLIPSIS
    <splita.causal.surrogate.SurrogateEstimator object at ...>
    >>> result = est.predict_long_term_effect(
    ...     short_term_control=rng.normal(5, 1, 50),
    ...     short_term_treatment=rng.normal(5, 1, 50),
    ... )
    >>> result.is_valid_surrogate
    True
    """

    def __init__(
        self,
        *,
        estimator: object | None = None,
        random_state: int | None = None,
    ) -> None:
        self._user_estimator = estimator
        self._random_state = random_state
        self._model: object | None = None
        self._r2: float | None = None
        self._fitted = False

    def fit(
        self,
        short_term_outcome: ArrayLike,
        long_term_outcome: ArrayLike,
        treatment: ArrayLike,
    ) -> SurrogateEstimator:
        """Fit the surrogate model on historical data.

        Parameters
        ----------
        short_term_outcome : array-like
            Short-term metric values for all units.
        long_term_outcome : array-like
            Long-term metric values for all units.
        treatment : array-like
            Binary treatment indicator (0 = control, 1 = treatment).

        Returns
        -------
        SurrogateEstimator
            The fitted estimator (self).

        Raises
        ------
        TypeError
            If inputs cannot be converted to numeric arrays.
        ValueError
            If arrays have mismatched lengths or fewer than 5 elements.
        """
        short = check_array_like(short_term_outcome, "short_term_outcome", min_length=5)
        long = check_array_like(long_term_outcome, "long_term_outcome", min_length=5)
        trt = check_array_like(treatment, "treatment", min_length=5)

        check_same_length(short, long, "short_term_outcome", "long_term_outcome")
        check_same_length(short, trt, "short_term_outcome", "treatment")

        unique_trt = np.unique(trt)
        if not np.all(np.isin(unique_trt, [0.0, 1.0])):
            raise ValueError(
                format_error(
                    "`treatment` must contain only 0 and 1.",
                    f"found unique values: {unique_trt.tolist()}.",
                    "encode control as 0 and treatment as 1.",
                )
            )

        if self._user_estimator is not None:
            model = self._user_estimator
        else:
            try:
                from sklearn.linear_model import Ridge
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    format_error(
                        "scikit-learn is required for the default surrogate estimator.",
                        "install it with: pip install scikit-learn",
                    )
                ) from exc
            model = Ridge(alpha=1.0, random_state=self._random_state)

        X = np.column_stack([short, trt])
        model.fit(X, long)  # type: ignore[union-attr]

        y_pred = model.predict(X)  # type: ignore[union-attr]
        ss_res = float(np.sum((long - y_pred) ** 2))
        ss_tot = float(np.sum((long - np.mean(long)) ** 2))
        self._r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        if self._r2 < 0.3:
            warnings.warn(
                f"Surrogate model has low R-squared ({self._r2:.3f}). "
                "The short-term metric may not be a valid surrogate for the "
                "long-term outcome. Results should be interpreted with caution.",
                RuntimeWarning,
                stacklevel=2,
            )

        self._model = model
        self._fitted = True
        return self

    def predict_long_term_effect(
        self,
        short_term_control: ArrayLike,
        short_term_treatment: ArrayLike,
    ) -> SurrogateResult:
        """Predict the long-term treatment effect from short-term data.

        Parameters
        ----------
        short_term_control : array-like
            Short-term metric values for the control group.
        short_term_treatment : array-like
            Short-term metric values for the treatment group.

        Returns
        -------
        SurrogateResult
            Predicted long-term lift with confidence interval and diagnostics.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called.
        TypeError
            If inputs cannot be converted to numeric arrays.
        ValueError
            If arrays have fewer than 2 elements.
        """
        if not self._fitted:
            raise RuntimeError(
                format_error(
                    "SurrogateEstimator must be fitted before predicting.",
                    "call .fit() with historical data first.",
                )
            )

        ctrl = check_array_like(short_term_control, "short_term_control", min_length=2)
        trt = check_array_like(short_term_treatment, "short_term_treatment", min_length=2)

        model = self._model

        X_ctrl = np.column_stack([ctrl, np.zeros(len(ctrl))])
        X_trt = np.column_stack([trt, np.ones(len(trt))])

        pred_ctrl = model.predict(X_ctrl)  # type: ignore[union-attr]
        pred_trt = model.predict(X_trt)  # type: ignore[union-attr]

        predicted_lift = float(np.mean(pred_trt) - np.mean(pred_ctrl))

        # Bootstrap CI for the predicted lift
        rng = np.random.default_rng(self._random_state)
        n_boot = 1000
        boot_lifts = np.empty(n_boot)
        for i in range(n_boot):
            idx_c = rng.integers(0, len(ctrl), size=len(ctrl))
            idx_t = rng.integers(0, len(trt), size=len(trt))
            boot_ctrl = ctrl[idx_c]
            boot_trt = trt[idx_t]
            X_bc = np.column_stack([boot_ctrl, np.zeros(len(boot_ctrl))])
            X_bt = np.column_stack([boot_trt, np.ones(len(boot_trt))])
            boot_lifts[i] = float(
                np.mean(model.predict(X_bt))  # type: ignore[union-attr]
                - np.mean(model.predict(X_bc))  # type: ignore[union-attr]
            )

        ci_lower = float(np.percentile(boot_lifts, 2.5))
        ci_upper = float(np.percentile(boot_lifts, 97.5))

        r2 = self._r2 if self._r2 is not None else 0.0

        return SurrogateResult(
            predicted_long_term_lift=predicted_lift,
            prediction_ci=(ci_lower, ci_upper),
            surrogate_r2=r2,
            is_valid_surrogate=r2 > 0.3,
        )
