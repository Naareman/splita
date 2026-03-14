"""HTE — Heterogeneous Treatment Effect estimation.

Estimates Conditional Average Treatment Effects (CATE) using T-learner
or S-learner meta-learning strategies.  Default base estimator is
Ridge regression (lazy sklearn import).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from splita._types import HTEResult
from splita._utils import ensure_rng
from splita._validation import (
    check_array_like,
    check_one_of,
    format_error,
)

_VALID_METHODS = ["t_learner", "s_learner"]

ArrayLike = list | tuple | np.ndarray


class HTEEstimator:
    """Estimate heterogeneous treatment effects via meta-learners.

    Parameters
    ----------
    method : {'t_learner', 's_learner'}, default 't_learner'
        Meta-learner strategy.

        - ``'t_learner'``: fits separate models for control and treatment.
          CATE = E[Y|X, T=1] - E[Y|X, T=0].
        - ``'s_learner'``: fits a single model with a treatment indicator
          column appended to ``X``.  CATE is computed from counterfactual
          predictions.

    estimator : sklearn-compatible estimator or None, default None
        Any object with ``.fit(X, y)`` and ``.predict(X)`` methods.
        If ``None``, uses ``sklearn.linear_model.Ridge(alpha=1.0)``.
    random_state : int, Generator, or None, default None
        Seed for reproducibility.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> X_ctrl = rng.normal(size=(200, 3))
    >>> X_trt = rng.normal(size=(200, 3))
    >>> y_ctrl = X_ctrl[:, 0] * 0.5 + rng.normal(0, 0.1, 200)
    >>> y_trt = X_trt[:, 0] * 1.5 + rng.normal(0, 0.1, 200)
    >>> hte = HTEEstimator(method="t_learner").fit(y_ctrl, y_trt, X_ctrl, X_trt)
    >>> result = hte.result()
    >>> result.method
    't_learner'
    """

    def __init__(
        self,
        *,
        method: Literal["t_learner", "s_learner"] = "t_learner",
        estimator: object | None = None,
        random_state: int | np.random.Generator | None = None,
    ):
        check_one_of(method, "method", _VALID_METHODS)
        self._method = method
        self._estimator = estimator
        self._rng = ensure_rng(random_state)
        self._random_state = random_state
        self._fitted = False
        self._cate: np.ndarray | None = None
        self._top_features: list[int] | None = None

    def _make_estimator(self) -> object:
        """Create the default estimator (lazy sklearn import)."""
        if self._estimator is not None:
            # Clone by importing clone
            try:
                from sklearn.base import clone

                return clone(self._estimator)
            except ImportError:
                raise ImportError(
                    format_error(
                        "scikit-learn is required for HTEEstimator.",
                        "install it with: pip install scikit-learn",
                    )
                ) from None
        try:
            from sklearn.linear_model import Ridge

            return Ridge(alpha=1.0)
        except ImportError:
            raise ImportError(
                format_error(
                    "scikit-learn is required for HTEEstimator.",
                    "install it with: pip install scikit-learn",
                )
            ) from None

    def fit(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
        X_control: ArrayLike,
        X_treatment: ArrayLike,
    ) -> HTEEstimator:
        """Fit the HTE model on control and treatment data.

        Parameters
        ----------
        control : array-like
            Outcome values for the control group.
        treatment : array-like
            Outcome values for the treatment group.
        X_control : array-like
            Feature matrix for the control group (n_control, n_features).
        X_treatment : array-like
            Feature matrix for the treatment group (n_treatment, n_features).

        Returns
        -------
        HTEEstimator
            The fitted estimator (self).

        Raises
        ------
        ValueError
            If array lengths are mismatched or arrays are too short.
        """
        y_ctrl = check_array_like(control, "control", min_length=2)
        y_trt = check_array_like(treatment, "treatment", min_length=2)

        # Convert feature matrices
        X_ctrl = np.asarray(X_control, dtype="float64")
        X_trt = np.asarray(X_treatment, dtype="float64")

        if X_ctrl.ndim == 1:
            X_ctrl = X_ctrl.reshape(-1, 1)
        if X_trt.ndim == 1:
            X_trt = X_trt.reshape(-1, 1)

        if len(y_ctrl) != X_ctrl.shape[0]:
            raise ValueError(
                format_error(
                    "`control` and `X_control` must have the same number of rows.",
                    f"control has {len(y_ctrl)} elements, "
                    f"X_control has {X_ctrl.shape[0]} rows.",
                )
            )
        if len(y_trt) != X_trt.shape[0]:
            raise ValueError(
                format_error(
                    "`treatment` and `X_treatment` must have the same number of rows.",
                    f"treatment has {len(y_trt)} elements, "
                    f"X_treatment has {X_trt.shape[0]} rows.",
                )
            )
        if X_ctrl.shape[1] != X_trt.shape[1]:
            raise ValueError(
                format_error(
                    "`X_control` and `X_treatment` must have the "
                    "same number of features.",
                    f"X_control has {X_ctrl.shape[1]} features, "
                    f"X_treatment has {X_trt.shape[1]} features.",
                )
            )

        X_all = np.vstack([X_ctrl, X_trt])

        if self._method == "t_learner":
            self._fit_t_learner(y_ctrl, y_trt, X_ctrl, X_trt, X_all)
        else:
            self._fit_s_learner(y_ctrl, y_trt, X_ctrl, X_trt, X_all)

        self._fitted = True
        return self

    def _fit_t_learner(
        self,
        y_ctrl: np.ndarray,
        y_trt: np.ndarray,
        X_ctrl: np.ndarray,
        X_trt: np.ndarray,
        X_all: np.ndarray,
    ) -> None:
        """T-learner: separate models for control and treatment."""
        model_ctrl = self._make_estimator()
        model_trt = self._make_estimator()

        model_ctrl.fit(X_ctrl, y_ctrl)
        model_trt.fit(X_trt, y_trt)

        self._model_ctrl = model_ctrl
        self._model_trt = model_trt

        # CATE on all data
        pred_ctrl = model_ctrl.predict(X_all)
        pred_trt = model_trt.predict(X_all)
        self._cate = pred_trt - pred_ctrl

        # Feature importances (if available from treatment model)
        self._extract_top_features(model_trt, X_all.shape[1])

    def _fit_s_learner(
        self,
        y_ctrl: np.ndarray,
        y_trt: np.ndarray,
        X_ctrl: np.ndarray,
        X_trt: np.ndarray,
        X_all: np.ndarray,
    ) -> None:
        """S-learner: single model with treatment indicator."""
        n_ctrl, n_trt = len(y_ctrl), len(y_trt)

        # Add treatment indicator column
        T_ctrl = np.zeros((n_ctrl, 1))
        T_trt = np.ones((n_trt, 1))
        X_ctrl_aug = np.hstack([X_ctrl, T_ctrl])
        X_trt_aug = np.hstack([X_trt, T_trt])

        X_combined = np.vstack([X_ctrl_aug, X_trt_aug])
        y_combined = np.concatenate([y_ctrl, y_trt])

        model = self._make_estimator()
        model.fit(X_combined, y_combined)

        self._model_single = model

        # CATE: predict with T=1 minus predict with T=0
        X_all_t0 = np.hstack([X_all, np.zeros((X_all.shape[0], 1))])
        X_all_t1 = np.hstack([X_all, np.ones((X_all.shape[0], 1))])
        self._cate = model.predict(X_all_t1) - model.predict(X_all_t0)

        self._extract_top_features(model, X_all.shape[1] + 1)

    def _extract_top_features(self, model: object, n_features: int) -> None:
        """Extract top feature indices if the model supports it."""
        if hasattr(model, "feature_importances_"):
            importances = np.asarray(model.feature_importances_)
            n_top = min(5, n_features)
            self._top_features = np.argsort(importances)[::-1][:n_top].tolist()
        elif hasattr(model, "coef_"):
            coef = np.asarray(model.coef_).ravel()
            n_top = min(5, len(coef))
            self._top_features = np.argsort(np.abs(coef))[::-1][:n_top].tolist()
        else:
            self._top_features = None

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict CATE for new feature vectors.

        Parameters
        ----------
        X : array-like
            Feature matrix (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted CATE for each row of ``X``.

        Raises
        ------
        RuntimeError
            If the estimator has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError(
                format_error(
                    "HTEEstimator must be fitted before calling predict().",
                    "call .fit() first.",
                )
            )

        X_arr = np.asarray(X, dtype="float64")
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        if self._method == "t_learner":
            return self._model_trt.predict(X_arr) - self._model_ctrl.predict(X_arr)
        else:
            X_t0 = np.hstack([X_arr, np.zeros((X_arr.shape[0], 1))])
            X_t1 = np.hstack([X_arr, np.ones((X_arr.shape[0], 1))])
            return self._model_single.predict(X_t1) - self._model_single.predict(X_t0)

    def result(self) -> HTEResult:
        """Return the HTE estimation result.

        Returns
        -------
        HTEResult
            Frozen dataclass with CATE estimates and summary statistics.

        Raises
        ------
        RuntimeError
            If the estimator has not been fitted.
        """
        if not self._fitted or self._cate is None:
            raise RuntimeError(
                format_error(
                    "HTEEstimator must be fitted before calling result().",
                    "call .fit() first.",
                )
            )

        cate = self._cate
        return HTEResult(
            cate_estimates=cate.tolist(),
            mean_cate=float(np.mean(cate)),
            cate_std=float(np.std(cate)),
            top_features=self._top_features,
            method=self._method,
        )
