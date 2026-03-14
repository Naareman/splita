"""Simplified Causal Forest for CATE estimation.

Wraps sklearn's RandomForest with a T-learner approach, adding
honest splitting and jackknife variance estimation for valid CIs.
"""

from __future__ import annotations

import numpy as np

from splita._types import CausalForestResult
from splita._utils import ensure_rng
from splita._validation import (
    check_array_like,
    check_is_integer,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


class CausalForest:
    """Estimate heterogeneous treatment effects via a simplified causal forest.

    Uses the T-learner approach with random forests, optionally with
    honest splitting (train/estimation split within each tree) and
    jackknife variance estimation for valid confidence intervals.

    Parameters
    ----------
    n_estimators : int, default 100
        Number of trees in each forest.
    max_depth : int or None, default None
        Maximum tree depth.  ``None`` for unlimited.
    honest : bool, default True
        If True, split data 50/50 — first half for tree structure,
        second half for leaf estimates.
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
    >>> cf = CausalForest(n_estimators=50, random_state=42)
    >>> cf = cf.fit(y_ctrl, y_trt, X_ctrl, X_trt)
    >>> cates = cf.predict(X_ctrl[:5])
    >>> len(cates)
    5
    """

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: int | None = None,
        honest: bool = True,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        check_is_integer(n_estimators, "n_estimators", min_value=1)
        if max_depth is not None:
            check_is_integer(max_depth, "max_depth", min_value=1)

        self._n_estimators = int(n_estimators)
        self._max_depth = int(max_depth) if max_depth is not None else None
        self._honest = honest
        self._rng = ensure_rng(random_state)
        self._random_state = random_state
        self._fitted = False

        self._model_ctrl = None
        self._model_trt = None
        self._cate: np.ndarray | None = None
        self._feature_importances: np.ndarray | None = None
        self._jackknife_mean: float | None = None
        self._jackknife_se: float | None = None

    def _make_forest(self, seed: int) -> object:
        """Create a RandomForestRegressor (lazy sklearn import).

        Parameters
        ----------
        seed : int
            Random seed for this forest.

        Returns
        -------
        object
            sklearn RandomForestRegressor instance.
        """
        try:
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(
                n_estimators=self._n_estimators,
                max_depth=self._max_depth,
                random_state=seed,
                n_jobs=-1,
            )
        except ImportError:  # pragma: no cover
            raise ImportError(
                format_error(
                    "scikit-learn is required for CausalForest.",
                    "install it with: pip install scikit-learn",
                )
            ) from None

    def fit(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
        X_control: ArrayLike,
        X_treatment: ArrayLike,
    ) -> CausalForest:
        """Fit the causal forest on control and treatment data.

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
        CausalForest
            The fitted estimator (self).

        Raises
        ------
        ValueError
            If array lengths are mismatched or arrays are too short.
        """
        y_ctrl = check_array_like(control, "control", min_length=2)
        y_trt = check_array_like(treatment, "treatment", min_length=2)

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
                    "`X_control` and `X_treatment` must have "
                    "the same number of features.",
                    f"X_control has {X_ctrl.shape[1]} features, "
                    f"X_treatment has {X_trt.shape[1]} features.",
                )
            )

        if self._honest:
            self._fit_honest(y_ctrl, y_trt, X_ctrl, X_trt)
        else:
            self._fit_standard(y_ctrl, y_trt, X_ctrl, X_trt)

        # Compute CATE on all data
        X_all = np.vstack([X_ctrl, X_trt])
        self._cate = self._model_trt.predict(X_all) - self._model_ctrl.predict(X_all)

        # Feature importances: average from both models
        imp_ctrl = np.asarray(self._model_ctrl.feature_importances_)
        imp_trt = np.asarray(self._model_trt.feature_importances_)
        self._feature_importances = (imp_ctrl + imp_trt) / 2

        # Jackknife variance estimation for mean CATE
        self._compute_jackknife(y_ctrl, y_trt, X_ctrl, X_trt, X_all)

        self._fitted = True
        return self

    def _fit_honest(
        self,
        y_ctrl: np.ndarray,
        y_trt: np.ndarray,
        X_ctrl: np.ndarray,
        X_trt: np.ndarray,
    ) -> None:
        """Fit with honest splitting: train/estimation split."""
        seed = int(self._rng.integers(0, 2**31))

        # Split control data
        n_ctrl = len(y_ctrl)
        idx_ctrl = self._rng.permutation(n_ctrl)
        mid_ctrl = n_ctrl // 2
        train_ctrl = idx_ctrl[:mid_ctrl]
        est_ctrl = idx_ctrl[mid_ctrl:]

        # Split treatment data
        n_trt = len(y_trt)
        idx_trt = self._rng.permutation(n_trt)
        mid_trt = n_trt // 2
        train_trt = idx_trt[:mid_trt]
        est_trt = idx_trt[mid_trt:]

        # Fit tree structure on training split
        model_ctrl = self._make_forest(seed)
        model_trt = self._make_forest(seed + 1)

        model_ctrl.fit(X_ctrl[train_ctrl], y_ctrl[train_ctrl])
        model_trt.fit(X_trt[train_trt], y_trt[train_trt])

        # Re-estimate leaf values using estimation split
        # For honest estimation, we get leaf assignments from training trees
        # and recompute leaf means from the estimation data
        self._honest_refit(model_ctrl, X_ctrl, y_ctrl, est_ctrl)
        self._honest_refit(model_trt, X_trt, y_trt, est_trt)

        self._model_ctrl = model_ctrl
        self._model_trt = model_trt

    def _honest_refit(
        self,
        model: object,
        X: np.ndarray,
        y: np.ndarray,
        est_indices: np.ndarray,
    ) -> None:
        """Re-estimate leaf values using held-out estimation data.

        Modifies the model's trees in place by updating leaf values
        based on the estimation sample.

        Parameters
        ----------
        model : RandomForestRegressor
            Fitted model whose leaf values will be updated.
        X : np.ndarray
            Full feature matrix.
        y : np.ndarray
            Full outcome array.
        est_indices : np.ndarray
            Indices of the estimation sample.
        """
        X_est = X[est_indices]
        y_est = y[est_indices]

        for tree in model.estimators_:
            # Get leaf assignments for estimation data
            leaf_ids = tree.apply(X_est)
            unique_leaves = np.unique(leaf_ids)

            # Update leaf values
            tree_obj = tree.tree_
            for leaf_id in unique_leaves:
                mask = leaf_ids == leaf_id
                if np.sum(mask) > 0:
                    tree_obj.value[leaf_id, 0, 0] = float(np.mean(y_est[mask]))

    def _fit_standard(
        self,
        y_ctrl: np.ndarray,
        y_trt: np.ndarray,
        X_ctrl: np.ndarray,
        X_trt: np.ndarray,
    ) -> None:
        """Fit without honest splitting (standard T-learner)."""
        seed = int(self._rng.integers(0, 2**31))

        model_ctrl = self._make_forest(seed)
        model_trt = self._make_forest(seed + 1)

        model_ctrl.fit(X_ctrl, y_ctrl)
        model_trt.fit(X_trt, y_trt)

        self._model_ctrl = model_ctrl
        self._model_trt = model_trt

    def _compute_jackknife(
        self,
        y_ctrl: np.ndarray,
        y_trt: np.ndarray,
        X_ctrl: np.ndarray,
        X_trt: np.ndarray,
        X_all: np.ndarray,
    ) -> None:
        """Compute jackknife variance estimate for the mean CATE.

        Uses leave-one-tree-out jackknife for computational efficiency.

        Parameters
        ----------
        y_ctrl, y_trt : np.ndarray
            Outcome arrays.
        X_ctrl, X_trt : np.ndarray
            Feature matrices.
        X_all : np.ndarray
            Combined feature matrix.
        """
        n_trees = len(self._model_ctrl.estimators_)
        if n_trees < 2:
            self._jackknife_mean = float(np.mean(self._cate))
            self._jackknife_se = float(np.std(self._cate) / np.sqrt(len(self._cate)))
            return

        # Leave-one-tree-out jackknife
        full_pred_ctrl = self._model_ctrl.predict(X_all)
        full_pred_trt = self._model_trt.predict(X_all)
        full_cate = full_pred_trt - full_pred_ctrl
        full_mean = float(np.mean(full_cate))

        jackknife_means = []
        for k in range(n_trees):
            # Predict without tree k
            pred_ctrl_k = np.zeros(X_all.shape[0])
            pred_trt_k = np.zeros(X_all.shape[0])

            for j, (tree_c, tree_t) in enumerate(
                zip(
                    self._model_ctrl.estimators_,
                    self._model_trt.estimators_,
                    strict=True,
                )
            ):
                if j == k:
                    continue
                pred_ctrl_k += tree_c.predict(X_all)
                pred_trt_k += tree_t.predict(X_all)

            pred_ctrl_k /= n_trees - 1
            pred_trt_k /= n_trees - 1

            cate_k = pred_trt_k - pred_ctrl_k
            jackknife_means.append(float(np.mean(cate_k)))

        jackknife_means = np.array(jackknife_means)

        # Jackknife SE
        self._jackknife_mean = full_mean
        self._jackknife_se = float(
            np.sqrt(
                (n_trees - 1) / n_trees * np.sum((jackknife_means - full_mean) ** 2)
            )
        )

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
            If the forest has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError(
                format_error(
                    "CausalForest must be fitted before calling predict().",
                    "call .fit() first.",
                )
            )

        X_arr = np.asarray(X, dtype="float64")
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        return self._model_trt.predict(X_arr) - self._model_ctrl.predict(X_arr)

    def result(self) -> CausalForestResult:
        """Return the causal forest estimation result.

        Returns
        -------
        CausalForestResult
            Frozen dataclass with CATE summary and jackknife CIs.

        Raises
        ------
        RuntimeError
            If the forest has not been fitted.
        """
        if not self._fitted or self._cate is None:
            raise RuntimeError(
                format_error(
                    "CausalForest must be fitted before calling result().",
                    "call .fit() first.",
                )
            )

        from scipy.stats import norm

        z = float(norm.ppf(0.975))
        mean_cate = (
            self._jackknife_mean
            if self._jackknife_mean is not None
            else float(np.mean(self._cate))
        )
        se = self._jackknife_se if self._jackknife_se is not None else 0.0

        return CausalForestResult(
            mean_cate=mean_cate,
            cate_std=float(np.std(self._cate)),
            feature_importances=self._feature_importances.tolist(),
            ci_lower=mean_cate - z * se,
            ci_upper=mean_cate + z * se,
        )
