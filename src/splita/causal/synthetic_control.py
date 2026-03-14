"""Synthetic Control method.

Constructs a synthetic control unit as a weighted combination of donor units
that best approximates the treated unit in the pre-treatment period, then
estimates the treatment effect as the post-treatment divergence.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from splita._types import SyntheticControlResult
from splita._validation import check_array_like, check_in_range, format_error

ArrayLike = list | tuple | np.ndarray


class SyntheticControl:
    """Synthetic Control estimator for causal inference.

    Finds non-negative weights summing to 1 that minimise pre-treatment
    MSE between the treated unit and the weighted combination of donors.
    The post-treatment effect is the difference between the treated unit
    and the synthetic control.

    Parameters
    ----------
    alpha : float, default 0.05
        Significance level (reserved for future CI implementation).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> treated_pre = np.array([10, 11, 12, 13, 14], dtype=float)
    >>> treated_post = np.array([20, 21, 22], dtype=float)
    >>> donors_pre = np.column_stack([
    ...     np.array([10, 11, 12, 13, 14], dtype=float),
    ...     np.array([5, 6, 7, 8, 9], dtype=float),
    ... ])
    >>> donors_post = np.column_stack([
    ...     np.array([15, 16, 17], dtype=float),
    ...     np.array([10, 11, 12], dtype=float),
    ... ])
    >>> sc = SyntheticControl()
    >>> sc.fit(treated_pre, treated_post, donors_pre, donors_post)  # doctest: +ELLIPSIS
    <splita.causal.synthetic_control.SyntheticControl object at ...>
    >>> r = sc.result()
    >>> r.effect > 0
    True
    """

    def __init__(self, *, alpha: float = 0.05) -> None:
        check_in_range(
            alpha,
            "alpha",
            0.0,
            1.0,
            hint="typical values are 0.05, 0.01, or 0.10",
        )
        self._alpha = alpha
        self._result: SyntheticControlResult | None = None

    def fit(
        self,
        treated_pre: ArrayLike,
        treated_post: ArrayLike,
        donors_pre: np.ndarray,
        donors_post: np.ndarray,
    ) -> SyntheticControl:
        """Fit the synthetic control model.

        Parameters
        ----------
        treated_pre : array-like
            Pre-treatment outcomes for the treated unit (length T_pre).
        treated_post : array-like
            Post-treatment outcomes for the treated unit (length T_post).
        donors_pre : np.ndarray
            Pre-treatment outcomes for donor units, shape (T_pre, n_donors).
        donors_post : np.ndarray
            Post-treatment outcomes for donor units, shape (T_post, n_donors).

        Returns
        -------
        SyntheticControl
            The fitted estimator (self).

        Raises
        ------
        TypeError
            If inputs cannot be converted to numeric arrays.
        ValueError
            If dimensions are inconsistent or there are fewer than 1 donor.
        """
        t_pre = check_array_like(treated_pre, "treated_pre", min_length=2)
        t_post = check_array_like(treated_post, "treated_post", min_length=1)

        # Validate donor matrices
        if not isinstance(donors_pre, np.ndarray):
            try:
                donors_pre = np.asarray(donors_pre, dtype=float)
            except (ValueError, TypeError) as exc:
                raise TypeError(
                    format_error(
                        "`donors_pre` can't be converted to a numeric array.",
                        str(exc),
                    )
                ) from exc

        if not isinstance(donors_post, np.ndarray):
            try:
                donors_post = np.asarray(donors_post, dtype=float)
            except (ValueError, TypeError) as exc:
                raise TypeError(
                    format_error(
                        "`donors_post` can't be converted to a numeric array.",
                        str(exc),
                    )
                ) from exc

        if donors_pre.ndim == 1:
            donors_pre = donors_pre.reshape(-1, 1)
        if donors_post.ndim == 1:
            donors_post = donors_post.reshape(-1, 1)

        if donors_pre.ndim != 2:
            raise ValueError(
                format_error(
                    "`donors_pre` must be a 2-D array (T_pre, n_donors).",
                    f"got {donors_pre.ndim}-D array with shape {donors_pre.shape}.",
                )
            )
        if donors_post.ndim != 2:
            raise ValueError(
                format_error(
                    "`donors_post` must be a 2-D array (T_post, n_donors).",
                    f"got {donors_post.ndim}-D array with shape {donors_post.shape}.",
                )
            )

        n_pre_treated = len(t_pre)
        n_post_treated = len(t_post)
        n_pre_donors, n_donors = donors_pre.shape
        n_post_donors, n_donors_post = donors_post.shape

        if n_pre_treated != n_pre_donors:
            raise ValueError(
                format_error(
                    "`treated_pre` and `donors_pre` must have the same "
                    "number of time periods.",
                    f"treated_pre has {n_pre_treated} periods, "
                    f"donors_pre has {n_pre_donors} periods.",
                )
            )
        if n_post_treated != n_post_donors:
            raise ValueError(
                format_error(
                    "`treated_post` and `donors_post` must have the same "
                    "number of time periods.",
                    f"treated_post has {n_post_treated} periods, "
                    f"donors_post has {n_post_donors} periods.",
                )
            )
        if n_donors != n_donors_post:
            raise ValueError(
                format_error(
                    "`donors_pre` and `donors_post` must have the same "
                    "number of donors.",
                    f"donors_pre has {n_donors} donors, "
                    f"donors_post has {n_donors_post} donors.",
                )
            )
        if n_donors < 1:
            raise ValueError(
                format_error(
                    "Need at least 1 donor unit.",
                    f"got {n_donors} donors.",
                )
            )

        # Solve: min_w ||treated_pre - donors_pre @ w||^2
        # s.t.  w >= 0, sum(w) = 1
        def objective(w: np.ndarray) -> float:
            residual = t_pre - donors_pre @ w
            return float(np.sum(residual**2))

        # Constraints
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0)] * n_donors

        # Initial: equal weights
        w0 = np.ones(n_donors) / n_donors

        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        weights = result.x
        # Clamp small negatives from numerical noise
        weights = np.maximum(weights, 0.0)
        weights = weights / np.sum(weights)

        # Pre-treatment fit
        synthetic_pre = donors_pre @ weights
        pre_rmse = float(np.sqrt(np.mean((t_pre - synthetic_pre) ** 2)))

        # Post-treatment effect
        synthetic_post = donors_post @ weights
        effect_series = (t_post - synthetic_post).tolist()
        effect = float(np.mean(effect_series))

        # Donor contributions
        donor_contributions = {i: float(w) for i, w in enumerate(weights)}

        self._result = SyntheticControlResult(
            effect=effect,
            weights=tuple(float(w) for w in weights),
            pre_treatment_rmse=pre_rmse,
            donor_contributions=donor_contributions,
            effect_series=effect_series,
        )
        return self

    def result(self) -> SyntheticControlResult:
        """Return the synthetic control result.

        Returns
        -------
        SyntheticControlResult
            The estimated treatment effect and model diagnostics.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called.
        """
        if self._result is None:
            raise RuntimeError(
                format_error(
                    "SyntheticControl must be fitted before calling result().",
                    "call .fit() with pre/post data first.",
                )
            )
        return self._result
