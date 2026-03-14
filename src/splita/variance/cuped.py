"""CUPED — Controlled-experiment Using Pre-Experiment Data.

Reduces variance by regressing out pre-experiment covariates.  Can reduce
required sample size by 20-65%.  Used at Microsoft, Netflix, Airbnb, Uber,
and Meta.

References
----------
.. [1] Deng, A., Xu, Y., Kohavi, R. & Walker, T.  "Improving the Sensitivity
       of Online Controlled Experiments by Utilizing Pre-Experiment Data."
       WSDM 2013.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np

from splita._validation import (
    check_array_like,
    check_same_length,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


class CUPED:
    """Reduce metric variance using pre-experiment covariates.

    CUPED (Controlled-experiment Using Pre-Experiment Data) adjusts
    post-experiment observations by subtracting the part that can be
    predicted from a pre-experiment covariate.  When the pre/post
    correlation is high, this dramatically reduces variance and
    therefore the sample size needed to detect an effect.

    Parameters
    ----------
    covariate : {'auto', 'custom'} or np.ndarray, default 'auto'
        - ``'auto'``: use pre-experiment values of the same metric
          (the most common pattern). ``control_pre`` and ``treatment_pre``
          must be supplied to :meth:`fit` and :meth:`transform`.
        - Pass a 1-D NumPy array to use a custom covariate.
    theta : float or None, default None
        CUPED adjustment coefficient.  ``None`` means estimate it from
        the data as ``Cov(Y, X) / Var(X)`` (the optimal value).

    Attributes
    ----------
    theta_ : float
        Fitted adjustment coefficient (set after :meth:`fit`).
    correlation_ : float
        Pearson correlation between the metric and the covariate.
    variance_reduction_ : float
        Fraction of variance explained (R²), equal to ``correlation_²``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> pre = rng.normal(10, 2, size=200)
    >>> ctrl = pre[:100] + rng.normal(0, 1, 100)
    >>> trt = pre[100:] + 0.5 + rng.normal(0, 1, 100)
    >>> cuped = CUPED()
    >>> ctrl_adj, trt_adj = cuped.fit_transform(ctrl, trt, pre[:100], pre[100:])
    >>> cuped.variance_reduction_ > 0.3
    True
    """

    def __init__(
        self,
        *,
        covariate: Literal["auto", "custom"] | np.ndarray = "auto",
        theta: float | None = None,
    ) -> None:
        if isinstance(covariate, str) and covariate not in ("auto", "custom"):
            raise ValueError(
                format_error(
                    f"`covariate` must be 'auto', 'custom', or a numpy array, "
                    f"got {covariate!r}.",
                    hint="use 'auto' for pre-experiment metric values, or pass "
                    "a 1-D numpy array.",
                )
            )
        self.covariate = covariate
        self.theta = theta

        # Fitted attributes (set by fit)
        self.theta_: float
        self.correlation_: float
        self.variance_reduction_: float
        self._x_pool_mean: float
        self._is_fitted = False

    # ── fit ──────────────────────────────────────────────────────────

    def fit(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
        control_pre: ArrayLike | None = None,
        treatment_pre: ArrayLike | None = None,
    ) -> CUPED:
        """Learn the CUPED adjustment coefficient from data.

        Parameters
        ----------
        control : array-like
            Post-experiment observations for the control group.
        treatment : array-like
            Post-experiment observations for the treatment group.
        control_pre : array-like or None
            Pre-experiment covariate values for the control group.
            Required when ``covariate='auto'``.
        treatment_pre : array-like or None
            Pre-experiment covariate values for the treatment group.
            Required when ``covariate='auto'``.

        Returns
        -------
        CUPED
            The fitted instance (for method chaining).

        Raises
        ------
        ValueError
            If pre-experiment data is missing when ``covariate='auto'``,
            or if array lengths are mismatched.
        """
        control_arr = check_array_like(control, "control", min_length=2)
        treatment_arr = check_array_like(treatment, "treatment", min_length=2)

        x_control, x_treatment = self._resolve_covariate(
            control_arr,
            treatment_arr,
            control_pre,
            treatment_pre,
        )

        y = np.concatenate([control_arr, treatment_arr])
        x = np.concatenate([x_control, x_treatment])

        # Store pooled covariate mean for transform
        self._x_pool_mean = float(np.mean(x))

        # Compute or use manual theta
        var_x = float(np.var(x, ddof=1))
        if var_x == 0.0:
            raise ValueError(
                format_error(
                    "Covariate has zero variance — CUPED adjustment is undefined.",
                    detail="all covariate values are identical.",
                    hint="use a covariate with variation across observations.",
                )
            )

        if self.theta is not None:
            self.theta_ = self.theta
        else:
            cov_yx = float(np.cov(y, x, ddof=1)[0, 1])
            self.theta_ = cov_yx / var_x

        # Correlation and variance reduction
        corr_matrix = np.corrcoef(y, x)
        self.correlation_ = float(corr_matrix[0, 1])
        self.variance_reduction_ = self.correlation_**2

        if abs(self.correlation_) < 0.3:
            warnings.warn(
                f"Low correlation ({self.correlation_:.2f}) between metric and "
                f"covariate. CUPED will provide minimal variance reduction.",
                RuntimeWarning,
                stacklevel=2,
            )

        self._is_fitted = True
        return self

    # ── transform ───────────────────────────────────────────────────

    def transform(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
        control_pre: ArrayLike | None = None,
        treatment_pre: ArrayLike | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the CUPED adjustment using the fitted theta.

        Parameters
        ----------
        control : array-like
            Post-experiment observations for the control group.
        treatment : array-like
            Post-experiment observations for the treatment group.
        control_pre : array-like or None
            Pre-experiment covariate values for the control group.
            Required when ``covariate='auto'``.
        treatment_pre : array-like or None
            Pre-experiment covariate values for the treatment group.
            Required when ``covariate='auto'``.

        Returns
        -------
        tuple of np.ndarray
            ``(control_adjusted, treatment_adjusted)`` — the CUPED-adjusted
            observations for each group.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        ValueError
            If pre-experiment data is missing or array lengths mismatch.
        """
        if not self._is_fitted:
            raise RuntimeError(
                format_error(
                    "CUPED must be fitted before calling transform().",
                    detail="theta_ has not been estimated yet.",
                    hint="call fit() or fit_transform() first.",
                )
            )

        control_arr = check_array_like(control, "control", min_length=2)
        treatment_arr = check_array_like(treatment, "treatment", min_length=2)

        x_control, x_treatment = self._resolve_covariate(
            control_arr,
            treatment_arr,
            control_pre,
            treatment_pre,
        )

        # Y_adj = Y - theta * (X - mean(X_pool))
        control_adj = control_arr - self.theta_ * (x_control - self._x_pool_mean)
        treatment_adj = treatment_arr - self.theta_ * (x_treatment - self._x_pool_mean)

        return control_adj, treatment_adj

    # ── fit_transform ───────────────────────────────────────────────

    def fit_transform(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
        control_pre: ArrayLike | None = None,
        treatment_pre: ArrayLike | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit CUPED and apply the adjustment in one step.

        Convenience method equivalent to calling :meth:`fit` followed
        by :meth:`transform` with the same arguments.

        Parameters
        ----------
        control : array-like
            Post-experiment observations for the control group.
        treatment : array-like
            Post-experiment observations for the treatment group.
        control_pre : array-like or None
            Pre-experiment covariate values for the control group.
            Required when ``covariate='auto'``.
        treatment_pre : array-like or None
            Pre-experiment covariate values for the treatment group.
            Required when ``covariate='auto'``.

        Returns
        -------
        tuple of np.ndarray
            ``(control_adjusted, treatment_adjusted)``.
        """
        self.fit(control, treatment, control_pre, treatment_pre)
        return self.transform(control, treatment, control_pre, treatment_pre)

    # ── private helpers ─────────────────────────────────────────────

    def _resolve_covariate(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        control_pre: ArrayLike | None,
        treatment_pre: ArrayLike | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resolve covariate arrays from the constructor setting.

        Returns validated ``(x_control, x_treatment)`` arrays.
        """
        if isinstance(self.covariate, np.ndarray):
            # Custom covariate passed as array at init time
            x = check_array_like(self.covariate, "covariate", min_length=2)
            n_ctrl = len(control)
            n_total = len(control) + len(treatment)
            if len(x) != n_total:
                raise ValueError(
                    format_error(
                        f"`covariate` array length must equal total observations "
                        f"({n_total}), got {len(x)}.",
                        hint="pass a covariate array with one value per observation "
                        "(control + treatment concatenated).",
                    )
                )
            return x[:n_ctrl], x[n_ctrl:]

        # String mode: "auto" or "custom"
        if self.covariate == "auto" and (control_pre is None or treatment_pre is None):
            raise ValueError(
                format_error(
                    "`control_pre` and `treatment_pre` are required when "
                    "covariate='auto'.",
                    detail="CUPED needs pre-experiment data to adjust for.",
                    hint="pass pre-experiment metric values, or set "
                    "covariate to a custom array.",
                )
            )

        # covariate="custom" also requires pre arrays
        if self.covariate == "custom" and (
            control_pre is None or treatment_pre is None
        ):
            raise ValueError(
                format_error(
                    "`control_pre` and `treatment_pre` are required when "
                    "covariate='custom'.",
                    detail="CUPED needs covariate data to adjust for.",
                    hint="pass the custom covariate values for each group.",
                )
            )

        x_control = check_array_like(control_pre, "control_pre", min_length=2)
        x_treatment = check_array_like(treatment_pre, "treatment_pre", min_length=2)

        check_same_length(control, x_control, "control", "control_pre")
        check_same_length(treatment, x_treatment, "treatment", "treatment_pre")

        return x_control, x_treatment
