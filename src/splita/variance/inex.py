"""In-Experiment Variance Reduction (InExperimentVR).

Uses control group in-experiment data as a covariate for variance reduction.
Unlike CUPED, the covariate comes from *during* the experiment (valid because
the control group is unaffected by the treatment).

References
----------
.. [1] Deng, A. et al.  "In-experiment variance reduction using control data."
       KDD 2023.
"""

from __future__ import annotations

import warnings

import numpy as np

from splita._validation import (
    check_array_like,
    check_same_length,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


class InExperimentVR:
    """Reduce metric variance using in-experiment control-group covariates.

    In-experiment variance reduction exploits the fact that the control
    group is unaffected by the treatment: a metric measured on the control
    group during the experiment is a valid covariate. This provides the
    same mechanics as CUPED but without requiring pre-experiment data.

    Attributes
    ----------
    theta_ : float
        Fitted adjustment coefficient (set after :meth:`fit`).
    variance_reduction_ : float
        Fraction of variance explained by the covariate (R-squared).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> base = rng.normal(10, 2, size=200)
    >>> ctrl = base[:100] + rng.normal(0, 1, 100)
    >>> trt = base[100:] + 0.5 + rng.normal(0, 1, 100)
    >>> cov_ctrl = base[:100] + rng.normal(0, 0.5, 100)
    >>> cov_trt = base[100:] + rng.normal(0, 0.5, 100)
    >>> vr = InExperimentVR()
    >>> ctrl_adj, trt_adj = vr.fit_transform(ctrl, trt, cov_ctrl, cov_trt)
    >>> vr.variance_reduction_ > 0.1
    True
    """

    def __init__(self) -> None:
        self.theta_: float
        self.variance_reduction_: float
        self._x_pool_mean: float
        self._is_fitted = False

    def fit(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
        control_covariate: ArrayLike,
        treatment_covariate: ArrayLike,
    ) -> InExperimentVR:
        """Learn the adjustment coefficient from data.

        Parameters
        ----------
        control : array-like
            Post-experiment outcome for the control group.
        treatment : array-like
            Post-experiment outcome for the treatment group.
        control_covariate : array-like
            In-experiment covariate for the control group.
        treatment_covariate : array-like
            In-experiment covariate for the treatment group.

        Returns
        -------
        InExperimentVR
            The fitted instance (for method chaining).

        Raises
        ------
        ValueError
            If arrays have mismatched lengths or zero-variance covariate.
        """
        ctrl = check_array_like(control, "control", min_length=2)
        trt = check_array_like(treatment, "treatment", min_length=2)
        cov_ctrl = check_array_like(control_covariate, "control_covariate", min_length=2)
        cov_trt = check_array_like(treatment_covariate, "treatment_covariate", min_length=2)

        check_same_length(ctrl, cov_ctrl, "control", "control_covariate")
        check_same_length(trt, cov_trt, "treatment", "treatment_covariate")

        y = np.concatenate([ctrl, trt])
        x = np.concatenate([cov_ctrl, cov_trt])

        self._x_pool_mean = float(np.mean(x))

        var_x = float(np.var(x, ddof=1))
        if var_x == 0.0:
            raise ValueError(
                format_error(
                    "Covariate has zero variance — adjustment is undefined.",
                    detail="all covariate values are identical.",
                    hint="use a covariate with variation across observations.",
                )
            )

        cov_yx = float(np.cov(y, x, ddof=1)[0, 1])
        self.theta_ = cov_yx / var_x

        corr_matrix = np.corrcoef(y, x)
        correlation = float(corr_matrix[0, 1])
        self.variance_reduction_ = correlation**2

        if abs(correlation) < 0.3:
            warnings.warn(
                f"Low correlation ({correlation:.2f}) between metric and "
                f"covariate. Variance reduction will be minimal.",
                RuntimeWarning,
                stacklevel=2,
            )

        self._is_fitted = True
        return self

    def transform(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
        control_covariate: ArrayLike,
        treatment_covariate: ArrayLike,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the adjustment using the fitted theta.

        Parameters
        ----------
        control : array-like
            Outcome for the control group.
        treatment : array-like
            Outcome for the treatment group.
        control_covariate : array-like
            In-experiment covariate for the control group.
        treatment_covariate : array-like
            In-experiment covariate for the treatment group.

        Returns
        -------
        tuple of np.ndarray
            ``(control_adjusted, treatment_adjusted)``.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        """
        if not self._is_fitted:
            raise RuntimeError(
                format_error(
                    "InExperimentVR must be fitted before calling transform().",
                    detail="theta_ has not been estimated yet.",
                    hint="call fit() or fit_transform() first.",
                )
            )

        ctrl = check_array_like(control, "control", min_length=2)
        trt = check_array_like(treatment, "treatment", min_length=2)
        cov_ctrl = check_array_like(control_covariate, "control_covariate", min_length=2)
        cov_trt = check_array_like(treatment_covariate, "treatment_covariate", min_length=2)

        check_same_length(ctrl, cov_ctrl, "control", "control_covariate")
        check_same_length(trt, cov_trt, "treatment", "treatment_covariate")

        ctrl_adj = ctrl - self.theta_ * (cov_ctrl - self._x_pool_mean)
        trt_adj = trt - self.theta_ * (cov_trt - self._x_pool_mean)

        return ctrl_adj, trt_adj

    def fit_transform(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
        control_covariate: ArrayLike,
        treatment_covariate: ArrayLike,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit and apply the adjustment in one step.

        Parameters
        ----------
        control : array-like
            Outcome for the control group.
        treatment : array-like
            Outcome for the treatment group.
        control_covariate : array-like
            In-experiment covariate for the control group.
        treatment_covariate : array-like
            In-experiment covariate for the treatment group.

        Returns
        -------
        tuple of np.ndarray
            ``(control_adjusted, treatment_adjusted)``.
        """
        self.fit(control, treatment, control_covariate, treatment_covariate)
        return self.transform(control, treatment, control_covariate, treatment_covariate)
