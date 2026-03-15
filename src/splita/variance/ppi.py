"""Prediction-Powered Inference (PPI).

Uses ML predictions to augment small labeled datasets for valid statistical
inference (Angelopoulos et al. 2023).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from splita._types import PPIResult
from splita._validation import check_array_like, check_same_length, format_error

ArrayLike = list | tuple | np.ndarray


class PredictionPoweredInference:
    """Prediction-powered inference for mean estimation.

    Combines a small labeled dataset with a large unlabeled dataset
    (where only ML predictions are available) to produce valid
    confidence intervals with reduced width.

    Parameters
    ----------
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n_labeled, n_unlabeled = 100, 5000
    >>> true_mean = 5.0
    >>> labeled_y = rng.normal(true_mean, 1, n_labeled)
    >>> labeled_pred = labeled_y + rng.normal(0, 0.5, n_labeled)
    >>> unlabeled_pred = rng.normal(true_mean, 1.1, n_unlabeled) + rng.normal(0, 0.5, n_unlabeled)
    >>> ppi = PredictionPoweredInference()
    >>> r = ppi.fit(labeled_y, labeled_pred, unlabeled_pred)
    >>> abs(r.mean_estimate - true_mean) < 1.0
    True
    """

    def __init__(self, *, alpha: float = 0.05) -> None:
        if not 0 < alpha < 1:
            raise ValueError(
                format_error(
                    "`alpha` must be in (0, 1), got {}.".format(alpha),
                    "alpha represents the significance level.",
                    "typical values are 0.05, 0.01, or 0.10.",
                )
            )
        self._alpha = alpha

    def fit(
        self,
        labeled_outcome: ArrayLike,
        labeled_predictions: ArrayLike,
        unlabeled_predictions: ArrayLike,
    ) -> PPIResult:
        """Compute the PPI mean estimate with valid confidence intervals.

        Parameters
        ----------
        labeled_outcome : array-like
            True outcomes for the labeled subset.
        labeled_predictions : array-like
            ML predictions for the labeled subset.
        unlabeled_predictions : array-like
            ML predictions for the unlabeled subset.

        Returns
        -------
        PPIResult
            Frozen dataclass with mean estimate and confidence interval.

        Raises
        ------
        TypeError
            If inputs cannot be converted to numeric arrays.
        ValueError
            If labeled arrays differ in length or have too few elements.
        """
        y = check_array_like(labeled_outcome, "labeled_outcome", min_length=2)
        f_labeled = check_array_like(
            labeled_predictions, "labeled_predictions", min_length=2
        )
        f_unlabeled = check_array_like(
            unlabeled_predictions, "unlabeled_predictions", min_length=2
        )
        check_same_length(y, f_labeled, "labeled_outcome", "labeled_predictions")

        n = len(y)
        N = len(f_unlabeled)

        # PPI estimator: theta_ppi = mean(f_unlabeled) + mean(y - f_labeled)
        # This is the "rectifier" approach from Angelopoulos et al.
        rectifier = y - f_labeled
        mean_rectifier = float(np.mean(rectifier))
        mean_f_unlabeled = float(np.mean(f_unlabeled))

        mean_estimate = mean_f_unlabeled + mean_rectifier

        # Variance of PPI estimator
        # Var(theta_ppi) = Var(mean(f_unlabeled)) + Var(mean(rectifier))
        var_rectifier = float(np.var(rectifier, ddof=1)) / n
        var_f_unlabeled = float(np.var(f_unlabeled, ddof=1)) / N

        se = float(np.sqrt(var_rectifier + var_f_unlabeled))

        z_crit = float(norm.ppf(1 - self._alpha / 2))
        ci_lower = mean_estimate - z_crit * se
        ci_upper = mean_estimate + z_crit * se

        return PPIResult(
            mean_estimate=mean_estimate,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_labeled=n,
            n_unlabeled=N,
        )
