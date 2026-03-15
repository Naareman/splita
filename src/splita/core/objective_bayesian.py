"""ObjectiveBayesianExperiment — empirical Bayes for A/B tests.

Learns a Normal prior from historical experiments, then applies
conjugate Bayesian updating for the current test.

References
----------
.. [1] Microsoft/Bing (2019).  "Objective Bayesian A/B testing using
       empirical Bayes priors."
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from splita._types import ObjectiveBayesianResult
from splita._validation import (
    check_array_like,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


class ObjectiveBayesianExperiment:
    """Bayesian A/B test with a prior learned from historical experiments.

    Uses empirical Bayes: the prior distribution is estimated from past
    experiment effect sizes, producing data-driven shrinkage.  With a
    vague prior (large variance), the result converges to the frequentist
    estimate.

    Parameters
    ----------
    alpha : float, default 0.05
        Credible interval level (``1 - alpha``).

    Attributes
    ----------
    prior_mean_ : float
        Mean of the learned prior (set after :meth:`fit_prior`).
    prior_std_ : float
        Standard deviation of the learned prior.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> historical = [0.01, -0.02, 0.03, 0.0, -0.01, 0.02, 0.01, -0.005]
    >>> exp = ObjectiveBayesianExperiment()
    >>> exp.fit_prior(historical)
    >>> ctrl = rng.normal(10, 2, size=500)
    >>> trt = rng.normal(10.3, 2, size=500)
    >>> result = exp.run(ctrl, trt)
    >>> result.posterior_mean > 0
    True
    """

    def __init__(self, *, alpha: float = 0.05) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(
                format_error(
                    f"`alpha` must be in (0, 1), got {alpha}.",
                    hint="typical values are 0.05, 0.01, or 0.10.",
                )
            )
        self._alpha = alpha
        self.prior_mean_: float = 0.0
        self.prior_std_: float = 1e6  # vague prior by default
        self._prior_fitted = False

    def fit_prior(self, historical_effects: list[float]) -> ObjectiveBayesianExperiment:
        """Learn a Normal prior from historical experiment effects.

        Parameters
        ----------
        historical_effects : list of float
            Effect sizes from past experiments (e.g. lift estimates).

        Returns
        -------
        ObjectiveBayesianExperiment
            The fitted instance (for method chaining).

        Raises
        ------
        ValueError
            If fewer than 2 historical effects are provided, or if all
            effects are identical.
        """
        if len(historical_effects) < 2:
            raise ValueError(
                format_error(
                    "`historical_effects` must have at least 2 elements, "
                    f"got {len(historical_effects)}.",
                    detail="need multiple experiments to estimate a prior.",
                    hint="pass a list of historical effect sizes.",
                )
            )

        effects = np.array(historical_effects, dtype=float)

        if np.all(effects == effects[0]):
            raise ValueError(
                format_error(
                    "`historical_effects` can't all be identical.",
                    detail="zero variance means the prior is degenerate.",
                    hint="provide a diverse set of historical effect sizes.",
                )
            )

        self.prior_mean_ = float(np.mean(effects))
        self.prior_std_ = float(np.std(effects, ddof=1))
        self._prior_fitted = True
        return self

    def run(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
    ) -> ObjectiveBayesianResult:
        """Run the Bayesian analysis with the (learned or default) prior.

        Parameters
        ----------
        control : array-like
            Observations from the control group.
        treatment : array-like
            Observations from the treatment group.

        Returns
        -------
        ObjectiveBayesianResult
            Posterior inference with learned prior.
        """
        ctrl = check_array_like(control, "control", min_length=2)
        trt = check_array_like(treatment, "treatment", min_length=2)

        # Observed effect and its standard error
        n_c, n_t = len(ctrl), len(trt)
        mean_c = float(np.mean(ctrl))
        mean_t = float(np.mean(trt))
        observed_effect = mean_t - mean_c

        var_c = float(np.var(ctrl, ddof=1))
        var_t = float(np.var(trt, ddof=1))
        se = math.sqrt(var_c / n_c + var_t / n_t)

        # Conjugate normal-normal update
        # Prior: effect ~ N(prior_mean, prior_std^2)
        # Likelihood: observed_effect ~ N(true_effect, se^2)
        prior_var = self.prior_std_**2
        data_var = se**2

        # Posterior precision = 1/prior_var + 1/data_var
        if prior_var > 0 and data_var > 0:
            posterior_var = 1.0 / (1.0 / prior_var + 1.0 / data_var)
            posterior_mean = posterior_var * (
                self.prior_mean_ / prior_var + observed_effect / data_var
            )
            shrinkage = posterior_var / data_var  # fraction shrunk toward prior
        else:  # pragma: no cover
            posterior_mean = observed_effect
            posterior_var = data_var
            shrinkage = 0.0

        posterior_std = math.sqrt(posterior_var)

        # Probability that effect is positive
        if posterior_std > 0:
            prob_positive = float(1.0 - norm.cdf(0.0, posterior_mean, posterior_std))
        else:
            prob_positive = 1.0 if posterior_mean > 0 else 0.0  # pragma: no cover

        # Credible interval
        z_crit = float(norm.ppf(1.0 - self._alpha / 2.0))
        ci_lower = posterior_mean - z_crit * posterior_std
        ci_upper = posterior_mean + z_crit * posterior_std

        return ObjectiveBayesianResult(
            prior_mean=self.prior_mean_,
            prior_std=self.prior_std_,
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
            prob_positive=prob_positive,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            shrinkage=shrinkage,
        )
