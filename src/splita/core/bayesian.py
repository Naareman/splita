"""BayesianExperiment — Bayesian A/B test analysis.

Computes P(B>A), expected loss, credible intervals, and ROPE
(Region of Practical Equivalence) using Monte Carlo posterior sampling.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from splita._types import BayesianResult
from splita._utils import auto_detect_metric, ensure_rng, relative_lift
from splita._validation import (
    check_array_like,
    check_one_of,
    format_error,
)

ArrayLike = list | tuple | np.ndarray

_VALID_METRICS = ["auto", "conversion", "continuous"]

# ─── Default priors ──────────────────────────────────────────────────

_DEFAULT_CONVERSION_PRIOR = {"alpha": 1.0, "beta": 1.0}
_DEFAULT_CONTINUOUS_PRIOR = {"mu": 0.0, "kappa": 0.001, "alpha": 1.0, "beta": 1.0}


class BayesianExperiment:
    """Run a Bayesian A/B test on two groups of observations.

    Parameters
    ----------
    control : array-like
        Observations from the control group.
    treatment : array-like
        Observations from the treatment group.
    metric : {'auto', 'conversion', 'continuous'}, default 'auto'
        Type of metric.  ``'auto'`` infers from the data.
    prior : dict or None, default None
        Prior hyperparameters.  ``None`` uses non-informative defaults.

        - Conversion: ``{"alpha": 1, "beta": 1}`` (uniform Beta).
        - Continuous: ``{"mu": 0, "kappa": 0.001, "alpha": 1, "beta": 1}``
          (vague Normal-Inverse-Gamma).
    n_samples : int, default 50_000
        Number of Monte Carlo posterior samples for inference.
    rope : tuple of (float, float) or None, default None
        Region of Practical Equivalence, e.g. ``(-0.01, 0.01)``.
        If set, computes P(effect in ROPE).
    random_state : int, Generator, or None, default None
        Seed for reproducibility.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> ctrl = rng.binomial(1, 0.10, 5000)
    >>> trt  = rng.binomial(1, 0.12, 5000)
    >>> result = BayesianExperiment(ctrl, trt, random_state=0).run()
    >>> result.metric
    'conversion'
    """

    def __init__(
        self,
        control: ArrayLike,
        treatment: ArrayLike,
        *,
        metric: Literal["auto", "conversion", "continuous"] = "auto",
        prior: dict | None = None,
        n_samples: int = 50_000,
        rope: tuple[float, float] | None = None,
        random_state: int | np.random.Generator | None = None,
    ):
        # ── validate metric ──────────────────────────────────────────
        check_one_of(metric, "metric", _VALID_METRICS)

        # ── validate n_samples ───────────────────────────────────────
        if n_samples < 1000:
            raise ValueError(
                format_error(
                    f"`n_samples` must be >= 1000, got {n_samples}.",
                    "too few Monte Carlo samples for reliable inference.",
                    "typical values are 10000-100000.",
                )
            )

        # ── validate rope ────────────────────────────────────────────
        if rope is not None and rope[0] >= rope[1]:
            raise ValueError(
                format_error(
                    f"`rope` lower bound must be < upper bound, got {rope}.",
                    f"rope[0]={rope[0]} is not less than rope[1]={rope[1]}.",
                    "pass rope=(-0.01, 0.01) for a symmetric ROPE.",
                )
            )

        # ── convert & clean arrays ───────────────────────────────────
        self._control = check_array_like(control, "control", min_length=2)
        self._treatment = check_array_like(treatment, "treatment", min_length=2)

        # ── infer metric ─────────────────────────────────────────────
        if metric == "auto":
            self._metric = auto_detect_metric(np.concatenate([self._control, self._treatment]))
        else:
            self._metric = metric

        # ── store config ─────────────────────────────────────────────
        self._n_samples = n_samples
        self._rope = rope
        self._rng = ensure_rng(random_state)

        # ── set prior ────────────────────────────────────────────────
        if prior is not None:
            self._prior = dict(prior)
        elif self._metric == "conversion":
            self._prior = dict(_DEFAULT_CONVERSION_PRIOR)
        else:
            self._prior = dict(_DEFAULT_CONTINUOUS_PRIOR)

    # ── posterior sampling ────────────────────────────────────────────

    def _sample_conversion(self) -> tuple[np.ndarray, np.ndarray]:
        """Draw posterior samples for conversion (Beta-Binomial) model."""
        alpha_prior = self._prior["alpha"]
        beta_prior = self._prior["beta"]

        # Control posterior
        s_ctrl = float(np.sum(self._control))
        n_ctrl = len(self._control)
        alpha_ctrl = alpha_prior + s_ctrl
        beta_ctrl = beta_prior + (n_ctrl - s_ctrl)

        # Treatment posterior
        s_trt = float(np.sum(self._treatment))
        n_trt = len(self._treatment)
        alpha_trt = alpha_prior + s_trt
        beta_trt = beta_prior + (n_trt - s_trt)

        samples_ctrl = self._rng.beta(alpha_ctrl, beta_ctrl, size=self._n_samples)
        samples_trt = self._rng.beta(alpha_trt, beta_trt, size=self._n_samples)

        return samples_ctrl, samples_trt

    def _sample_continuous(self) -> tuple[np.ndarray, np.ndarray]:
        """Draw posterior samples for continuous (Normal-Inverse-Gamma) model.

        NIG prior: mu, kappa, alpha, beta
        Posterior update:
            n = len(data), x_bar = mean(data), s2 = sum((x - x_bar)^2)
            kappa_n = kappa + n
            mu_n = (kappa * mu + n * x_bar) / kappa_n
            alpha_n = alpha + n / 2
            beta_n = beta + s2 / 2 + kappa * n * (x_bar - mu)^2 / (2 * kappa_n)
        Sample: precision ~ Gamma(alpha_n, 1/beta_n),
        then mean ~ Normal(mu_n, 1/(kappa_n * precision))
        """
        mu_prior = self._prior["mu"]
        kappa_prior = self._prior["kappa"]
        alpha_prior = self._prior["alpha"]
        beta_prior = self._prior["beta"]

        def _draw(data: np.ndarray) -> np.ndarray:
            n = len(data)
            x_bar = float(np.mean(data))
            s2 = float(np.sum((data - x_bar) ** 2))

            kappa_n = kappa_prior + n
            mu_n = (kappa_prior * mu_prior + n * x_bar) / kappa_n
            alpha_n = alpha_prior + n / 2.0
            beta_n = (
                beta_prior + s2 / 2.0 + kappa_prior * n * (x_bar - mu_prior) ** 2 / (2.0 * kappa_n)
            )

            # Sample precision from Gamma, then mean from Normal
            precision = self._rng.gamma(alpha_n, 1.0 / beta_n, size=self._n_samples)
            # Avoid zero precision
            precision = np.maximum(precision, 1e-300)
            means = self._rng.normal(mu_n, 1.0 / np.sqrt(kappa_n * precision))
            return means

        samples_ctrl = _draw(self._control)
        samples_trt = _draw(self._treatment)
        return samples_ctrl, samples_trt

    # ── public API ───────────────────────────────────────────────────

    def run(self) -> BayesianResult:
        """Execute the Bayesian analysis and return results.

        Returns
        -------
        BayesianResult
            Frozen dataclass with posterior inference outputs.
        """
        if self._metric == "conversion":
            samples_ctrl, samples_trt = self._sample_conversion()
        else:
            samples_ctrl, samples_trt = self._sample_continuous()

        diff = samples_trt - samples_ctrl

        # Core metrics
        prob_b_beats_a = float(np.mean(samples_trt > samples_ctrl))
        expected_loss_a = float(np.mean(np.maximum(samples_trt - samples_ctrl, 0.0)))
        expected_loss_b = float(np.mean(np.maximum(samples_ctrl - samples_trt, 0.0)))
        lift = float(np.mean(diff))

        # Relative lift
        ctrl_mean = float(np.mean(samples_ctrl))
        trt_mean = float(np.mean(samples_trt))
        rel_lift = relative_lift(ctrl_mean, trt_mean)

        # 95% credible interval
        credible_level = 0.95
        tail = (1.0 - credible_level) / 2.0
        ci_lower = float(np.percentile(diff, 100.0 * tail))
        ci_upper = float(np.percentile(diff, 100.0 * (1.0 - tail)))

        # ROPE
        prob_in_rope: float | None = None
        if self._rope is not None:
            prob_in_rope = float(np.mean((diff > self._rope[0]) & (diff < self._rope[1])))

        return BayesianResult(
            prob_b_beats_a=prob_b_beats_a,
            expected_loss_a=expected_loss_a,
            expected_loss_b=expected_loss_b,
            lift=lift,
            relative_lift=rel_lift,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            credible_level=credible_level,
            control_mean=ctrl_mean,
            treatment_mean=trt_mean,
            prob_in_rope=prob_in_rope,
            rope=self._rope,
            metric=self._metric,
            control_n=len(self._control),
            treatment_n=len(self._treatment),
        )
