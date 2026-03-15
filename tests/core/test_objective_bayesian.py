"""Tests for ObjectiveBayesianExperiment."""

from __future__ import annotations

import numpy as np
import pytest

from splita.core.objective_bayesian import ObjectiveBayesianExperiment


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


class TestFitPrior:
    def test_learns_prior_mean(self):
        effects = [0.01, -0.02, 0.03, 0.0, -0.01, 0.02, 0.01, -0.005]
        exp = ObjectiveBayesianExperiment()
        exp.fit_prior(effects)
        assert abs(exp.prior_mean_ - np.mean(effects)) < 1e-10

    def test_learns_prior_std(self):
        effects = [0.01, -0.02, 0.03, 0.0, -0.01, 0.02, 0.01, -0.005]
        exp = ObjectiveBayesianExperiment()
        exp.fit_prior(effects)
        assert abs(exp.prior_std_ - np.std(effects, ddof=1)) < 1e-10

    def test_returns_self(self):
        effects = [0.01, -0.02, 0.03]
        exp = ObjectiveBayesianExperiment()
        result = exp.fit_prior(effects)
        assert result is exp


class TestRun:
    def test_basic_run(self, rng):
        exp = ObjectiveBayesianExperiment()
        exp.fit_prior([0.01, -0.02, 0.03, 0.0, -0.01])
        ctrl = rng.normal(10, 2, 500)
        trt = rng.normal(10.3, 2, 500)
        result = exp.run(ctrl, trt)
        assert result.posterior_mean > 0

    def test_posterior_shrinks_toward_prior(self, rng):
        # Small data + informative prior => posterior closer to prior
        exp = ObjectiveBayesianExperiment()
        exp.fit_prior([0.0, 0.0, 0.0, 0.0, 0.0, 0.001, -0.001])
        ctrl = rng.normal(10, 2, 20)
        trt = rng.normal(11, 2, 20)  # large effect
        result = exp.run(ctrl, trt)
        observed = np.mean(trt) - np.mean(ctrl)
        # Posterior should be between prior mean and observed
        assert abs(result.posterior_mean) < abs(observed)
        assert result.shrinkage > 0

    def test_vague_prior_converges_to_frequentist(self, rng):
        exp = ObjectiveBayesianExperiment()
        # Don't fit prior — uses default vague prior
        ctrl = rng.normal(10, 2, 500)
        trt = rng.normal(10.3, 2, 500)
        result = exp.run(ctrl, trt)
        observed = np.mean(trt) - np.mean(ctrl)
        # With vague prior, posterior ≈ data
        assert abs(result.posterior_mean - observed) < 0.01

    def test_credible_interval(self, rng):
        exp = ObjectiveBayesianExperiment()
        exp.fit_prior([0.01, -0.02, 0.03, 0.0])
        ctrl = rng.normal(10, 2, 500)
        trt = rng.normal(10.3, 2, 500)
        result = exp.run(ctrl, trt)
        assert result.ci_lower < result.posterior_mean < result.ci_upper

    def test_prob_positive(self, rng):
        exp = ObjectiveBayesianExperiment()
        exp.fit_prior([0.01, 0.02, 0.03, 0.01])
        ctrl = rng.normal(10, 2, 500)
        trt = rng.normal(10.5, 2, 500)
        result = exp.run(ctrl, trt)
        assert result.prob_positive > 0.5

    def test_to_dict(self, rng):
        exp = ObjectiveBayesianExperiment()
        exp.fit_prior([0.01, -0.02, 0.03])
        result = exp.run(rng.normal(10, 2, 200), rng.normal(10.3, 2, 200))
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "posterior_mean" in d

    def test_repr(self, rng):
        exp = ObjectiveBayesianExperiment()
        exp.fit_prior([0.01, -0.02, 0.03])
        result = exp.run(rng.normal(10, 2, 200), rng.normal(10.3, 2, 200))
        assert "ObjectiveBayesianResult" in repr(result)


class TestValidation:
    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            ObjectiveBayesianExperiment(alpha=1.5)

    def test_too_few_historical(self):
        with pytest.raises(ValueError, match="at least 2"):
            ObjectiveBayesianExperiment().fit_prior([0.01])

    def test_identical_historical(self):
        with pytest.raises(ValueError, match="identical"):
            ObjectiveBayesianExperiment().fit_prior([0.01, 0.01, 0.01])

    def test_short_control(self, rng):
        exp = ObjectiveBayesianExperiment()
        with pytest.raises(ValueError, match="at least"):
            exp.run([1.0], rng.normal(0, 1, 50))
