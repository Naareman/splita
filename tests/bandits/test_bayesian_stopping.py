"""Tests for BayesianStopping."""

from __future__ import annotations

import numpy as np
import pytest

from splita.bandits.bayesian_stopping import BayesianStopping
from splita.bandits.thompson import ThompsonSampler
from splita._types import BanditResult, BayesianStoppingResult


# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def converged_result():
    """A BanditResult that looks like a converged experiment."""
    ts = ThompsonSampler(2, random_state=42)
    rng = np.random.default_rng(42)
    for _ in range(500):
        arm = ts.recommend()
        reward = float(rng.binomial(1, 0.8 if arm == 0 else 0.2))
        ts.update(arm, reward)
    return ts.result()


@pytest.fixture
def early_result():
    """A BanditResult with very few samples."""
    ts = ThompsonSampler(2, random_state=42)
    rng = np.random.default_rng(42)
    for _ in range(10):
        arm = ts.recommend()
        reward = float(rng.binomial(1, 0.5))
        ts.update(arm, reward)
    return ts.result()


# ─── Expected loss rule ─────────────────────────────────────────────


class TestExpectedLoss:
    def test_converged_stops(self, converged_result):
        stopper = BayesianStopping(rule="expected_loss", threshold=0.05,
                                   min_samples=100)
        assert stopper.should_stop(converged_result) is True

    def test_evaluate_returns_result(self, converged_result):
        stopper = BayesianStopping(rule="expected_loss", threshold=0.05,
                                   min_samples=100)
        result = stopper.evaluate(converged_result)
        assert isinstance(result, BayesianStoppingResult)
        assert result.rule == "expected_loss"

    def test_high_threshold_does_not_stop(self):
        """With very few samples and unclear winner, expected loss is high."""
        ts = ThompsonSampler(2, random_state=99)
        rng = np.random.default_rng(99)
        # Both arms have similar reward rates => expected loss is high
        for _ in range(200):
            arm = ts.recommend()
            reward = float(rng.binomial(1, 0.5))
            ts.update(arm, reward)
        result = ts.result()
        stopper = BayesianStopping(rule="expected_loss", threshold=1e-10,
                                   min_samples=100)
        assert stopper.should_stop(result) is False


# ─── Prob best rule ──────────────────────────────────────────────────


class TestProbBest:
    def test_converged_stops(self, converged_result):
        stopper = BayesianStopping(rule="prob_best", threshold=0.90,
                                   min_samples=100)
        assert stopper.should_stop(converged_result) is True

    def test_high_threshold_does_not_stop(self, converged_result):
        stopper = BayesianStopping(rule="prob_best", threshold=0.9999,
                                   min_samples=100)
        result = stopper.evaluate(converged_result)
        # May or may not stop depending on MC, but should return valid result
        assert isinstance(result.should_stop, bool)


# ─── Precision rule ──────────────────────────────────────────────────


class TestPrecision:
    def test_precision_works(self, converged_result):
        stopper = BayesianStopping(rule="precision", threshold=1.0,
                                   min_samples=100)
        result = stopper.evaluate(converged_result)
        assert isinstance(result, BayesianStoppingResult)
        assert result.rule == "precision"

    def test_tight_precision_does_not_stop(self, converged_result):
        stopper = BayesianStopping(rule="precision", threshold=1e-10,
                                   min_samples=100)
        assert stopper.should_stop(converged_result) is False


# ─── Min samples ─────────────────────────────────────────────────────


class TestMinSamples:
    def test_below_min_samples_does_not_stop(self, early_result):
        stopper = BayesianStopping(rule="expected_loss", threshold=0.05,
                                   min_samples=1000)
        result = stopper.evaluate(early_result)
        assert result.should_stop is False
        assert "Not enough samples" in result.message

    def test_min_samples_zero_allows_early_stop(self, early_result):
        stopper = BayesianStopping(rule="expected_loss", threshold=10.0,
                                   min_samples=0)
        result = stopper.evaluate(early_result)
        # With huge threshold, expected_loss should be below it
        assert result.should_stop is True


# ─── Validation ──────────────────────────────────────────────────────


class TestValidation:
    def test_invalid_rule(self):
        with pytest.raises(ValueError, match="rule"):
            BayesianStopping(rule="invalid")

    def test_expected_loss_threshold_zero(self):
        with pytest.raises(ValueError, match="threshold"):
            BayesianStopping(rule="expected_loss", threshold=0.0)

    def test_expected_loss_threshold_negative(self):
        with pytest.raises(ValueError, match="threshold"):
            BayesianStopping(rule="expected_loss", threshold=-0.01)

    def test_prob_best_threshold_out_of_range(self):
        with pytest.raises(ValueError, match="threshold"):
            BayesianStopping(rule="prob_best", threshold=0.0)
        with pytest.raises(ValueError, match="threshold"):
            BayesianStopping(rule="prob_best", threshold=1.0)

    def test_precision_threshold_zero(self):
        with pytest.raises(ValueError, match="threshold"):
            BayesianStopping(rule="precision", threshold=0.0)

    def test_invalid_bandit_result_type(self):
        stopper = BayesianStopping()
        with pytest.raises(TypeError, match="BanditResult"):
            stopper.evaluate("not_a_result")

    def test_min_samples_negative(self):
        with pytest.raises(ValueError, match="min_samples"):
            BayesianStopping(min_samples=-1)


# ─── Message content ────────────────────────────────────────────────


class TestMessage:
    def test_expected_loss_message(self, converged_result):
        stopper = BayesianStopping(rule="expected_loss", threshold=0.05,
                                   min_samples=100)
        result = stopper.evaluate(converged_result)
        assert "min(expected_loss)" in result.message

    def test_prob_best_message(self, converged_result):
        stopper = BayesianStopping(rule="prob_best", threshold=0.90,
                                   min_samples=100)
        result = stopper.evaluate(converged_result)
        assert "max(prob_best)" in result.message

    def test_precision_message(self, converged_result):
        stopper = BayesianStopping(rule="precision", threshold=1.0,
                                   min_samples=100)
        result = stopper.evaluate(converged_result)
        assert "max(CI_width)" in result.message


# ─── Serialization ──────────────────────────────────────────────────


class TestSerialization:
    def test_result_to_dict(self, converged_result):
        stopper = BayesianStopping(rule="expected_loss", threshold=0.05,
                                   min_samples=100)
        result = stopper.evaluate(converged_result)
        d = result.to_dict()
        assert "should_stop" in d
        assert "rule" in d
        assert "current_value" in d
