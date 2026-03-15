"""Tests for DoublyRobustEstimator."""

from __future__ import annotations

import numpy as np
import pytest

from splita.causal.doubly_robust import DoublyRobustEstimator


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


@pytest.fixture()
def simple_data(rng):
    n = 500
    x = rng.normal(0, 1, (n, 3))
    t = (x[:, 0] + rng.normal(0, 1, n) > 0).astype(float)
    y = 2.0 * t + x[:, 0] + rng.normal(0, 1, n)
    return y, t, x


class TestFit:
    def test_basic_fit(self, simple_data):
        y, t, x = simple_data
        result = DoublyRobustEstimator(random_state=42).fit(y, t, x)
        assert abs(result.ate - 2.0) < 1.5

    def test_returns_result_type(self, simple_data):
        y, t, x = simple_data
        result = DoublyRobustEstimator(random_state=42).fit(y, t, x)
        assert hasattr(result, "ate")
        assert hasattr(result, "se")
        assert hasattr(result, "pvalue")

    def test_confidence_interval(self, simple_data):
        y, t, x = simple_data
        result = DoublyRobustEstimator(random_state=42).fit(y, t, x)
        assert result.ci_lower < result.ate < result.ci_upper

    def test_pvalue_range(self, simple_data):
        y, t, x = simple_data
        result = DoublyRobustEstimator(random_state=42).fit(y, t, x)
        assert 0.0 <= result.pvalue <= 1.0

    def test_outcome_r2(self, simple_data):
        y, t, x = simple_data
        result = DoublyRobustEstimator(random_state=42).fit(y, t, x)
        assert result.outcome_r2 > 0.0  # model should explain some variance

    def test_propensity_auc(self, simple_data):
        y, t, x = simple_data
        result = DoublyRobustEstimator(random_state=42).fit(y, t, x)
        assert 0.0 <= result.propensity_auc <= 1.0

    def test_zero_effect(self, rng):
        n = 500
        x = rng.normal(0, 1, (n, 2))
        t = rng.binomial(1, 0.5, n).astype(float)
        y = x[:, 0] + rng.normal(0, 1, n)  # no treatment effect
        result = DoublyRobustEstimator(random_state=42).fit(y, t, x)
        assert abs(result.ate) < 1.0  # should be close to 0


class TestResultMethods:
    def test_to_dict(self, simple_data):
        y, t, x = simple_data
        result = DoublyRobustEstimator(random_state=42).fit(y, t, x)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "ate" in d

    def test_repr(self, simple_data):
        y, t, x = simple_data
        result = DoublyRobustEstimator(random_state=42).fit(y, t, x)
        assert "DoublyRobustResult" in repr(result)


class TestValidation:
    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            DoublyRobustEstimator(alpha=0.0)

    def test_invalid_n_folds(self):
        with pytest.raises(ValueError, match="n_folds"):
            DoublyRobustEstimator(n_folds=1)

    def test_1d_covariates(self, rng):
        with pytest.raises(ValueError, match="2-D"):
            DoublyRobustEstimator().fit(
                rng.normal(0, 1, 50),
                rng.binomial(1, 0.5, 50).astype(float),
                rng.normal(0, 1, 50),
            )

    def test_non_binary_treatment(self, rng):
        with pytest.raises(ValueError, match="0s and 1s"):
            DoublyRobustEstimator().fit(
                rng.normal(0, 1, 50),
                rng.normal(0, 1, 50),  # not binary
                rng.normal(0, 1, (50, 2)),
            )

    def test_mismatched_lengths(self, rng):
        with pytest.raises(ValueError, match="same number"):
            DoublyRobustEstimator().fit(
                rng.normal(0, 1, 50),
                rng.binomial(1, 0.5, 40).astype(float),
                rng.normal(0, 1, (50, 2)),
            )
