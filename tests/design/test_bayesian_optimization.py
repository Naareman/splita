"""Tests for BayesianExperimentOptimizer (Meta 2025)."""

from __future__ import annotations

import numpy as np
import pytest

from splita._types import BayesOptResult
from splita.design.bayesian_optimization import BayesianExperimentOptimizer


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def optimizer():
    return BayesianExperimentOptimizer(
        param_bounds={"price": (5.0, 50.0), "discount": (0.0, 0.5)}
    )


class TestBayesianOptBasic:
    """Basic functionality tests."""

    def test_result_type(self, optimizer):
        """result() should return BayesOptResult."""
        optimizer.add_experiment({"price": 10.0, "discount": 0.1}, 5.0, 8.0)
        optimizer.add_experiment({"price": 20.0, "discount": 0.2}, 7.0, 12.0)
        optimizer.add_experiment({"price": 30.0, "discount": 0.3}, 4.0, 6.0)
        result = optimizer.result()
        assert isinstance(result, BayesOptResult)

    def test_n_experiments(self, optimizer):
        """n_experiments should match number of added experiments."""
        for i in range(5):
            optimizer.add_experiment(
                {"price": 10.0 + i * 5, "discount": 0.1 + i * 0.05},
                float(i),
                float(i * 2),
            )
        result = optimizer.result()
        assert result.n_experiments == 5

    def test_best_params_within_bounds(self, optimizer):
        """Best params should be within the specified bounds."""
        optimizer.add_experiment({"price": 10.0, "discount": 0.1}, 5.0, 8.0)
        optimizer.add_experiment({"price": 20.0, "discount": 0.2}, 7.0, 12.0)
        optimizer.add_experiment({"price": 30.0, "discount": 0.3}, 4.0, 6.0)
        result = optimizer.result()
        assert 5.0 <= result.best_params["price"] <= 50.0
        assert 0.0 <= result.best_params["discount"] <= 0.5

    def test_suggest_next_within_bounds(self, optimizer):
        """suggest_next() should return params within bounds."""
        optimizer.add_experiment({"price": 10.0, "discount": 0.1}, 5.0, 8.0)
        optimizer.add_experiment({"price": 20.0, "discount": 0.2}, 7.0, 12.0)
        suggestion = optimizer.suggest_next()
        assert isinstance(suggestion, dict)
        assert 5.0 <= suggestion["price"] <= 50.0
        assert 0.0 <= suggestion["discount"] <= 0.5

    def test_surrogate_r2_reasonable(self, optimizer):
        """R2 should be between 0 and 1 with well-behaved data."""
        # Perfect linear relationship
        for x in [10, 20, 30, 40]:
            optimizer.add_experiment(
                {"price": float(x), "discount": 0.1},
                float(x),
                float(x * 2),
            )
        result = optimizer.result()
        assert 0.0 <= result.surrogate_r2 <= 1.0

    def test_to_dict(self, optimizer):
        """to_dict() should return a plain dict."""
        optimizer.add_experiment({"price": 10.0, "discount": 0.1}, 5.0, 8.0)
        optimizer.add_experiment({"price": 20.0, "discount": 0.2}, 7.0, 12.0)
        d = optimizer.result().to_dict()
        assert isinstance(d, dict)
        assert "best_params" in d
        assert "surrogate_r2" in d

    def test_finds_best_outcome(self):
        """Should identify the params that produced the best long-term outcome."""
        opt = BayesianExperimentOptimizer(
            param_bounds={"x": (0.0, 10.0)}
        )
        opt.add_experiment({"x": 2.0}, 1.0, 3.0)
        opt.add_experiment({"x": 5.0}, 2.0, 10.0)  # best
        opt.add_experiment({"x": 8.0}, 1.5, 5.0)
        result = opt.result()
        assert result.predicted_long_term >= 5.0


class TestBayesianOptValidation:
    """Validation and error handling tests."""

    def test_empty_param_bounds(self):
        """Empty param_bounds should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty dict"):
            BayesianExperimentOptimizer(param_bounds={})

    def test_invalid_bounds(self):
        """Lower >= upper should raise ValueError."""
        with pytest.raises(ValueError, match="lower bound must be < upper"):
            BayesianExperimentOptimizer(param_bounds={"x": (10.0, 5.0)})

    def test_missing_param_key(self, optimizer):
        """Missing key in treatment_params should raise ValueError."""
        with pytest.raises(ValueError, match="must contain key"):
            optimizer.add_experiment({"price": 10.0}, 5.0, 8.0)

    def test_not_enough_experiments(self, optimizer):
        """result() with < 2 long-term experiments should raise."""
        optimizer.add_experiment({"price": 10.0, "discount": 0.1}, 5.0, 8.0)
        with pytest.raises(ValueError, match="at least 2 experiments"):
            optimizer.result()

    def test_negative_exploration_weight(self):
        """Negative exploration_weight should raise ValueError."""
        with pytest.raises(ValueError, match="exploration_weight"):
            BayesianExperimentOptimizer(
                param_bounds={"x": (0.0, 1.0)}, exploration_weight=-1.0
            )

    def test_repr(self, optimizer):
        """__repr__ should produce a readable string."""
        optimizer.add_experiment({"price": 10.0, "discount": 0.1}, 5.0, 8.0)
        optimizer.add_experiment({"price": 20.0, "discount": 0.2}, 7.0, 12.0)
        s = repr(optimizer.result())
        assert "BayesOptResult" in s
