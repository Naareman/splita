"""Tests for BudgetSplitDesign (Liu et al. KDD 2021)."""

from __future__ import annotations

import numpy as np
import pytest

from splita._types import BudgetSplitResult
from splita.design.budget_split import BudgetSplitDesign


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestBudgetSplitBasic:
    """Basic functionality tests."""

    def test_design_returns_result(self):
        """design() should return a BudgetSplitResult."""
        design = BudgetSplitDesign()
        budgets = np.array([100, 200, 150, 300, 250], dtype=float)
        result = design.design(budgets, split_fraction=0.5)
        assert isinstance(result, BudgetSplitResult)

    def test_budget_allocation(self):
        """Budgets should split according to split_fraction."""
        design = BudgetSplitDesign()
        budgets = np.array([100, 200, 300], dtype=float)
        result = design.design(budgets, split_fraction=0.6)
        total = 600.0
        assert abs(result.treatment_budget - total * 0.6) < 0.01
        assert abs(result.control_budget - total * 0.4) < 0.01

    def test_design_is_cannibalization_free(self):
        """Design should always be marked cannibalization-free."""
        design = BudgetSplitDesign()
        result = design.design(np.array([100, 200], dtype=float))
        assert result.cannibalization_free is True

    def test_design_no_data_pvalue_one(self):
        """Before analysis, pvalue should be 1.0 and ate should be 0."""
        design = BudgetSplitDesign()
        result = design.design(np.array([100, 200], dtype=float))
        assert result.pvalue == 1.0
        assert result.ate == 0.0
        assert result.significant is False

    def test_analyze_detects_effect(self, rng):
        """Large treatment effect should be detected."""
        design = BudgetSplitDesign()
        design.design(np.array([100, 200, 300], dtype=float), split_fraction=0.5)
        y_t = rng.normal(15, 1, 100)
        y_c = rng.normal(10, 1, 100)
        result = design.analyze(y_t, y_c)
        assert result.pvalue < 0.05
        assert result.ate > 0
        assert result.significant is True

    def test_analyze_no_effect(self, rng):
        """No treatment effect should yield high p-value."""
        design = BudgetSplitDesign()
        design.design(np.array([100, 200], dtype=float))
        y_t = rng.normal(10, 1, 100)
        y_c = rng.normal(10, 1, 100)
        result = design.analyze(y_t, y_c)
        assert result.pvalue > 0.01

    def test_to_dict(self, rng):
        """to_dict() should return a plain dict."""
        design = BudgetSplitDesign()
        design.design(np.array([100, 200], dtype=float))
        result = design.analyze(rng.normal(10, 1, 50), rng.normal(10, 1, 50))
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "cannibalization_free" in d


class TestBudgetSplitValidation:
    """Validation and error handling tests."""

    def test_invalid_alpha(self):
        """alpha outside (0, 1) should raise ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            BudgetSplitDesign(alpha=1.5)

    def test_negative_budgets(self):
        """Negative budgets should raise ValueError."""
        design = BudgetSplitDesign()
        with pytest.raises(ValueError, match="non-negative"):
            design.design(np.array([-100, 200], dtype=float))

    def test_split_fraction_zero(self):
        """split_fraction of 0 should raise ValueError."""
        design = BudgetSplitDesign()
        with pytest.raises(ValueError, match="split_fraction"):
            design.design(np.array([100, 200], dtype=float), split_fraction=0.0)

    def test_split_fraction_one(self):
        """split_fraction of 1 should raise ValueError."""
        design = BudgetSplitDesign()
        with pytest.raises(ValueError, match="split_fraction"):
            design.design(np.array([100, 200], dtype=float), split_fraction=1.0)

    def test_analyze_too_few_treatment(self, rng):
        """Too few treatment outcomes should raise ValueError."""
        design = BudgetSplitDesign()
        with pytest.raises(ValueError, match="at least"):
            design.analyze(np.array([1.0]), rng.normal(10, 1, 50))

    def test_repr(self):
        """__repr__ should produce a readable string."""
        design = BudgetSplitDesign()
        result = design.design(np.array([100, 200], dtype=float))
        s = repr(result)
        assert "BudgetSplitResult" in s
        assert "cannibalization_free" in s
