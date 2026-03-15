"""Tests for DynamicCausalEffect (Shi, Deng et al. JASA 2022)."""

from __future__ import annotations

import numpy as np
import pytest

from splita._types import DynamicResult
from splita.causal.dynamic_effects import DynamicCausalEffect


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestDynamicCausalEffectBasic:
    """Basic functionality tests."""

    def test_returns_dynamic_result(self, rng):
        """fit() should return a DynamicResult."""
        T = 4
        outcomes = [rng.normal(10, 1, 100) for _ in range(T)]
        treatments = [rng.binomial(1, 0.5, 100).astype(float) for _ in range(T)]
        result = DynamicCausalEffect().fit(outcomes, treatments, list(range(T)))
        assert isinstance(result, DynamicResult)

    def test_detects_significant_effect(self, rng):
        """Large effect should yield small p-value."""
        T = 5
        outcomes = []
        treatments = []
        for _ in range(T):
            t = rng.binomial(1, 0.5, 200).astype(float)
            y = rng.normal(10, 1, 200) + t * 3.0
            outcomes.append(y)
            treatments.append(t)
        result = DynamicCausalEffect().fit(outcomes, treatments, list(range(T)))
        assert result.pvalue < 0.05
        assert result.cumulative_effect > 0

    def test_no_effect_high_pvalue(self, rng):
        """No treatment effect should yield high p-value."""
        T = 4
        outcomes = [rng.normal(10, 1, 100) for _ in range(T)]
        treatments = [rng.binomial(1, 0.5, 100).astype(float) for _ in range(T)]
        result = DynamicCausalEffect().fit(outcomes, treatments, list(range(T)))
        assert result.pvalue > 0.01

    def test_effects_over_time_structure(self, rng):
        """Each period should have effect, se, pvalue keys."""
        T = 3
        outcomes = [rng.normal(10, 1, 100) for _ in range(T)]
        treatments = [rng.binomial(1, 0.5, 100).astype(float) for _ in range(T)]
        result = DynamicCausalEffect().fit(outcomes, treatments, list(range(T)))
        assert len(result.effects_over_time) == T
        for rec in result.effects_over_time:
            assert "period" in rec
            assert "effect" in rec
            assert "se" in rec
            assert "pvalue" in rec

    def test_increasing_trend(self, rng):
        """Increasing effect over time should detect 'increasing' trend."""
        T = 8
        outcomes = []
        treatments = []
        for t_idx in range(T):
            t = rng.binomial(1, 0.5, 200).astype(float)
            # Effect grows over time
            y = rng.normal(10, 0.5, 200) + t * (1.0 + t_idx * 1.0)
            outcomes.append(y)
            treatments.append(t)
        result = DynamicCausalEffect().fit(outcomes, treatments, list(range(T)))
        assert result.trend in ("increasing", "stable")

    def test_to_dict(self, rng):
        """to_dict() should return a plain dict."""
        T = 3
        outcomes = [rng.normal(10, 1, 50) for _ in range(T)]
        treatments = [rng.binomial(1, 0.5, 50).astype(float) for _ in range(T)]
        result = DynamicCausalEffect().fit(outcomes, treatments, list(range(T)))
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "cumulative_effect" in d
        assert "trend" in d


class TestDynamicCausalEffectValidation:
    """Validation and error handling tests."""

    def test_too_few_periods(self, rng):
        """Fewer than 2 periods should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2 time periods"):
            DynamicCausalEffect().fit(
                [rng.normal(0, 1, 10)],
                [rng.binomial(1, 0.5, 10).astype(float)],
                [0],
            )

    def test_mismatched_lengths(self, rng):
        """Mismatched outcomes/timestamps lengths should raise."""
        with pytest.raises(ValueError, match="same length"):
            DynamicCausalEffect().fit(
                [rng.normal(0, 1, 10), rng.normal(0, 1, 10)],
                [rng.binomial(1, 0.5, 10).astype(float)] * 2,
                [0, 1, 2],
            )

    def test_invalid_alpha(self):
        """alpha outside (0, 1) should raise ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            DynamicCausalEffect(alpha=0.0)

    def test_outcomes_not_list(self, rng):
        """Non-list outcomes should raise TypeError."""
        with pytest.raises(TypeError, match="list of arrays"):
            DynamicCausalEffect().fit(
                rng.normal(0, 1, 10),  # type: ignore[arg-type]
                [rng.binomial(1, 0.5, 10).astype(float)],
                [0],
            )

    def test_treatments_not_binary(self, rng):
        """Non-binary treatments should raise ValueError."""
        with pytest.raises(ValueError, match="only 0 and 1"):
            DynamicCausalEffect().fit(
                [rng.normal(0, 1, 10), rng.normal(0, 1, 10)],
                [np.array([0, 1, 2, 0, 1, 0, 1, 0, 1, 0], dtype=float)] * 2,
                [0, 1],
            )

    def test_repr(self, rng):
        """__repr__ should produce a readable string."""
        T = 3
        outcomes = [rng.normal(10, 1, 50) for _ in range(T)]
        treatments = [rng.binomial(1, 0.5, 50).astype(float) for _ in range(T)]
        result = DynamicCausalEffect().fit(outcomes, treatments, list(range(T)))
        s = repr(result)
        assert "DynamicResult" in s
        assert "cumulative_effect" in s
