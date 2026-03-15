"""Tests for RiskAwareDecision."""

from __future__ import annotations

import numpy as np
import pytest

from splita.core.risk_aware import RiskAwareDecision


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


class TestDecisions:
    def test_ship_when_positive_and_constraints_met(self, rng):
        rd = RiskAwareDecision()
        rd.add_metric(
            "revenue",
            rng.normal(10, 1, 1000),
            rng.normal(11, 1, 1000),
            min_acceptable=-0.5,
        )
        result = rd.decide()
        assert result.decision == "ship"
        assert result.constraints_met

    def test_hold_when_constraint_violated(self, rng):
        rd = RiskAwareDecision()
        rd.add_metric(
            "revenue",
            rng.normal(10, 1, 1000),
            rng.normal(11, 1, 1000),
            min_acceptable=0.0,
        )
        rd.add_metric(
            "latency",
            rng.normal(100, 5, 1000),
            rng.normal(120, 5, 1000),  # big degradation
            max_acceptable=5.0,  # can't exceed 5ms
        )
        result = rd.decide()
        assert result.decision == "hold"
        assert not result.constraints_met
        assert "latency" in result.violations

    def test_investigate_when_no_significant_positive(self, rng):
        rd = RiskAwareDecision()
        rd.add_metric(
            "revenue",
            rng.normal(10, 1, 100),
            rng.normal(10.01, 1, 100),  # tiny effect, not significant
        )
        result = rd.decide()
        assert result.decision == "investigate"

    def test_multiple_metrics(self, rng):
        rd = RiskAwareDecision()
        rd.add_metric("m1", rng.normal(10, 1, 500), rng.normal(10.5, 1, 500))
        rd.add_metric("m2", rng.normal(5, 0.5, 500), rng.normal(5.1, 0.5, 500))
        result = rd.decide()
        assert len(result.metric_details) == 2

    def test_metric_details_have_lift(self, rng):
        rd = RiskAwareDecision()
        rd.add_metric("m1", rng.normal(10, 1, 500), rng.normal(10.5, 1, 500))
        result = rd.decide()
        assert "lift" in result.metric_details["m1"]
        assert "pvalue" in result.metric_details["m1"]
        assert "ci_lower" in result.metric_details["m1"]


class TestMethodChaining:
    def test_add_metric_returns_self(self, rng):
        rd = RiskAwareDecision()
        result = rd.add_metric("m1", rng.normal(0, 1, 50), rng.normal(0, 1, 50))
        assert result is rd


class TestResultMethods:
    def test_to_dict(self, rng):
        rd = RiskAwareDecision()
        rd.add_metric("m1", rng.normal(10, 1, 500), rng.normal(10.5, 1, 500))
        result = rd.decide()
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "decision" in d

    def test_repr(self, rng):
        rd = RiskAwareDecision()
        rd.add_metric("m1", rng.normal(10, 1, 500), rng.normal(10.5, 1, 500))
        result = rd.decide()
        assert "RiskAwareResult" in repr(result)


class TestValidation:
    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            RiskAwareDecision(alpha=0.0)

    def test_empty_name(self, rng):
        rd = RiskAwareDecision()
        with pytest.raises(ValueError, match="empty"):
            rd.add_metric("", rng.normal(0, 1, 50), rng.normal(0, 1, 50))

    def test_duplicate_metric(self, rng):
        rd = RiskAwareDecision()
        rd.add_metric("m1", rng.normal(0, 1, 50), rng.normal(0, 1, 50))
        with pytest.raises(ValueError, match="already been added"):
            rd.add_metric("m1", rng.normal(0, 1, 50), rng.normal(0, 1, 50))

    def test_no_metrics(self):
        rd = RiskAwareDecision()
        with pytest.raises(RuntimeError, match="without any metrics"):
            rd.decide()

    def test_short_arrays(self, rng):
        rd = RiskAwareDecision()
        with pytest.raises(ValueError, match="at least"):
            rd.add_metric("m1", [1.0], [2.0])
