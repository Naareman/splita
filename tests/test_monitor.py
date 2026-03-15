"""Tests for splita.monitor."""

from __future__ import annotations

import numpy as np
import pytest

from splita.monitor import monitor


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


class TestMonitorBasic:
    """Basic monitor functionality."""

    def test_returns_monitor_result(self, rng: np.random.Generator) -> None:
        ctrl = rng.binomial(1, 0.10, 500).astype(float)
        trt = rng.binomial(1, 0.12, 500).astype(float)
        result = monitor(ctrl, trt)
        assert hasattr(result, "current_lift")
        assert hasattr(result, "current_pvalue")
        assert hasattr(result, "recommendation")

    def test_recommendation_is_valid(self, rng: np.random.Generator) -> None:
        ctrl = rng.binomial(1, 0.10, 500).astype(float)
        trt = rng.binomial(1, 0.12, 500).astype(float)
        result = monitor(ctrl, trt)
        assert result.recommendation in (
            "continue", "stop_winner", "stop_harm", "investigate"
        )

    def test_current_n_equals_total(self, rng: np.random.Generator) -> None:
        ctrl = rng.binomial(1, 0.10, 300).astype(float)
        trt = rng.binomial(1, 0.12, 400).astype(float)
        result = monitor(ctrl, trt)
        assert result.current_n == 700

    def test_pvalue_between_0_and_1(self, rng: np.random.Generator) -> None:
        ctrl = rng.binomial(1, 0.10, 500).astype(float)
        trt = rng.binomial(1, 0.12, 500).astype(float)
        result = monitor(ctrl, trt)
        assert 0.0 <= result.current_pvalue <= 1.0

    def test_srm_passes_equal_split(self, rng: np.random.Generator) -> None:
        ctrl = rng.binomial(1, 0.10, 500).astype(float)
        trt = rng.binomial(1, 0.12, 500).astype(float)
        result = monitor(ctrl, trt)
        assert result.srm_passed is True


class TestMonitorDaysRemaining:
    """Days remaining calculation."""

    def test_days_remaining_with_target_and_daily(
        self, rng: np.random.Generator
    ) -> None:
        ctrl = rng.binomial(1, 0.10, 200).astype(float)
        trt = rng.binomial(1, 0.12, 200).astype(float)
        result = monitor(ctrl, trt, target_n=1000, daily_users=100)
        assert result.days_remaining is not None
        assert result.days_remaining == 6  # (1000 - 400) / 100 = 6

    def test_days_remaining_none_without_target(
        self, rng: np.random.Generator
    ) -> None:
        ctrl = rng.binomial(1, 0.10, 200).astype(float)
        trt = rng.binomial(1, 0.12, 200).astype(float)
        result = monitor(ctrl, trt)
        assert result.days_remaining is None

    def test_days_remaining_zero_when_target_reached(
        self, rng: np.random.Generator
    ) -> None:
        ctrl = rng.binomial(1, 0.10, 500).astype(float)
        trt = rng.binomial(1, 0.12, 500).astype(float)
        result = monitor(ctrl, trt, target_n=1000, daily_users=100)
        assert result.days_remaining == 0


class TestMonitorGuardrails:
    """Guardrail checks."""

    def test_empty_guardrails_by_default(
        self, rng: np.random.Generator
    ) -> None:
        ctrl = rng.binomial(1, 0.10, 500).astype(float)
        trt = rng.binomial(1, 0.12, 500).astype(float)
        result = monitor(ctrl, trt)
        assert result.guardrail_status == []

    def test_guardrail_results_returned(
        self, rng: np.random.Generator
    ) -> None:
        ctrl = rng.binomial(1, 0.10, 500).astype(float)
        trt = rng.binomial(1, 0.12, 500).astype(float)
        guardrails = [{"name": "latency", "threshold": -0.05}]
        result = monitor(ctrl, trt, guardrails=guardrails)
        assert len(result.guardrail_status) == 1
        assert result.guardrail_status[0]["name"] == "latency"
        assert "passed" in result.guardrail_status[0]

    def test_guardrail_failure_triggers_stop_harm(self) -> None:
        # Treatment is much worse: ~80% vs ~20% conversion
        rng = np.random.default_rng(99)
        ctrl = rng.binomial(1, 0.80, 500).astype(float)
        trt = rng.binomial(1, 0.20, 500).astype(float)
        guardrails = [{"name": "revenue", "threshold": 0.0}]
        result = monitor(ctrl, trt, guardrails=guardrails)
        assert result.recommendation == "stop_harm"


class TestMonitorSRM:
    """SRM detection."""

    def test_srm_failure_with_skewed_split(self) -> None:
        ctrl = np.ones(100)
        trt = np.ones(900)
        result = monitor(ctrl, trt)
        assert result.srm_passed is False
        assert result.recommendation == "investigate"


class TestMonitorPrediction:
    """Predicted significance."""

    def test_predicted_significant_with_large_target(
        self, rng: np.random.Generator
    ) -> None:
        ctrl = rng.binomial(1, 0.10, 500).astype(float)
        trt = rng.binomial(1, 0.15, 500).astype(float)
        result = monitor(ctrl, trt, target_n=100000)
        assert result.predicted_significant is True

    def test_no_prediction_without_target(
        self, rng: np.random.Generator
    ) -> None:
        ctrl = rng.binomial(1, 0.10, 100).astype(float)
        trt = rng.binomial(1, 0.10, 100).astype(float)
        result = monitor(ctrl, trt)
        assert result.predicted_significant is False


class TestMonitorToDict:
    """Serialization."""

    def test_to_dict(self, rng: np.random.Generator) -> None:
        ctrl = rng.binomial(1, 0.10, 500).astype(float)
        trt = rng.binomial(1, 0.12, 500).astype(float)
        result = monitor(ctrl, trt)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "current_lift" in d
        assert "recommendation" in d

    def test_to_json(self, rng: np.random.Generator) -> None:
        ctrl = rng.binomial(1, 0.10, 500).astype(float)
        trt = rng.binomial(1, 0.12, 500).astype(float)
        result = monitor(ctrl, trt)
        j = result.to_json()
        assert isinstance(j, str)
        assert "current_lift" in j
