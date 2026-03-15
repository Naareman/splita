"""Tests for GuardrailMonitor."""

import numpy as np
import pytest

from splita._types import GuardrailResult
from splita.governance.guardrail import GuardrailMonitor


# ─── Happy path ────────────────────────────────────────────────────


class TestAllPass:
    def test_single_guardrail_passes(self):
        rng = np.random.default_rng(42)
        ctrl = rng.normal(100, 10, 500)
        trt = rng.normal(100, 10, 500)
        mon = GuardrailMonitor(alpha=0.05)
        mon.add_guardrail("latency", ctrl, trt, direction="increase")
        result = mon.check()
        assert result.all_passed is True
        assert result.recommendation == "safe"
        assert len(result.guardrail_results) == 1
        assert result.guardrail_results[0]["passed"] is True

    def test_multiple_guardrails_all_pass(self):
        rng = np.random.default_rng(42)
        mon = GuardrailMonitor(alpha=0.05)
        for name in ["latency", "error_rate", "crash_rate"]:
            ctrl = rng.normal(50, 5, 300)
            trt = rng.normal(50, 5, 300)
            mon.add_guardrail(name, ctrl, trt)
        result = mon.check()
        assert result.all_passed is True
        assert result.recommendation == "safe"
        assert len(result.guardrail_results) == 3

    def test_result_is_frozen_dataclass(self):
        rng = np.random.default_rng(42)
        mon = GuardrailMonitor()
        mon.add_guardrail("m", rng.normal(0, 1, 100), rng.normal(0, 1, 100))
        result = mon.check()
        assert isinstance(result, GuardrailResult)
        with pytest.raises(AttributeError):
            result.all_passed = False  # type: ignore[misc]

    def test_to_dict(self):
        rng = np.random.default_rng(42)
        mon = GuardrailMonitor()
        mon.add_guardrail("m", rng.normal(0, 1, 100), rng.normal(0, 1, 100))
        d = mon.check().to_dict()
        assert isinstance(d, dict)
        assert "all_passed" in d
        assert "guardrail_results" in d


# ─── Failure / breach cases ────────────────────────────────────────


class TestBreach:
    def test_one_guardrail_fails(self):
        rng = np.random.default_rng(42)
        mon = GuardrailMonitor(alpha=0.05)
        # Safe guardrail
        mon.add_guardrail("safe_metric",
                          rng.normal(100, 10, 500),
                          rng.normal(100, 10, 500),
                          direction="increase")
        # Breached guardrail: treatment significantly higher
        mon.add_guardrail("bad_metric",
                          rng.normal(100, 10, 500),
                          rng.normal(115, 10, 500),
                          direction="increase")
        result = mon.check()
        assert result.all_passed is False
        assert result.recommendation == "warning"
        bad = [r for r in result.guardrail_results if r["name"] == "bad_metric"]
        assert len(bad) == 1
        assert bad[0]["passed"] is False

    def test_all_guardrails_fail_recommends_stop(self):
        rng = np.random.default_rng(42)
        mon = GuardrailMonitor(alpha=0.05)
        mon.add_guardrail("m1",
                          rng.normal(100, 5, 500),
                          rng.normal(120, 5, 500),
                          direction="increase")
        mon.add_guardrail("m2",
                          rng.normal(100, 5, 500),
                          rng.normal(120, 5, 500),
                          direction="increase")
        result = mon.check()
        assert result.all_passed is False
        assert result.recommendation == "stop"


# ─── Direction checking ────────────────────────────────────────────


class TestDirection:
    def test_increase_direction_no_breach_when_treatment_lower(self):
        rng = np.random.default_rng(42)
        # Treatment is significantly LOWER, but direction="increase" only flags increases
        mon = GuardrailMonitor(alpha=0.05)
        mon.add_guardrail("latency",
                          rng.normal(100, 5, 500),
                          rng.normal(85, 5, 500),
                          direction="increase")
        result = mon.check()
        assert result.all_passed is True

    def test_decrease_direction_catches_significant_drop(self):
        rng = np.random.default_rng(42)
        mon = GuardrailMonitor(alpha=0.05)
        mon.add_guardrail("revenue",
                          rng.normal(100, 5, 500),
                          rng.normal(85, 5, 500),
                          direction="decrease")
        result = mon.check()
        assert result.all_passed is False

    def test_any_direction_catches_both(self):
        rng = np.random.default_rng(42)
        mon = GuardrailMonitor(alpha=0.05)
        mon.add_guardrail("metric",
                          rng.normal(100, 5, 500),
                          rng.normal(120, 5, 500),
                          direction="any")
        result = mon.check()
        assert result.all_passed is False


# ─── Threshold ─────────────────────────────────────────────────────


class TestThreshold:
    def test_threshold_breach(self):
        rng = np.random.default_rng(42)
        mon = GuardrailMonitor(alpha=0.05)
        # Even with same distribution, large threshold breach
        ctrl = rng.normal(100, 1, 500)
        trt = rng.normal(105, 1, 500)
        mon.add_guardrail("metric", ctrl, trt, threshold=2.0)
        result = mon.check()
        assert result.all_passed is False

    def test_threshold_not_breached(self):
        rng = np.random.default_rng(42)
        mon = GuardrailMonitor(alpha=0.05)
        ctrl = rng.normal(100, 10, 100)
        trt = rng.normal(100, 10, 100)
        # No real effect, high threshold won't be hit
        mon.add_guardrail("metric", ctrl, trt, direction="increase", threshold=50.0)
        result = mon.check()
        assert result.guardrail_results[0]["passed"] is True


# ─── Validation ────────────────────────────────────────────────────


class TestValidation:
    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            GuardrailMonitor(alpha=1.5)

    def test_empty_name(self):
        mon = GuardrailMonitor()
        with pytest.raises(ValueError, match="name"):
            mon.add_guardrail("", [1, 2, 3], [1, 2, 3])

    def test_invalid_direction(self):
        mon = GuardrailMonitor()
        with pytest.raises(ValueError, match="direction"):
            mon.add_guardrail("m", [1, 2, 3], [1, 2, 3], direction="sideways")

    def test_negative_threshold(self):
        mon = GuardrailMonitor()
        with pytest.raises(ValueError, match="threshold"):
            mon.add_guardrail("m", [1, 2, 3], [1, 2, 3], threshold=-1.0)

    def test_check_without_guardrails_raises(self):
        mon = GuardrailMonitor()
        with pytest.raises(ValueError, match="No guardrails"):
            mon.check()

    def test_invalid_control_type(self):
        mon = GuardrailMonitor()
        with pytest.raises(TypeError, match="control"):
            mon.add_guardrail("m", "not_an_array", [1, 2, 3])


# ─── Bonferroni correction ────────────────────────────────────────


class TestBonferroni:
    def test_bonferroni_makes_test_stricter(self):
        """With many guardrails, Bonferroni should reduce false positives."""
        rng = np.random.default_rng(42)
        # Under H0 (no effect), adding more guardrails should not inflate FPR
        n_sims = 50
        false_positives = 0
        for i in range(n_sims):
            seed_rng = np.random.default_rng(i + 100)
            mon = GuardrailMonitor(alpha=0.05)
            for j in range(10):
                ctrl = seed_rng.normal(100, 10, 100)
                trt = seed_rng.normal(100, 10, 100)
                mon.add_guardrail(f"g{j}", ctrl, trt)
            result = mon.check()
            if not result.all_passed:
                false_positives += 1
        # With Bonferroni across 10 guardrails, FPR should be well below 20%
        assert false_positives / n_sims < 0.20


# ─── Repr / message ───────────────────────────────────────────────


class TestRepr:
    def test_repr_contains_key_info(self):
        rng = np.random.default_rng(42)
        mon = GuardrailMonitor()
        mon.add_guardrail("m", rng.normal(0, 1, 100), rng.normal(0, 1, 100))
        result = mon.check()
        text = repr(result)
        assert "GuardrailResult" in text
        assert "all_passed" in text
