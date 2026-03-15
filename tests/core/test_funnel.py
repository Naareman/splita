"""Tests for FunnelExperiment."""

from __future__ import annotations

import numpy as np
import pytest

from splita._types import FunnelResult
from splita.core.funnel import FunnelExperiment


# ── Basic behaviour ──────────────────────────────────────────────────


class TestBasic:
    """Basic FunnelExperiment behaviour."""

    def test_basic_funnel(self):
        """Basic 4-step funnel returns correct structure."""
        exp = FunnelExperiment()
        exp.add_step(900, 1000, 920, 1000, name="landing")
        exp.add_step(400, 900, 450, 920, name="cart")
        exp.add_step(200, 400, 240, 450, name="checkout")
        exp.add_step(150, 200, 190, 240, name="purchase")
        result = exp.run()
        assert isinstance(result, FunnelResult)
        assert len(result.step_results) == 4

    def test_result_frozen(self):
        """Result is frozen."""
        exp = FunnelExperiment()
        exp.add_step(50, 100, 60, 100, name="s1")
        result = exp.run()
        with pytest.raises(AttributeError):
            result.overall_lift = 999.0  # type: ignore[misc]

    def test_to_dict(self):
        """Result serializes to dict."""
        exp = FunnelExperiment()
        exp.add_step(50, 100, 60, 100, name="s1")
        result = exp.run()
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "step_results" in d

    def test_repr(self):
        """Result has string representation."""
        exp = FunnelExperiment()
        exp.add_step(50, 100, 60, 100, name="s1")
        result = exp.run()
        s = repr(result)
        assert "FunnelResult" in s


# ── Step results ─────────────────────────────────────────────────────


class TestStepResults:
    """Per-step result correctness."""

    def test_step_result_keys(self):
        """Each step result has expected keys."""
        exp = FunnelExperiment()
        exp.add_step(50, 100, 60, 100, name="s1")
        exp.add_step(30, 50, 40, 60, name="s2")
        result = exp.run()
        for sr in result.step_results:
            assert "name" in sr
            assert "control_rate" in sr
            assert "treatment_rate" in sr
            assert "lift" in sr
            assert "pvalue" in sr
            assert "significant" in sr
            assert "conditional_lift" in sr
            assert "conditional_pvalue" in sr

    def test_conversion_rates_correct(self):
        """Conversion rates are computed correctly."""
        exp = FunnelExperiment()
        exp.add_step(50, 100, 75, 100, name="s1")
        result = exp.run()
        sr = result.step_results[0]
        np.testing.assert_allclose(sr["control_rate"], 0.5)
        np.testing.assert_allclose(sr["treatment_rate"], 0.75)

    def test_significant_step(self):
        """Large lift is significant."""
        exp = FunnelExperiment()
        exp.add_step(100, 1000, 200, 1000, name="s1")
        result = exp.run()
        assert result.step_results[0]["significant"]


# ── Bottleneck detection ─────────────────────────────────────────────


class TestBottleneck:
    """Bottleneck step detection."""

    def test_bottleneck_is_worst_step(self):
        """Bottleneck is the step with the most negative (or smallest) lift."""
        exp = FunnelExperiment()
        exp.add_step(900, 1000, 910, 1000, name="landing")  # +1%
        exp.add_step(400, 900, 380, 910, name="cart")  # negative lift
        exp.add_step(200, 400, 220, 380, name="checkout")
        result = exp.run()
        assert result.bottleneck_step == "cart"


# ── Overall lift ─────────────────────────────────────────────────────


class TestOverallLift:
    """Overall funnel lift."""

    def test_overall_lift_computed(self):
        """Overall lift is end-to-end conversion difference."""
        exp = FunnelExperiment()
        exp.add_step(500, 1000, 600, 1000, name="s1")
        exp.add_step(200, 500, 300, 600, name="s2")
        result = exp.run()
        # Overall = last_converted/first_total for each group
        expected = (300 / 1000) - (200 / 1000)
        np.testing.assert_allclose(result.overall_lift, expected)


# ── Step names ───────────────────────────────────────────────────────


class TestStepNames:
    """Step naming behaviour."""

    def test_default_step_names(self):
        """Steps get default names when none provided."""
        exp = FunnelExperiment()
        exp.add_step(50, 100, 60, 100)
        exp.add_step(30, 50, 40, 60)
        result = exp.run()
        assert result.step_results[0]["name"] == "step_0"
        assert result.step_results[1]["name"] == "step_1"

    def test_constructor_step_names(self):
        """Step names from constructor are used."""
        exp = FunnelExperiment(step_names=["landing", "cart"])
        exp.add_step(50, 100, 60, 100)
        exp.add_step(30, 50, 40, 60)
        result = exp.run()
        assert result.step_results[0]["name"] == "landing"
        assert result.step_results[1]["name"] == "cart"


# ── Conditional conversion ───────────────────────────────────────────


class TestConditional:
    """Conditional conversion (step N | step N-1)."""

    def test_first_step_no_conditional(self):
        """First step has zero conditional lift."""
        exp = FunnelExperiment()
        exp.add_step(50, 100, 60, 100, name="s1")
        exp.add_step(30, 50, 40, 60, name="s2")
        result = exp.run()
        np.testing.assert_allclose(result.step_results[0]["conditional_lift"], 0.0)

    def test_conditional_computed(self):
        """Conditional conversion is computed for step 2+."""
        exp = FunnelExperiment()
        exp.add_step(50, 100, 60, 100, name="s1")
        exp.add_step(30, 50, 40, 60, name="s2")
        result = exp.run()
        # conditional: s2_converted / s1_converted
        expected = (40 / 60) - (30 / 50)
        np.testing.assert_allclose(
            result.step_results[1]["conditional_lift"], expected, atol=1e-10
        )


# ── Validation ───────────────────────────────────────────────────────


class TestValidation:
    """Input validation tests."""

    def test_no_steps_raises(self):
        """run() without steps raises ValueError."""
        with pytest.raises(ValueError, match="No funnel steps"):
            FunnelExperiment().run()

    def test_converted_exceeds_total_raises(self):
        """converted > total raises ValueError."""
        exp = FunnelExperiment()
        with pytest.raises(ValueError, match="can't exceed"):
            exp.add_step(150, 100, 50, 100)

    def test_negative_count_raises(self):
        """Negative counts raise ValueError."""
        exp = FunnelExperiment()
        with pytest.raises(ValueError, match="must be >= 0"):
            exp.add_step(-1, 100, 50, 100)

    def test_alpha_out_of_range(self):
        """alpha outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            FunnelExperiment(alpha=1.5)

    def test_float_count_raises(self):
        """Non-integer count raises ValueError."""
        exp = FunnelExperiment()
        with pytest.raises(ValueError, match="integer"):
            exp.add_step(50.5, 100, 60, 100)


# ── E2E scenario ─────────────────────────────────────────────────────


class TestE2E:
    """End-to-end scenario."""

    def test_ecommerce_funnel(self):
        """Full e-commerce funnel analysis."""
        exp = FunnelExperiment(
            step_names=["landing", "product_view", "cart", "checkout", "purchase"]
        )
        exp.add_step(9500, 10000, 9600, 10000)  # landing
        exp.add_step(4000, 9500, 4300, 9600)  # product view
        exp.add_step(1500, 4000, 1700, 4300)  # cart
        exp.add_step(800, 1500, 950, 1700)  # checkout
        exp.add_step(600, 800, 750, 950)  # purchase
        result = exp.run()
        assert len(result.step_results) == 5
        assert isinstance(result.bottleneck_step, str)
        assert isinstance(result.overall_lift, float)
