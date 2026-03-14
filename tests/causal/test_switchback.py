"""Tests for SwitchbackExperiment (M16: Network Effects — Switchback)."""

from __future__ import annotations

import numpy as np
import pytest

from splita.causal.switchback import SwitchbackExperiment


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _make_switchback_data(rng, n_periods=20, n_per_period=50, effect=2.0):
    """Helper to create switchback data."""
    outcomes, treatments, periods = [], [], []
    for t in range(n_periods):
        trt = t % 2
        y = rng.normal(10 + effect * trt, 1, n_per_period)
        outcomes.extend(y.tolist())
        treatments.extend([float(trt)] * n_per_period)
        periods.extend([t] * n_per_period)
    return (
        np.array(outcomes),
        np.array(treatments),
        np.array(periods),
    )


class TestSwitchbackBasic:
    """Basic functionality tests."""

    def test_significant_effect(self, rng):
        """Large treatment effect should be detected."""
        outcomes, treatments, periods = _make_switchback_data(rng, effect=3.0)
        result = SwitchbackExperiment(outcomes, treatments, periods).run()
        assert result.significant is True
        assert result.lift > 0

    def test_no_effect(self, rng):
        """No treatment effect should yield non-significant result."""
        outcomes, treatments, periods = _make_switchback_data(rng, effect=0.0)
        result = SwitchbackExperiment(outcomes, treatments, periods).run()
        assert result.pvalue > 0.01

    def test_period_counts(self, rng):
        """Period counts should match the data."""
        n_periods = 20
        outcomes, treatments, periods = _make_switchback_data(
            rng, n_periods=n_periods
        )
        result = SwitchbackExperiment(outcomes, treatments, periods).run()
        assert result.n_periods == n_periods
        assert result.n_treatment_periods == n_periods // 2
        assert result.n_control_periods == n_periods // 2

    def test_lift_direction(self, rng):
        """Lift should be positive when treatment increases outcomes."""
        outcomes, treatments, periods = _make_switchback_data(rng, effect=5.0)
        result = SwitchbackExperiment(outcomes, treatments, periods).run()
        assert result.lift > 0

    def test_negative_effect(self, rng):
        """Negative treatment effect should produce negative lift."""
        outcomes, treatments, periods = _make_switchback_data(rng, effect=-3.0)
        result = SwitchbackExperiment(outcomes, treatments, periods).run()
        assert result.lift < 0

    def test_ci_contains_lift(self, rng):
        """CI should contain the point estimate."""
        outcomes, treatments, periods = _make_switchback_data(rng)
        result = SwitchbackExperiment(outcomes, treatments, periods).run()
        assert result.ci_lower <= result.lift <= result.ci_upper

    def test_ci_contains_true_effect(self, rng):
        """CI should contain the true effect (most of the time)."""
        true_effect = 2.0
        outcomes, treatments, periods = _make_switchback_data(
            rng, n_periods=40, effect=true_effect
        )
        result = SwitchbackExperiment(outcomes, treatments, periods).run()
        assert result.ci_lower <= true_effect <= result.ci_upper

    def test_alpha_parameter(self, rng):
        outcomes, treatments, periods = _make_switchback_data(rng, effect=1.0)

        result_05 = SwitchbackExperiment(
            outcomes, treatments, periods, alpha=0.05
        ).run()
        result_001 = SwitchbackExperiment(
            outcomes, treatments, periods, alpha=0.001
        ).run()

        # Same p-value
        assert abs(result_05.pvalue - result_001.pvalue) < 0.001

    def test_odd_number_of_periods(self, rng):
        """Should handle odd number of periods."""
        outcomes, treatments, periods = _make_switchback_data(
            rng, n_periods=11
        )
        result = SwitchbackExperiment(outcomes, treatments, periods).run()
        assert result.n_periods == 11
        # 6 even (ctrl) + 5 odd (trt) or vice versa
        assert result.n_treatment_periods + result.n_control_periods == 11

    def test_unequal_period_sizes(self, rng):
        """Periods with different numbers of observations should work."""
        outcomes, treatments, periods = [], [], []
        for t in range(10):
            n = rng.integers(20, 80)
            trt = float(t % 2)
            y = rng.normal(10 + 2 * trt, 1, n)
            outcomes.extend(y.tolist())
            treatments.extend([trt] * n)
            periods.extend([t] * n)
        result = SwitchbackExperiment(
            np.array(outcomes),
            np.array(treatments),
            np.array(periods),
        ).run()
        assert result.n_periods == 10


class TestSwitchbackValidation:
    """Tests for input validation."""

    def test_mismatched_lengths(self, rng):
        with pytest.raises(ValueError, match="same length"):
            SwitchbackExperiment(
                rng.normal(0, 1, 50),
                rng.normal(0, 1, 30),
                np.arange(50),
            )

    def test_mismatched_periods_length(self, rng):
        with pytest.raises(ValueError, match="same length"):
            SwitchbackExperiment(
                rng.normal(0, 1, 50),
                np.zeros(50),
                np.arange(30),
            )

    def test_treatments_not_binary(self, rng):
        with pytest.raises(ValueError, match="must contain only 0 and 1"):
            SwitchbackExperiment(
                rng.normal(0, 1, 50),
                rng.integers(0, 3, 50).astype(float),
                np.arange(50),
            )

    def test_too_few_control_periods(self, rng):
        """All periods are treatment — not enough control."""
        outcomes = rng.normal(0, 1, 50)
        treatments = np.ones(50)
        periods = np.repeat(np.arange(5), 10)
        with pytest.raises(ValueError, match="at least 2 control"):
            SwitchbackExperiment(outcomes, treatments, periods).run()

    def test_too_few_treatment_periods(self, rng):
        """All periods are control — not enough treatment."""
        outcomes = rng.normal(0, 1, 50)
        treatments = np.zeros(50)
        periods = np.repeat(np.arange(5), 10)
        with pytest.raises(ValueError, match="at least 2 treatment"):
            SwitchbackExperiment(outcomes, treatments, periods).run()

    def test_invalid_alpha(self, rng):
        with pytest.raises(ValueError, match="alpha"):
            SwitchbackExperiment(
                rng.normal(0, 1, 50),
                np.zeros(50),
                np.arange(50),
                alpha=1.5,
            )

    def test_2d_periods_rejected(self, rng):
        with pytest.raises(ValueError, match="1-D"):
            SwitchbackExperiment(
                rng.normal(0, 1, 20),
                np.zeros(20),
                np.zeros((20, 2)),
            )


class TestSwitchbackResult:
    """Tests for result properties."""

    def test_to_dict(self, rng):
        outcomes, treatments, periods = _make_switchback_data(rng)
        result = SwitchbackExperiment(outcomes, treatments, periods).run()
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "lift" in d
        assert "n_periods" in d

    def test_repr(self, rng):
        outcomes, treatments, periods = _make_switchback_data(rng)
        result = SwitchbackExperiment(outcomes, treatments, periods).run()
        assert "SwitchbackResult" in repr(result)
