"""Tests for EValue sequential testing."""

from __future__ import annotations

import numpy as np
import pytest

from splita.sequential.evalue import EValue
from splita._types import EValueResult, EValueState

# Auto-tau warning is expected in tests that don't set tau explicitly
pytestmark = pytest.mark.filterwarnings(
    "ignore:tau auto-set:RuntimeWarning"
)


# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ─── Basic functionality ─────────────────────────────────────────────


class TestBasic:
    def test_update_returns_evalue_state(self, rng):
        ev = EValue(alpha=0.05, metric="continuous")
        ctrl = rng.normal(0.0, 1.0, size=100)
        trt = rng.normal(0.0, 1.0, size=100)
        state = ev.update(ctrl, trt)
        assert isinstance(state, EValueState)

    def test_result_returns_evalue_result(self, rng):
        ev = EValue(alpha=0.05, metric="continuous")
        ctrl = rng.normal(0.0, 1.0, size=100)
        trt = rng.normal(0.0, 1.0, size=100)
        ev.update(ctrl, trt)
        result = ev.result()
        assert isinstance(result, EValueResult)

    def test_no_effect_does_not_stop(self, rng):
        ev = EValue(alpha=0.05, metric="continuous")
        ctrl = rng.normal(0.0, 1.0, size=200)
        trt = rng.normal(0.0, 1.0, size=200)
        state = ev.update(ctrl, trt)
        assert not state.should_stop

    def test_large_effect_stops(self, rng):
        ev = EValue(alpha=0.05, metric="continuous")
        ctrl = rng.normal(0.0, 1.0, size=500)
        trt = rng.normal(0.5, 1.0, size=500)
        state = ev.update(ctrl, trt)
        assert state.should_stop
        assert state.e_value >= 1.0 / 0.05

    def test_e_value_is_positive(self, rng):
        ev = EValue(alpha=0.05, metric="continuous")
        ctrl = rng.normal(0.0, 1.0, size=100)
        trt = rng.normal(0.1, 1.0, size=100)
        state = ev.update(ctrl, trt)
        assert state.e_value > 0.0


# ─── Conversion metric ──────────────────────────────────────────────


class TestConversion:
    def test_conversion_metric(self, rng):
        ev = EValue(alpha=0.05, metric="conversion")
        ctrl = rng.binomial(1, 0.1, size=500).astype(float)
        trt = rng.binomial(1, 0.1, size=500).astype(float)
        state = ev.update(ctrl, trt)
        assert isinstance(state, EValueState)

    def test_conversion_with_large_effect(self, rng):
        ev = EValue(alpha=0.05, metric="conversion")
        ctrl = rng.binomial(1, 0.1, size=1000).astype(float)
        trt = rng.binomial(1, 0.3, size=1000).astype(float)
        state = ev.update(ctrl, trt)
        assert state.e_value > 1.0


# ─── Incremental updates ────────────────────────────────────────────


class TestIncremental:
    def test_multiple_updates(self, rng):
        ev = EValue(alpha=0.05, metric="continuous")
        for _ in range(5):
            ctrl = rng.normal(0.0, 1.0, size=50)
            trt = rng.normal(0.5, 1.0, size=50)
            state = ev.update(ctrl, trt)
        assert state.n_control == 250
        assert state.n_treatment == 250

    def test_empty_arrays_after_first_update(self, rng):
        ev = EValue(alpha=0.05, metric="continuous")
        ctrl = rng.normal(0.0, 1.0, size=100)
        trt = rng.normal(0.0, 1.0, size=100)
        state1 = ev.update(ctrl, trt)
        state2 = ev.update([], [])
        assert state2.e_value == state1.e_value


# ─── Tau parameter ───────────────────────────────────────────────────


class TestTau:
    def test_explicit_tau(self, rng):
        ev = EValue(alpha=0.05, metric="continuous", tau=0.1)
        ctrl = rng.normal(0.0, 1.0, size=100)
        trt = rng.normal(0.0, 1.0, size=100)
        state = ev.update(ctrl, trt)
        assert isinstance(state, EValueState)

    def test_auto_tau_warns(self, rng):
        ev = EValue(alpha=0.05, metric="continuous")
        ctrl = rng.normal(0.0, 1.0, size=100)
        trt = rng.normal(0.0, 1.0, size=100)
        with pytest.warns(RuntimeWarning, match="tau auto-set"):
            ev.update(ctrl, trt)


# ─── Result / stopping reason ───────────────────────────────────────


class TestResult:
    def test_stopping_reason_not_stopped(self, rng):
        ev = EValue(alpha=0.05, metric="continuous")
        ctrl = rng.normal(0.0, 1.0, size=50)
        trt = rng.normal(0.0, 1.0, size=50)
        ev.update(ctrl, trt)
        result = ev.result()
        assert result.stopping_reason == "not_stopped"

    def test_stopping_reason_threshold_crossed(self, rng):
        ev = EValue(alpha=0.05, metric="continuous")
        ctrl = rng.normal(0.0, 1.0, size=500)
        trt = rng.normal(0.5, 1.0, size=500)
        ev.update(ctrl, trt)
        result = ev.result()
        assert result.stopping_reason == "e_value_threshold_crossed"

    def test_result_before_update_raises(self):
        ev = EValue(alpha=0.05)
        with pytest.raises(RuntimeError, match="can't produce a result"):
            ev.result()


# ─── Validation ──────────────────────────────────────────────────────


class TestValidation:
    def test_alpha_out_of_range(self):
        with pytest.raises(ValueError, match="alpha"):
            EValue(alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            EValue(alpha=1.0)

    def test_invalid_metric(self):
        with pytest.raises(ValueError, match="metric"):
            EValue(metric="invalid")

    def test_negative_tau(self):
        with pytest.raises(ValueError, match="tau"):
            EValue(tau=-1.0)

    def test_empty_first_update_raises(self):
        ev = EValue(alpha=0.05)
        with pytest.raises(ValueError, match="can't both be empty"):
            ev.update([], [])


# ─── to_dict ─────────────────────────────────────────────────────────


class TestSerialization:
    def test_state_to_dict(self, rng):
        ev = EValue(alpha=0.05, metric="continuous")
        ctrl = rng.normal(0.0, 1.0, size=100)
        trt = rng.normal(0.0, 1.0, size=100)
        state = ev.update(ctrl, trt)
        d = state.to_dict()
        assert "e_value" in d
        assert "should_stop" in d

    def test_result_to_dict(self, rng):
        ev = EValue(alpha=0.05, metric="continuous")
        ctrl = rng.normal(0.0, 1.0, size=100)
        trt = rng.normal(0.0, 1.0, size=100)
        ev.update(ctrl, trt)
        result = ev.result()
        d = result.to_dict()
        assert "stopping_reason" in d


# ─── Edge cases ──────────────────────────────────────────────────────


class TestEdgeCases:
    def test_one_obs_each(self, rng):
        ev = EValue(alpha=0.05, metric="continuous", tau=1.0)
        state = ev.update([1.0], [2.0])
        assert state.n_control == 1
        assert state.n_treatment == 1

    def test_update_with_only_control_data(self):
        """Lines 155-161: when only control obs are present, e_value=1.0."""
        ev = EValue(alpha=0.05, metric="continuous", tau=1.0)
        # Only control data, no treatment
        state = ev.update([1.0, 2.0, 3.0], [])
        assert state.e_value == 1.0
        assert state.n_control == 3
        assert state.n_treatment == 0
        assert state.should_stop is False

    def test_update_with_only_treatment_data(self):
        """Lines 155-161: when only treatment obs are present, e_value=1.0."""
        ev = EValue(alpha=0.05, metric="continuous", tau=1.0)
        state = ev.update([], [1.0, 2.0, 3.0])
        assert state.e_value == 1.0
        assert state.n_control == 0
        assert state.n_treatment == 3
        assert state.should_stop is False

    def test_zero_variance_data(self):
        ev = EValue(alpha=0.05, metric="continuous", tau=1.0)
        ctrl = np.ones(100)
        trt = np.ones(100)
        state = ev.update(ctrl, trt)
        # Zero variance => e_value should be 1.0 (degenerate)
        assert state.e_value == 1.0
