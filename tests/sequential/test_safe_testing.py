"""Tests for EProcess (safe testing, Grunwald et al. 2020)."""

from __future__ import annotations

import numpy as np
import pytest

from splita._types import EProcessResult, EProcessState
from splita.sequential.safe_testing import EProcess


# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ─── Basic functionality ─────────────────────────────────────────────


class TestBasic:
    def test_update_returns_eprocess_state(self, rng):
        ep = EProcess(alpha=0.05)
        ctrl = rng.normal(0.0, 1.0, size=100)
        trt = rng.normal(0.0, 1.0, size=100)
        state = ep.update(ctrl, trt)
        assert isinstance(state, EProcessState)

    def test_result_returns_eprocess_result(self, rng):
        ep = EProcess(alpha=0.05)
        ctrl = rng.normal(0.0, 1.0, size=100)
        trt = rng.normal(0.0, 1.0, size=100)
        ep.update(ctrl, trt)
        result = ep.result()
        assert isinstance(result, EProcessResult)

    def test_no_effect_does_not_stop(self, rng):
        ep = EProcess(alpha=0.05)
        ctrl = rng.normal(0.0, 1.0, size=200)
        trt = rng.normal(0.0, 1.0, size=200)
        state = ep.update(ctrl, trt)
        assert not state.should_stop

    def test_large_effect_stops(self, rng):
        ep = EProcess(alpha=0.05, method="grapa")
        ctrl = rng.normal(0.0, 1.0, size=500)
        trt = rng.normal(0.5, 1.0, size=500)
        state = ep.update(ctrl, trt)
        assert state.should_stop
        assert state.e_value >= 1.0 / 0.05

    def test_e_value_is_positive(self, rng):
        ep = EProcess(alpha=0.05)
        ctrl = rng.normal(0.0, 1.0, size=100)
        trt = rng.normal(0.1, 1.0, size=100)
        state = ep.update(ctrl, trt)
        assert state.e_value > 0.0


# ─── Multiplicative accumulation ────────────────────────────────────


class TestMultiplicative:
    def test_e_value_accumulates_multiplicatively(self, rng):
        """E-process is a product of sequential e-values, so log should grow."""
        ep = EProcess(alpha=0.05, method="grapa")
        log_values = []
        for _ in range(5):
            ctrl = rng.normal(0.0, 1.0, size=100)
            trt = rng.normal(0.3, 1.0, size=100)
            state = ep.update(ctrl, trt)
            log_values.append(state.log_e_value)
        # Under the alternative, log e-value should generally increase
        assert log_values[-1] > log_values[0]

    def test_log_e_value_consistent_with_e_value(self, rng):
        """log(e_value) should equal log_e_value."""
        ep = EProcess(alpha=0.05)
        ctrl = rng.normal(0.0, 1.0, size=200)
        trt = rng.normal(0.2, 1.0, size=200)
        state = ep.update(ctrl, trt)
        assert abs(np.log(state.e_value) - state.log_e_value) < 1e-6

    def test_single_update_e_value_ge_one_under_alternative(self, rng):
        """With a real effect, a single batch should yield e >= 1."""
        ep = EProcess(alpha=0.05, method="grapa")
        ctrl = rng.normal(0.0, 1.0, size=500)
        trt = rng.normal(0.5, 1.0, size=500)
        state = ep.update(ctrl, trt)
        assert state.e_value >= 1.0


# ─── Type I error control ───────────────────────────────────────────


class TestTypeIError:
    def test_rejection_under_null(self):
        """Under the null, rejection rate <= alpha + tolerance."""
        n_sims = 1000
        alpha = 0.05
        rejections = 0
        rng = np.random.default_rng(314)

        for _ in range(n_sims):
            ep = EProcess(alpha=alpha, method="grapa")
            ctrl = rng.normal(0, 1, size=200)
            trt = rng.normal(0, 1, size=200)
            state = ep.update(ctrl, trt)
            if state.should_stop:
                rejections += 1

        rejection_rate = rejections / n_sims
        assert rejection_rate <= alpha + 0.03, (
            f"Type I error rate {rejection_rate:.3f} exceeds alpha={alpha} + 0.03"
        )

    def test_always_valid_with_peeking(self):
        """Under null with sequential peeking, FP rate is controlled."""
        n_sims = 1000
        alpha = 0.05
        false_positives = 0
        rng = np.random.default_rng(42)

        for _ in range(n_sims):
            ep = EProcess(alpha=alpha, method="grapa")
            ever_rejected = False
            for _day in range(10):
                ctrl = rng.normal(0, 1, size=50)
                trt = rng.normal(0, 1, size=50)
                state = ep.update(ctrl, trt)
                if state.should_stop:
                    ever_rejected = True
                    break
            if ever_rejected:
                false_positives += 1

        rejection_rate = false_positives / n_sims
        assert rejection_rate <= alpha + 0.03, (
            f"FP rate {rejection_rate:.3f} exceeds alpha={alpha} + tolerance"
        )


# ─── Safe CI ─────────────────────────────────────────────────────────


class TestSafeCI:
    def test_safe_ci_contains_zero_under_null(self, rng):
        """Under null, safe CI should contain 0 most of the time."""
        n_sims = 500
        contains_zero = 0
        for i in range(n_sims):
            seed_rng = np.random.default_rng(i + 3000)
            ep = EProcess(alpha=0.05)
            ctrl = seed_rng.normal(0, 1, size=200)
            trt = seed_rng.normal(0, 1, size=200)
            ep.update(ctrl, trt)
            result = ep.result()
            if result.safe_ci_lower <= 0.0 <= result.safe_ci_upper:
                contains_zero += 1
        coverage = contains_zero / n_sims
        assert coverage >= 0.90, (
            f"Safe CI coverage {coverage:.3f} is below 0.90"
        )

    def test_safe_ci_lower_less_than_upper(self, rng):
        ep = EProcess(alpha=0.05)
        ctrl = rng.normal(0.0, 1.0, size=200)
        trt = rng.normal(0.0, 1.0, size=200)
        ep.update(ctrl, trt)
        result = ep.result()
        assert result.safe_ci_lower < result.safe_ci_upper


# ─── Method variants ────────────────────────────────────────────────


class TestMethods:
    def test_grapa_method(self, rng):
        ep = EProcess(alpha=0.05, method="grapa")
        ctrl = rng.normal(0.0, 1.0, size=200)
        trt = rng.normal(0.0, 1.0, size=200)
        state = ep.update(ctrl, trt)
        assert isinstance(state, EProcessState)

    def test_universal_method(self, rng):
        ep = EProcess(alpha=0.05, method="universal")
        ctrl = rng.normal(0.0, 1.0, size=200)
        trt = rng.normal(0.0, 1.0, size=200)
        state = ep.update(ctrl, trt)
        assert isinstance(state, EProcessState)

    def test_both_methods_detect_effect(self, rng):
        """Both methods should detect a large effect."""
        ctrl = rng.normal(0.0, 1.0, size=500)
        trt = rng.normal(0.3, 1.0, size=500)

        grapa = EProcess(alpha=0.05, method="grapa")
        grapa_state = grapa.update(ctrl, trt)

        universal = EProcess(alpha=0.05, method="universal")
        universal_state = universal.update(ctrl, trt)

        # Both should produce e-values above 1
        assert grapa_state.e_value > 1.0
        assert universal_state.e_value > 1.0


# ─── Comparison with basic EValue ────────────────────────────────────


class TestComparisonWithEValue:
    def test_eprocess_vs_evalue_direction(self, rng):
        """Both should agree on the direction of evidence."""
        from splita.sequential.evalue import EValue

        ctrl = rng.normal(0.0, 1.0, size=500)
        trt = rng.normal(0.5, 1.0, size=500)

        ep = EProcess(alpha=0.05, method="grapa")
        ep_state = ep.update(ctrl, trt)

        ev = EValue(alpha=0.05, metric="continuous", tau=0.25)
        ev_state = ev.update(ctrl, trt)

        # Both should detect the effect
        assert ep_state.e_value > 1.0
        assert ev_state.e_value > 1.0


# ─── Incremental updates ────────────────────────────────────────────


class TestIncremental:
    def test_multiple_updates_accumulate(self, rng):
        ep = EProcess(alpha=0.05)
        for i in range(5):
            ctrl = rng.normal(0.0, 1.0, size=50)
            trt = rng.normal(0.0, 1.0, size=50)
            state = ep.update(ctrl, trt)
        assert state.n_control == 250
        assert state.n_treatment == 250

    def test_empty_arrays_after_first_update(self, rng):
        ep = EProcess(alpha=0.05)
        ctrl = rng.normal(0.0, 1.0, size=100)
        trt = rng.normal(0.0, 1.0, size=100)
        state1 = ep.update(ctrl, trt)
        state2 = ep.update([], [])
        assert state2.e_value == state1.e_value


# ─── Result / stopping reason ───────────────────────────────────────


class TestResult:
    def test_stopping_reason_not_stopped(self, rng):
        ep = EProcess(alpha=0.05)
        ctrl = rng.normal(0.0, 1.0, size=50)
        trt = rng.normal(0.0, 1.0, size=50)
        ep.update(ctrl, trt)
        result = ep.result()
        assert result.stopping_reason == "not_stopped"

    def test_stopping_reason_threshold_crossed(self, rng):
        ep = EProcess(alpha=0.05)
        ctrl = rng.normal(0.0, 1.0, size=500)
        trt = rng.normal(0.5, 1.0, size=500)
        ep.update(ctrl, trt)
        result = ep.result()
        assert result.stopping_reason == "e_process_threshold_crossed"

    def test_result_before_update_raises(self):
        ep = EProcess(alpha=0.05)
        with pytest.raises(RuntimeError, match="can't produce a result"):
            ep.result()


# ─── Validation ──────────────────────────────────────────────────────


class TestValidation:
    def test_alpha_out_of_range(self):
        with pytest.raises(ValueError, match="alpha"):
            EProcess(alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            EProcess(alpha=1.0)

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method"):
            EProcess(method="invalid")

    def test_empty_first_update_raises(self):
        ep = EProcess(alpha=0.05)
        with pytest.raises(ValueError, match="can't both be empty"):
            ep.update([], [])


# ─── Serialization ──────────────────────────────────────────────────


class TestSerialization:
    def test_state_to_dict(self, rng):
        ep = EProcess(alpha=0.05)
        ctrl = rng.normal(0.0, 1.0, size=100)
        trt = rng.normal(0.0, 1.0, size=100)
        state = ep.update(ctrl, trt)
        d = state.to_dict()
        assert "e_value" in d
        assert "log_e_value" in d
        assert "should_stop" in d

    def test_result_to_dict(self, rng):
        ep = EProcess(alpha=0.05)
        ctrl = rng.normal(0.0, 1.0, size=100)
        trt = rng.normal(0.0, 1.0, size=100)
        ep.update(ctrl, trt)
        result = ep.result()
        d = result.to_dict()
        assert "stopping_reason" in d
        assert "safe_ci_lower" in d


# ─── Edge cases ──────────────────────────────────────────────────────


class TestEdgeCases:
    def test_one_obs_each(self):
        ep = EProcess(alpha=0.05)
        state = ep.update([1.0], [2.0])
        assert state.n_control == 1
        assert state.n_treatment == 1

    def test_only_control_data(self):
        ep = EProcess(alpha=0.05)
        state = ep.update([1.0, 2.0, 3.0], [])
        assert state.e_value == 1.0
        assert state.n_control == 3
        assert state.n_treatment == 0
        assert state.should_stop is False

    def test_zero_variance_data(self):
        ep = EProcess(alpha=0.05)
        ctrl = np.ones(100)
        trt = np.ones(100)
        state = ep.update(ctrl, trt)
        assert state.e_value >= 0.0

    def test_repr_does_not_crash(self, rng):
        ep = EProcess(alpha=0.05)
        ctrl = rng.normal(0.0, 1.0, size=50)
        trt = rng.normal(0.0, 1.0, size=50)
        state = ep.update(ctrl, trt)
        result = ep.result()
        assert "EProcessState" in repr(state)
        assert "EProcessResult" in repr(result)

    def test_universal_zero_variance(self):
        """Line 290: universal method with V<=0 returns 0."""
        ep = EProcess(alpha=0.05, method="universal")
        ctrl = np.ones(100)
        trt = np.ones(100)
        state = ep.update(ctrl, trt)
        # With zero variance, universal log_e should be 0 => e_value = 1
        assert state.e_value == pytest.approx(1.0)

    def test_safe_ci_only_control_data(self):
        """Line 308: safe CI with n_t==0 returns (-inf, inf)."""
        ep = EProcess(alpha=0.05)
        ep.update([1.0, 2.0, 3.0], [])
        result = ep.result()
        assert result.safe_ci_lower == float("-inf")
        assert result.safe_ci_upper == float("inf")

    def test_safe_ci_zero_variance(self):
        """Line 316: safe CI with V<=0 returns (delta_hat, delta_hat)."""
        ep = EProcess(alpha=0.05)
        # Constant but different values => V=0, delta_hat != 0
        ctrl = np.full(100, 5.0)
        trt = np.full(100, 7.0)
        state = ep.update(ctrl, trt)
        result = ep.result()
        assert result.safe_ci_lower == result.safe_ci_upper
        assert result.safe_ci_lower == pytest.approx(2.0)
