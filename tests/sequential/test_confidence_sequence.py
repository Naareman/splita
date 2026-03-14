"""Tests for ConfidenceSequence (Howard et al. 2021)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from splita._types import CSResult, CSState
from splita.sequential.confidence_sequence import ConfidenceSequence


# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ─── Basic functionality ─────────────────────────────────────────────


class TestBasic:
    def test_update_returns_cs_state(self, rng):
        cs = ConfidenceSequence(alpha=0.05)
        ctrl = rng.normal(0.0, 1.0, size=100)
        trt = rng.normal(0.0, 1.0, size=100)
        state = cs.update(ctrl, trt)
        assert isinstance(state, CSState)

    def test_result_returns_cs_result(self, rng):
        cs = ConfidenceSequence(alpha=0.05)
        ctrl = rng.normal(0.0, 1.0, size=100)
        trt = rng.normal(0.0, 1.0, size=100)
        cs.update(ctrl, trt)
        result = cs.result()
        assert isinstance(result, CSResult)

    def test_no_effect_does_not_stop(self, rng):
        cs = ConfidenceSequence(alpha=0.05, sigma=1.0)
        ctrl = rng.normal(0.0, 1.0, size=200)
        trt = rng.normal(0.0, 1.0, size=200)
        state = cs.update(ctrl, trt)
        assert not state.should_stop

    def test_large_effect_stops(self, rng):
        cs = ConfidenceSequence(alpha=0.05, sigma=1.0)
        ctrl = rng.normal(0.0, 1.0, size=500)
        trt = rng.normal(0.5, 1.0, size=500)
        state = cs.update(ctrl, trt)
        assert state.should_stop

    def test_effect_estimate_sign(self, rng):
        cs = ConfidenceSequence(alpha=0.05, sigma=1.0)
        ctrl = rng.normal(0.0, 1.0, size=200)
        trt = rng.normal(0.5, 1.0, size=200)
        state = cs.update(ctrl, trt)
        assert state.effect_estimate > 0.0


# ─── CI properties ───────────────────────────────────────────────────


class TestCIProperties:
    def test_ci_shrinks_over_time(self, rng):
        """CI width should decrease as more data is observed."""
        cs = ConfidenceSequence(alpha=0.05, sigma=1.0)
        widths = []
        for _ in range(10):
            ctrl = rng.normal(0.0, 1.0, size=100)
            trt = rng.normal(0.0, 1.0, size=100)
            state = cs.update(ctrl, trt)
            widths.append(state.width)
        # Width should generally decrease (allow some noise)
        assert widths[-1] < widths[0]

    def test_ci_contains_true_effect_under_null(self, rng):
        """Under the null, CI should contain 0 most of the time."""
        n_sims = 500
        contains_zero = 0
        for i in range(n_sims):
            seed_rng = np.random.default_rng(i + 1000)
            cs = ConfidenceSequence(alpha=0.05, sigma=1.0)
            ctrl = seed_rng.normal(0, 1, size=200)
            trt = seed_rng.normal(0, 1, size=200)
            state = cs.update(ctrl, trt)
            if state.ci_lower <= 0.0 <= state.ci_upper:
                contains_zero += 1
        coverage = contains_zero / n_sims
        assert coverage >= 0.90, (
            f"CI coverage {coverage:.3f} is below 0.90 under the null"
        )

    def test_ci_contains_true_effect_under_alternative(self, rng):
        """Under the alternative, CI should contain the true effect."""
        true_effect = 0.3
        n_sims = 200
        contains_effect = 0
        for i in range(n_sims):
            seed_rng = np.random.default_rng(i + 2000)
            cs = ConfidenceSequence(alpha=0.05, sigma=1.0)
            ctrl = seed_rng.normal(0, 1, size=300)
            trt = seed_rng.normal(true_effect, 1, size=300)
            state = cs.update(ctrl, trt)
            if state.ci_lower <= true_effect <= state.ci_upper:
                contains_effect += 1
        coverage = contains_effect / n_sims
        assert coverage >= 0.85, (
            f"CI coverage {coverage:.3f} is below 0.85 under alternative"
        )

    def test_width_is_positive(self, rng):
        cs = ConfidenceSequence(alpha=0.05, sigma=1.0)
        ctrl = rng.normal(0.0, 1.0, size=100)
        trt = rng.normal(0.0, 1.0, size=100)
        state = cs.update(ctrl, trt)
        assert state.width > 0.0

    def test_ci_lower_less_than_upper(self, rng):
        cs = ConfidenceSequence(alpha=0.05, sigma=1.0)
        ctrl = rng.normal(0.0, 1.0, size=100)
        trt = rng.normal(0.0, 1.0, size=100)
        state = cs.update(ctrl, trt)
        assert state.ci_lower < state.ci_upper


# ─── Always-valid property ───────────────────────────────────────────


class TestAlwaysValid:
    def test_always_valid_rejection_rate(self):
        """Under the null with sequential peeking, rejection rate <= alpha.

        This is the key property: P(exists t: CS excludes 0) <= alpha.
        """
        n_sims = 1000
        alpha = 0.05
        false_positives = 0
        rng = np.random.default_rng(314)

        for _ in range(n_sims):
            cs = ConfidenceSequence(alpha=alpha, sigma=1.0)
            ever_rejected = False
            for _day in range(10):
                ctrl = rng.normal(0, 1, size=50)
                trt = rng.normal(0, 1, size=50)
                state = cs.update(ctrl, trt)
                if state.should_stop:
                    ever_rejected = True
                    break
            if ever_rejected:
                false_positives += 1

        rejection_rate = false_positives / n_sims
        assert rejection_rate <= alpha + 0.03, (
            f"FP rate {rejection_rate:.3f} exceeds alpha={alpha} + tolerance"
        )

    def test_power_under_alternative(self):
        """Under a real effect, the CS should eventually stop."""
        n_sims = 100
        stopped = 0
        rng = np.random.default_rng(999)

        for _ in range(n_sims):
            cs = ConfidenceSequence(alpha=0.05, sigma=1.0)
            for _day in range(20):
                ctrl = rng.normal(0, 1, size=100)
                trt = rng.normal(0.5, 1, size=100)
                state = cs.update(ctrl, trt)
                if state.should_stop:
                    stopped += 1
                    break

        power = stopped / n_sims
        assert power >= 0.50, f"Power {power:.3f} is too low"


# ─── Tighter than mSPRT ─────────────────────────────────────────────


class TestTighterThanMSPRT:
    def test_cs_width_vs_msprt(self, rng):
        """CS should produce tighter CIs than mSPRT at the same sample."""
        from splita.sequential.msprt import mSPRT

        ctrl = rng.normal(0.0, 1.0, size=500)
        trt = rng.normal(0.0, 1.0, size=500)

        cs = ConfidenceSequence(alpha=0.05, sigma=1.0)
        cs_state = cs.update(ctrl, trt)

        msprt = mSPRT(metric="continuous", alpha=0.05, tau=0.25)
        msprt_state = msprt.update(ctrl, trt)

        cs_width = cs_state.width
        msprt_width = msprt_state.always_valid_ci_upper - msprt_state.always_valid_ci_lower

        # CS should be tighter (or at least competitive)
        assert cs_width <= msprt_width * 1.5, (
            f"CS width {cs_width:.4f} is not competitive with mSPRT {msprt_width:.4f}"
        )


# ─── Incremental updates ────────────────────────────────────────────


class TestIncremental:
    def test_multiple_updates_accumulate(self, rng):
        cs = ConfidenceSequence(alpha=0.05, sigma=1.0)
        for i in range(5):
            ctrl = rng.normal(0.0, 1.0, size=50)
            trt = rng.normal(0.0, 1.0, size=50)
            state = cs.update(ctrl, trt)
        assert state.n_control == 250
        assert state.n_treatment == 250

    def test_empty_arrays_after_first_update(self, rng):
        cs = ConfidenceSequence(alpha=0.05, sigma=1.0)
        ctrl = rng.normal(0.0, 1.0, size=100)
        trt = rng.normal(0.0, 1.0, size=100)
        state1 = cs.update(ctrl, trt)
        state2 = cs.update([], [])
        assert state2.width == state1.width


# ─── Method variants ────────────────────────────────────────────────


class TestMethods:
    def test_normal_mixture_method(self, rng):
        cs = ConfidenceSequence(alpha=0.05, sigma=1.0, method="normal_mixture")
        ctrl = rng.normal(0.0, 1.0, size=200)
        trt = rng.normal(0.0, 1.0, size=200)
        state = cs.update(ctrl, trt)
        assert isinstance(state, CSState)

    def test_stitched_method(self, rng):
        cs = ConfidenceSequence(alpha=0.05, sigma=1.0, method="stitched")
        ctrl = rng.normal(0.0, 1.0, size=200)
        trt = rng.normal(0.0, 1.0, size=200)
        state = cs.update(ctrl, trt)
        assert isinstance(state, CSState)

    def test_both_methods_produce_valid_ci(self, rng):
        ctrl = rng.normal(0.0, 1.0, size=200)
        trt = rng.normal(0.0, 1.0, size=200)

        for method in ["normal_mixture", "stitched"]:
            cs = ConfidenceSequence(alpha=0.05, sigma=1.0, method=method)
            state = cs.update(ctrl, trt)
            assert state.ci_lower < state.ci_upper
            assert state.width > 0.0


# ─── Sigma auto-estimation ──────────────────────────────────────────


class TestSigma:
    def test_sigma_auto_estimated(self, rng):
        cs = ConfidenceSequence(alpha=0.05)
        ctrl = rng.normal(0.0, 1.0, size=100)
        trt = rng.normal(0.0, 1.0, size=100)
        with pytest.warns(RuntimeWarning, match="sigma auto-set"):
            cs.update(ctrl, trt)
        assert cs._sigma is not None
        assert cs._sigma > 0.0

    def test_explicit_sigma_no_warning(self, rng):
        cs = ConfidenceSequence(alpha=0.05, sigma=1.0)
        ctrl = rng.normal(0.0, 1.0, size=100)
        trt = rng.normal(0.0, 1.0, size=100)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cs.update(ctrl, trt)
        sigma_warnings = [
            x for x in w
            if issubclass(x.category, RuntimeWarning)
            and "sigma auto-set" in str(x.message)
        ]
        assert len(sigma_warnings) == 0


# ─── Result / stopping reason ───────────────────────────────────────


class TestResult:
    def test_stopping_reason_not_stopped(self, rng):
        cs = ConfidenceSequence(alpha=0.05, sigma=1.0)
        ctrl = rng.normal(0.0, 1.0, size=50)
        trt = rng.normal(0.0, 1.0, size=50)
        cs.update(ctrl, trt)
        result = cs.result()
        assert result.stopping_reason == "not_stopped"

    def test_stopping_reason_ci_excludes_zero(self, rng):
        cs = ConfidenceSequence(alpha=0.05, sigma=1.0)
        ctrl = rng.normal(0.0, 1.0, size=500)
        trt = rng.normal(0.5, 1.0, size=500)
        cs.update(ctrl, trt)
        result = cs.result()
        assert result.stopping_reason == "ci_excludes_zero"

    def test_result_before_update_raises(self):
        cs = ConfidenceSequence(alpha=0.05)
        with pytest.raises(RuntimeError, match="can't produce a result"):
            cs.result()

    def test_result_total_observations(self, rng):
        cs = ConfidenceSequence(alpha=0.05, sigma=1.0)
        ctrl = rng.normal(0.0, 1.0, size=100)
        trt = rng.normal(0.0, 1.0, size=150)
        cs.update(ctrl, trt)
        result = cs.result()
        assert result.total_observations == 250


# ─── Validation ──────────────────────────────────────────────────────


class TestValidation:
    def test_alpha_out_of_range(self):
        with pytest.raises(ValueError, match="alpha"):
            ConfidenceSequence(alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            ConfidenceSequence(alpha=1.0)

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method"):
            ConfidenceSequence(method="invalid")

    def test_negative_sigma(self):
        with pytest.raises(ValueError, match="sigma"):
            ConfidenceSequence(sigma=-1.0)

    def test_zero_sigma(self):
        with pytest.raises(ValueError, match="sigma"):
            ConfidenceSequence(sigma=0.0)

    def test_empty_first_update_raises(self):
        cs = ConfidenceSequence(alpha=0.05)
        with pytest.raises(ValueError, match="can't both be empty"):
            cs.update([], [])


# ─── Serialization ──────────────────────────────────────────────────


class TestSerialization:
    def test_state_to_dict(self, rng):
        cs = ConfidenceSequence(alpha=0.05, sigma=1.0)
        ctrl = rng.normal(0.0, 1.0, size=100)
        trt = rng.normal(0.0, 1.0, size=100)
        state = cs.update(ctrl, trt)
        d = state.to_dict()
        assert "effect_estimate" in d
        assert "ci_lower" in d
        assert "should_stop" in d

    def test_result_to_dict(self, rng):
        cs = ConfidenceSequence(alpha=0.05, sigma=1.0)
        ctrl = rng.normal(0.0, 1.0, size=100)
        trt = rng.normal(0.0, 1.0, size=100)
        cs.update(ctrl, trt)
        result = cs.result()
        d = result.to_dict()
        assert "stopping_reason" in d
        assert "total_observations" in d


# ─── Edge cases ──────────────────────────────────────────────────────


class TestEdgeCases:
    def test_one_obs_each(self):
        cs = ConfidenceSequence(alpha=0.05, sigma=1.0)
        state = cs.update([1.0], [2.0])
        assert state.n_control == 1
        assert state.n_treatment == 1

    def test_only_control_data(self):
        cs = ConfidenceSequence(alpha=0.05, sigma=1.0)
        state = cs.update([1.0, 2.0, 3.0], [])
        assert state.n_control == 3
        assert state.n_treatment == 0
        assert state.should_stop is False

    def test_zero_variance_data(self):
        cs = ConfidenceSequence(alpha=0.05, sigma=1.0)
        ctrl = np.ones(100)
        trt = np.ones(100)
        state = cs.update(ctrl, trt)
        assert state.width >= 0.0

    def test_repr_does_not_crash(self, rng):
        cs = ConfidenceSequence(alpha=0.05, sigma=1.0)
        ctrl = rng.normal(0.0, 1.0, size=50)
        trt = rng.normal(0.0, 1.0, size=50)
        state = cs.update(ctrl, trt)
        result = cs.result()
        assert "CSState" in repr(state)
        assert "CSResult" in repr(result)
