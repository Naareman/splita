"""Tests for the mSPRT (mixture Sequential Probability Ratio Test) class."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from splita._types import mSPRTResult, mSPRTState
from splita.sequential.msprt import mSPRT

# ─── Basic ─────────────────────────────────────────────────────────────


class TestBasic:
    """Single batch, incremental, significant, and non-significant cases."""

    def test_single_batch(self):
        """One update with all data returns a valid mSPRTState."""
        rng = np.random.default_rng(42)
        ctrl = rng.normal(0.0, 1.0, size=200)
        trt = rng.normal(0.0, 1.0, size=200)
        test = mSPRT(metric="continuous", alpha=0.05)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            state = test.update(ctrl, trt)

        assert isinstance(state, mSPRTState)
        assert state.n_control == 200
        assert state.n_treatment == 200
        assert state.mixture_lr >= 0.0
        assert 0.0 <= state.always_valid_pvalue <= 1.0

    def test_incremental_updates_approximate_single_batch(self):
        """Multiple small batches give approximately the same result as one batch."""
        rng = np.random.default_rng(99)
        ctrl = rng.normal(0.0, 1.0, size=400)
        trt = rng.normal(0.3, 1.0, size=400)

        # Single batch
        test_batch = mSPRT(metric="continuous", alpha=0.05, tau=0.25)
        state_batch = test_batch.update(ctrl, trt)

        # Incremental (4 batches of 100)
        test_inc = mSPRT(metric="continuous", alpha=0.05, tau=0.25)
        for i in range(4):
            s = slice(i * 100, (i + 1) * 100)
            state_inc = test_inc.update(ctrl[s], trt[s])

        assert state_inc.n_control == state_batch.n_control
        assert state_inc.n_treatment == state_batch.n_treatment
        # The MLR depends on running variance estimates, which differ slightly
        # between single-pass and incremental. Allow some tolerance.
        assert (
            abs(state_inc.mixture_lr - state_batch.mixture_lr)
            / max(state_batch.mixture_lr, 1e-10)
            < 0.15
        )

    def test_significant_result(self):
        """Large effect -> should_stop=True, p < alpha."""
        rng = np.random.default_rng(7)
        ctrl = rng.normal(0.0, 1.0, size=500)
        trt = rng.normal(0.5, 1.0, size=500)
        test = mSPRT(metric="continuous", alpha=0.05, tau=0.25)
        state = test.update(ctrl, trt)

        assert state.should_stop is True
        assert state.always_valid_pvalue < 0.05

    def test_non_significant_result(self):
        """No effect -> should_stop=False, p > alpha."""
        rng = np.random.default_rng(123)
        ctrl = rng.normal(0.0, 1.0, size=100)
        trt = rng.normal(0.0, 1.0, size=100)
        test = mSPRT(metric="continuous", alpha=0.05, tau=0.25)
        state = test.update(ctrl, trt)

        assert state.should_stop is False
        assert state.always_valid_pvalue > 0.05


# ─── Statistical correctness ──────────────────────────────────────────


class TestStatisticalCorrectness:
    """Always-valid p-value properties and Type I error control."""

    def test_pvalue_in_zero_one(self):
        """Always-valid p-value is always in [0, 1]."""
        rng = np.random.default_rng(1)
        for _ in range(20):
            ctrl = rng.normal(0, 1, size=50)
            trt = rng.normal(0, 1, size=50)
            test = mSPRT(metric="continuous", alpha=0.05, tau=0.25)
            state = test.update(ctrl, trt)
            assert 0.0 <= state.always_valid_pvalue <= 1.0

    def test_mlr_non_negative(self):
        """Mixture likelihood ratio is always >= 0."""
        rng = np.random.default_rng(2)
        ctrl = rng.normal(0, 1, size=200)
        trt = rng.normal(0.1, 1, size=200)
        test = mSPRT(metric="continuous", alpha=0.05, tau=0.25)
        state = test.update(ctrl, trt)
        assert state.mixture_lr >= 0.0

    def test_type_i_error_control(self):
        """Under the null, rejection rate <= alpha + tolerance.

        Simulate 1000 null experiments and check that the always-valid
        p-value rejects no more than alpha + margin of the time.
        """
        alpha = 0.05
        n_sims = 1000
        n_per_group = 200
        rejections = 0

        rng = np.random.default_rng(314)
        for _ in range(n_sims):
            ctrl = rng.normal(0, 1, size=n_per_group)
            trt = rng.normal(0, 1, size=n_per_group)
            test = mSPRT(metric="continuous", alpha=alpha, tau=0.25)
            state = test.update(ctrl, trt)
            if state.always_valid_pvalue < alpha:
                rejections += 1

        rejection_rate = rejections / n_sims
        # Allow generous tolerance for simulation noise
        assert rejection_rate <= alpha + 0.03, (
            f"Type I error rate {rejection_rate:.3f} exceeds alpha={alpha} + 0.03"
        )

    def test_always_valid_under_optional_stopping(self):
        """Under null, P(exists t: p_t < alpha) <= alpha + tolerance.

        This is the core mSPRT guarantee: the probability of EVER getting
        a significant result across all peeking times is controlled at
        the nominal level.
        """
        n_sims = 500
        alpha = 0.05
        false_positives = 0
        rng = np.random.default_rng(42)

        for _ in range(n_sims):
            test = mSPRT(metric="continuous", alpha=alpha, tau=0.25)
            ever_significant = False
            for _day in range(10):
                ctrl = rng.normal(0, 1, size=50)
                trt = rng.normal(0, 1, size=50)  # null: same distribution
                state = test.update(ctrl, trt)
                if state.should_stop:
                    ever_significant = True
                    break
            if ever_significant:
                false_positives += 1

        rejection_rate = false_positives / n_sims
        # Allow some tolerance for simulation noise
        assert rejection_rate <= alpha + 0.03, (
            f"FP rate {rejection_rate:.3f} exceeds alpha={alpha} + tolerance"
        )

    def test_ci_contains_zero_under_null(self):
        """Under the null, the CI should contain 0 most of the time."""
        n_sims = 500
        contains_zero = 0

        rng = np.random.default_rng(271)
        for _ in range(n_sims):
            ctrl = rng.normal(0, 1, size=150)
            trt = rng.normal(0, 1, size=150)
            test = mSPRT(metric="continuous", alpha=0.05, tau=0.25)
            state = test.update(ctrl, trt)
            if state.always_valid_ci_lower <= 0.0 <= state.always_valid_ci_upper:
                contains_zero += 1

        coverage = contains_zero / n_sims
        assert coverage >= 0.90, (
            f"CI coverage {coverage:.3f} is below 0.90 under the null"
        )


# ─── Streaming ────────────────────────────────────────────────────────


class TestStreaming:
    """Incremental updates accumulate correctly."""

    def test_multiple_updates_accumulate(self):
        """n_control increases with each update."""
        rng = np.random.default_rng(55)
        test = mSPRT(metric="continuous", alpha=0.05, tau=0.25)

        for i in range(1, 6):
            ctrl = rng.normal(0, 1, size=50)
            trt = rng.normal(0, 1, size=50)
            state = test.update(ctrl, trt)
            assert state.n_control == i * 50
            assert state.n_treatment == i * 50

    def test_state_is_consistent(self):
        """state.n_control matches total observations fed."""
        rng = np.random.default_rng(66)
        test = mSPRT(metric="continuous", alpha=0.05, tau=0.25)
        total_ctrl = 0
        total_trt = 0

        for batch_size in [30, 70, 100]:
            ctrl = rng.normal(0, 1, size=batch_size)
            trt = rng.normal(0, 1, size=batch_size + 10)
            total_ctrl += batch_size
            total_trt += batch_size + 10
            state = test.update(ctrl, trt)

        assert state.n_control == total_ctrl
        assert state.n_treatment == total_trt

    def test_result_after_updates(self):
        """result() returns mSPRTResult with all fields after updates."""
        rng = np.random.default_rng(77)
        test = mSPRT(metric="continuous", alpha=0.05, tau=0.25)

        ctrl = rng.normal(0, 1, size=100)
        trt = rng.normal(0, 1, size=100)
        test.update(ctrl, trt)

        res = test.result()
        assert isinstance(res, mSPRTResult)
        assert res.total_observations == 200
        assert res.stopping_reason in ("boundary_crossed", "not_stopped", "truncation")
        assert res.n_control == 100
        assert res.n_treatment == 100


# ─── Auto-tuning ──────────────────────────────────────────────────────


class TestAutoTuning:
    """tau auto-set behaviour."""

    def test_tau_auto_set(self):
        """When tau=None, first update sets tau > 0."""
        rng = np.random.default_rng(88)
        test = mSPRT(metric="continuous", alpha=0.05, tau=None)

        ctrl = rng.normal(0, 1, size=100)
        trt = rng.normal(0, 1, size=100)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            test.update(ctrl, trt)

        assert test._tau is not None
        assert test._tau > 0.0

    def test_tau_auto_set_warning(self):
        """Auto-tuning emits a RuntimeWarning with the tau value."""
        rng = np.random.default_rng(89)
        test = mSPRT(metric="continuous", alpha=0.05, tau=None)

        ctrl = rng.normal(0, 1, size=100)
        trt = rng.normal(0, 1, size=100)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            test.update(ctrl, trt)

        runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        tau_warnings = [x for x in runtime_warnings if "tau auto-set" in str(x.message)]
        assert len(tau_warnings) >= 1, "Expected a RuntimeWarning about tau auto-set"

    def test_manual_tau_used_directly(self):
        """When tau is provided, it is used without warning."""
        rng = np.random.default_rng(90)
        test = mSPRT(metric="continuous", alpha=0.05, tau=0.5)

        ctrl = rng.normal(0, 1, size=100)
        trt = rng.normal(0, 1, size=100)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            test.update(ctrl, trt)

        tau_warnings = [
            x
            for x in w
            if issubclass(x.category, RuntimeWarning)
            and "tau auto-set" in str(x.message)
        ]
        assert len(tau_warnings) == 0, "Should not warn when tau is set manually"
        assert test._tau == 0.5


# ─── Truncation ───────────────────────────────────────────────────────


class TestTruncation:
    """Truncation stops the test at the specified sample size."""

    def test_truncation_stops(self):
        """When n >= truncation, should_stop=True."""
        rng = np.random.default_rng(100)
        test = mSPRT(metric="continuous", alpha=0.05, tau=0.25, truncation=200)

        ctrl = rng.normal(0, 1, size=100)
        trt = rng.normal(0, 1, size=100)
        state = test.update(ctrl, trt)

        assert state.should_stop is True

    def test_truncation_stopping_reason(self):
        """Truncated test result shows 'truncation' as stopping reason."""
        rng = np.random.default_rng(101)
        # Use no-effect data so the test wouldn't stop on its own
        test = mSPRT(metric="continuous", alpha=0.05, tau=0.25, truncation=200)

        ctrl = rng.normal(0, 1, size=100)
        trt = rng.normal(0, 1, size=100)
        test.update(ctrl, trt)
        res = test.result()

        # If p >= alpha but we hit truncation, reason is "truncation"
        if res.always_valid_pvalue >= 0.05:
            assert res.stopping_reason == "truncation"
        else:
            # Effect was significant anyway
            assert res.stopping_reason == "boundary_crossed"


# ─── Validation ───────────────────────────────────────────────────────


class TestValidation:
    """Input validation raises appropriate errors."""

    def test_invalid_metric(self):
        with pytest.raises(ValueError, match="metric"):
            mSPRT(metric="revenue")

    def test_invalid_alpha_too_low(self):
        with pytest.raises(ValueError, match="alpha"):
            mSPRT(alpha=0.0)

    def test_invalid_alpha_too_high(self):
        with pytest.raises(ValueError, match="alpha"):
            mSPRT(alpha=1.0)

    def test_invalid_tau_zero(self):
        with pytest.raises(ValueError, match="tau"):
            mSPRT(tau=0.0)

    def test_invalid_tau_negative(self):
        with pytest.raises(ValueError, match="tau"):
            mSPRT(tau=-0.5)

    def test_invalid_truncation_zero(self):
        with pytest.raises(ValueError, match="truncation"):
            mSPRT(truncation=0)

    def test_invalid_truncation_negative(self):
        with pytest.raises(ValueError, match="truncation"):
            mSPRT(truncation=-10)

    def test_update_both_empty_first_call(self):
        """Both arrays empty on first call raises ValueError."""
        test = mSPRT(metric="continuous", alpha=0.05, tau=0.25)
        with pytest.raises(ValueError, match="can't both be empty"):
            test.update([], [])

    def test_update_empty_after_data_is_noop(self):
        """Empty arrays after initial data are allowed (no-op)."""
        rng = np.random.default_rng(200)
        test = mSPRT(metric="continuous", alpha=0.05, tau=0.25)
        ctrl = rng.normal(0, 1, size=50)
        trt = rng.normal(0, 1, size=50)
        state1 = test.update(ctrl, trt)
        state2 = test.update([], [])
        assert state2.n_control == state1.n_control


# ─── Conversion metric ────────────────────────────────────────────────


class TestConversion:
    """Conversion (Bernoulli) metric tests."""

    def test_binary_data_works(self):
        """Conversion metric with 0/1 data works."""
        rng = np.random.default_rng(300)
        ctrl = rng.binomial(1, 0.10, size=500).astype(float)
        trt = rng.binomial(1, 0.10, size=500).astype(float)

        test = mSPRT(metric="conversion", alpha=0.05, tau=0.01)
        state = test.update(ctrl, trt)

        assert isinstance(state, mSPRTState)
        assert 0.0 <= state.always_valid_pvalue <= 1.0
        assert state.mixture_lr >= 0.0

    def test_significant_conversion(self):
        """Large conversion difference stops the test."""
        rng = np.random.default_rng(301)
        ctrl = rng.binomial(1, 0.05, size=1000).astype(float)
        trt = rng.binomial(1, 0.15, size=1000).astype(float)

        test = mSPRT(metric="conversion", alpha=0.05, tau=0.01)
        state = test.update(ctrl, trt)

        assert state.should_stop is True
        assert state.always_valid_pvalue < 0.05


# ─── Edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    """Boundary and degenerate scenarios."""

    def test_very_small_effect_does_not_stop(self):
        """Very small effect should not trigger early stopping."""
        rng = np.random.default_rng(400)
        ctrl = rng.normal(0.0, 1.0, size=100)
        trt = rng.normal(0.01, 1.0, size=100)

        test = mSPRT(metric="continuous", alpha=0.05, tau=0.25)
        state = test.update(ctrl, trt)

        assert state.should_stop is False

    def test_result_before_update_raises(self):
        """result() before any update raises RuntimeError."""
        test = mSPRT(metric="continuous", alpha=0.05)
        with pytest.raises(RuntimeError, match="can't produce a result"):
            test.result()

    def test_to_dict_works(self):
        """State and result to_dict produce plain dictionaries."""
        rng = np.random.default_rng(500)
        test = mSPRT(metric="continuous", alpha=0.05, tau=0.25)
        ctrl = rng.normal(0, 1, size=100)
        trt = rng.normal(0, 1, size=100)
        state = test.update(ctrl, trt)
        res = test.result()

        assert isinstance(state.to_dict(), dict)
        assert isinstance(res.to_dict(), dict)

    def test_repr_does_not_crash(self):
        """repr on state and result does not raise."""
        rng = np.random.default_rng(501)
        test = mSPRT(metric="continuous", alpha=0.05, tau=0.25)
        ctrl = rng.normal(0, 1, size=50)
        trt = rng.normal(0, 1, size=50)
        state = test.update(ctrl, trt)
        res = test.result()

        assert "mSPRTState" in repr(state)
        assert "mSPRTResult" in repr(res)

    def test_one_obs_per_group(self):
        """Single observation per group does not crash."""
        test = mSPRT(metric="continuous", alpha=0.05, tau=0.25)
        state = test.update([1.0], [2.0])
        assert state.n_control == 1
        assert state.n_treatment == 1
        assert 0.0 <= state.always_valid_pvalue <= 1.0


# ─── Coverage gap tests ──────────────────────────────────────────────


class TestCoverageGaps:
    """Tests targeting uncovered lines."""

    def test_truncation_invalid_type(self):
        """Non-numeric truncation should raise TypeError."""
        with pytest.raises(TypeError, match="truncation"):
            mSPRT(truncation="abc")

    def test_one_sided_control_only(self):
        """First update with only control data (no treatment) returns valid state."""
        test = mSPRT(metric="continuous", alpha=0.05, tau=0.25)
        state = test.update([1.0, 2.0, 3.0], np.array([], dtype=float))
        assert state.n_control == 3
        assert state.n_treatment == 0
        assert state.should_stop is False
        assert state.always_valid_pvalue == 1.0

    def test_boundary_crossed_stopping_reason(self):
        """When test stops due to significance, stopping_reason is boundary_crossed."""
        rng = np.random.default_rng(9999)
        test = mSPRT(metric="continuous", alpha=0.05, tau=0.25)
        ctrl = rng.normal(0.0, 1.0, size=500)
        trt = rng.normal(0.5, 1.0, size=500)
        test.update(ctrl, trt)
        res = test.result()
        assert res.should_stop is True
        assert res.stopping_reason == "boundary_crossed"
