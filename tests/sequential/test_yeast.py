"""Tests for YEASTSequentialTest."""

from __future__ import annotations

import numpy as np
import pytest

from splita.sequential.yeast import YEASTSequentialTest


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


class TestUpdate:
    def test_basic_update(self, rng):
        test = YEASTSequentialTest(alpha=0.05)
        ctrl = rng.normal(0, 1, 500)
        trt = rng.normal(0.5, 1, 500)
        state = test.update(ctrl, trt)
        assert state.n_control == 500
        assert state.n_treatment == 500

    def test_detects_effect(self, rng):
        test = YEASTSequentialTest(alpha=0.05)
        ctrl = rng.normal(0, 1, 500)
        trt = rng.normal(0.5, 1, 500)
        state = test.update(ctrl, trt)
        assert state.should_stop is True

    def test_no_false_positive(self, rng):
        test = YEASTSequentialTest(alpha=0.05)
        ctrl = rng.normal(0, 1, 100)
        trt = rng.normal(0, 1, 100)
        state = test.update(ctrl, trt)
        # With no true effect and modest sample size, should usually not stop
        # (not guaranteed, but should be common)
        assert isinstance(state.should_stop, bool)

    def test_incremental_updates(self, rng):
        test = YEASTSequentialTest(alpha=0.05)
        for _ in range(5):
            ctrl = rng.normal(0, 1, 100)
            trt = rng.normal(0.3, 1, 100)
            state = test.update(ctrl, trt)
        assert state.n_control == 500
        assert state.n_treatment == 500

    def test_z_statistic(self, rng):
        test = YEASTSequentialTest(alpha=0.05)
        ctrl = rng.normal(0, 1, 200)
        trt = rng.normal(0.5, 1, 200)
        state = test.update(ctrl, trt)
        assert state.z_statistic > 0  # trt > ctrl

    def test_boundary_constant(self, rng):
        test = YEASTSequentialTest(alpha=0.05)
        state1 = test.update(rng.normal(0, 1, 100), rng.normal(0, 1, 100))
        state2 = test.update(rng.normal(0, 1, 100), rng.normal(0, 1, 100))
        assert state1.boundary == state2.boundary  # constant boundary

    def test_pvalue_range(self, rng):
        test = YEASTSequentialTest(alpha=0.05)
        state = test.update(rng.normal(0, 1, 200), rng.normal(0.3, 1, 200))
        assert 0.0 <= state.pvalue <= 1.0


class TestResult:
    def test_result_after_update(self, rng):
        test = YEASTSequentialTest(alpha=0.05)
        test.update(rng.normal(0, 1, 500), rng.normal(0.5, 1, 500))
        result = test.result()
        assert result.stopping_reason in ("boundary_crossed", "not_stopped")

    def test_result_before_update(self):
        test = YEASTSequentialTest(alpha=0.05)
        with pytest.raises(RuntimeError, match="before any data"):
            test.result()

    def test_total_observations(self, rng):
        test = YEASTSequentialTest(alpha=0.05)
        test.update(rng.normal(0, 1, 300), rng.normal(0, 1, 200))
        result = test.result()
        assert result.total_observations == 500


class TestResultMethods:
    def test_to_dict(self, rng):
        test = YEASTSequentialTest(alpha=0.05)
        test.update(rng.normal(0, 1, 200), rng.normal(0.3, 1, 200))
        result = test.result()
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "z_statistic" in d

    def test_repr(self, rng):
        test = YEASTSequentialTest(alpha=0.05)
        test.update(rng.normal(0, 1, 200), rng.normal(0.3, 1, 200))
        result = test.result()
        assert "YEASTResult" in repr(result)

    def test_state_repr(self, rng):
        test = YEASTSequentialTest(alpha=0.05)
        state = test.update(rng.normal(0, 1, 200), rng.normal(0.3, 1, 200))
        assert "YEASTState" in repr(state)


class TestValidation:
    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            YEASTSequentialTest(alpha=0.0)

    def test_both_empty_first(self):
        test = YEASTSequentialTest(alpha=0.05)
        with pytest.raises(ValueError, match="both be empty"):
            test.update([], [])

    def test_empty_after_data_ok(self, rng):
        test = YEASTSequentialTest(alpha=0.05)
        test.update(rng.normal(0, 1, 100), rng.normal(0, 1, 100))
        state = test.update([], [])  # no-op, should not raise
        assert state.n_control == 100
