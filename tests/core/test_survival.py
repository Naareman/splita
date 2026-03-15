"""Tests for SurvivalExperiment (time-to-event analysis)."""

from __future__ import annotations

import numpy as np
import pytest

from splita.core.survival import SurvivalExperiment


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestSurvivalBasic:
    """Basic functionality tests."""

    def test_significant_difference(self, rng):
        """Groups with different survival should be significant."""
        time_ctrl = rng.exponential(5, 200)
        event_ctrl = np.ones(200)
        time_trt = rng.exponential(15, 200)
        event_trt = np.ones(200)

        r = SurvivalExperiment().fit(
            time_ctrl, event_ctrl, time_trt, event_trt
        ).result()

        assert r.significant is True
        assert r.logrank_pvalue < 0.05

    def test_no_difference(self, rng):
        """Same survival distributions should not be significant."""
        time_ctrl = rng.exponential(10, 200)
        event_ctrl = np.ones(200)
        time_trt = rng.exponential(10, 200)
        event_trt = np.ones(200)

        r = SurvivalExperiment().fit(
            time_ctrl, event_ctrl, time_trt, event_trt
        ).result()

        assert r.logrank_pvalue > 0.01

    def test_hazard_ratio_direction(self, rng):
        """Treatment with longer survival should have HR != 1."""
        time_ctrl = rng.exponential(5, 300)
        event_ctrl = np.ones(300)
        time_trt = rng.exponential(20, 300)
        event_trt = np.ones(300)

        r = SurvivalExperiment().fit(
            time_ctrl, event_ctrl, time_trt, event_trt
        ).result()

        assert r.hazard_ratio > 0

    def test_median_survival(self, rng):
        """Median survival should be estimable when enough events occur."""
        time_ctrl = rng.exponential(5, 200)
        event_ctrl = np.ones(200)
        time_trt = rng.exponential(10, 200)
        event_trt = np.ones(200)

        r = SurvivalExperiment().fit(
            time_ctrl, event_ctrl, time_trt, event_trt
        ).result()

        assert r.median_survival_ctrl is not None
        assert r.median_survival_trt is not None
        assert r.median_survival_trt > r.median_survival_ctrl

    def test_censored_data(self, rng):
        """Should handle censored observations correctly."""
        time_ctrl = rng.exponential(10, 200)
        event_ctrl = rng.binomial(1, 0.6, 200).astype(float)
        time_trt = rng.exponential(10, 200)
        event_trt = rng.binomial(1, 0.6, 200).astype(float)

        r = SurvivalExperiment().fit(
            time_ctrl, event_ctrl, time_trt, event_trt
        ).result()

        assert r.n_events_ctrl < r.n_ctrl
        assert r.n_events_trt < r.n_trt

    def test_counts_correct(self, rng):
        """n_ctrl and n_trt should match input sizes."""
        n_c, n_t = 100, 150
        r = SurvivalExperiment().fit(
            rng.exponential(10, n_c), np.ones(n_c),
            rng.exponential(10, n_t), np.ones(n_t),
        ).result()

        assert r.n_ctrl == n_c
        assert r.n_trt == n_t

    def test_ci_contains_hr(self, rng):
        """CI should contain the hazard ratio."""
        time_ctrl = rng.exponential(10, 300)
        event_ctrl = np.ones(300)
        time_trt = rng.exponential(10, 300)
        event_trt = np.ones(300)

        r = SurvivalExperiment().fit(
            time_ctrl, event_ctrl, time_trt, event_trt
        ).result()

        assert r.ci_lower <= r.hazard_ratio <= r.ci_upper


class TestSurvivalValidation:
    """Validation and error handling tests."""

    def test_negative_times_rejected(self, rng):
        """Negative survival times should raise ValueError."""
        time_ctrl = np.array([-1.0, 2.0, 3.0])
        event_ctrl = np.array([1.0, 1.0, 1.0])

        with pytest.raises(ValueError, match="non-negative"):
            SurvivalExperiment().fit(
                time_ctrl, event_ctrl,
                rng.exponential(10, 3), np.ones(3),
            )

    def test_invalid_event_values(self, rng):
        """Event indicators must be 0 or 1."""
        with pytest.raises(ValueError, match="0 and 1"):
            SurvivalExperiment().fit(
                rng.exponential(10, 5), np.array([0.0, 1.0, 2.0, 1.0, 0.0]),
                rng.exponential(10, 5), np.ones(5),
            )

    def test_mismatched_lengths(self, rng):
        """Time and event arrays must have the same length."""
        with pytest.raises(ValueError, match="same length"):
            SurvivalExperiment().fit(
                rng.exponential(10, 5), np.ones(3),
                rng.exponential(10, 5), np.ones(5),
            )

    def test_result_before_fit(self):
        """Calling result() before fit() should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="fitted"):
            SurvivalExperiment().result()

    def test_invalid_alpha(self):
        """Alpha outside (0, 1) should raise ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            SurvivalExperiment(alpha=0.0)

    def test_to_dict(self, rng):
        """to_dict should return a plain dictionary."""
        r = SurvivalExperiment().fit(
            rng.exponential(10, 50), np.ones(50),
            rng.exponential(10, 50), np.ones(50),
        ).result()

        d = r.to_dict()
        assert isinstance(d, dict)
        assert "hazard_ratio" in d
        assert "logrank_pvalue" in d

    def test_repr(self, rng):
        """repr should return a formatted string."""
        r = SurvivalExperiment().fit(
            rng.exponential(10, 50), np.ones(50),
            rng.exponential(10, 50), np.ones(50),
        ).result()

        s = repr(r)
        assert "SurvivalResult" in s

    def test_too_few_observations(self):
        """Should reject arrays with fewer than 2 elements."""
        with pytest.raises(ValueError, match="at least 2"):
            SurvivalExperiment().fit(
                [1.0], [1.0], [1.0, 2.0], [1.0, 1.0],
            )

    def test_chaining(self, rng):
        """fit() should return self for chaining."""
        exp = SurvivalExperiment()
        ret = exp.fit(
            rng.exponential(10, 20), np.ones(20),
            rng.exponential(10, 20), np.ones(20),
        )
        assert ret is exp
