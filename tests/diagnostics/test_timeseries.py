"""Tests for EffectTimeSeries diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from splita._types import EffectTimeSeriesResult
from splita.core.experiment import Experiment
from splita.diagnostics.effect_timeseries import EffectTimeSeries


# ─── Helpers ─────────────────────────────────────────────────────────


def _make_stable_ts_data(
    rng: np.random.Generator,
    n_days: int = 10,
    n_per_day: int = 200,
    ctrl_rate: float = 0.10,
    trt_rate: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create data with a stable treatment effect over time."""
    n = n_days * n_per_day
    control = rng.binomial(1, ctrl_rate, size=n)
    treatment = rng.binomial(1, trt_rate, size=n)
    ts_ctrl = np.repeat(np.arange(n_days), n_per_day)
    ts_trt = np.repeat(np.arange(n_days), n_per_day)
    timestamps = np.concatenate([ts_ctrl, ts_trt])
    return control, treatment, timestamps


def _make_growing_ts_data(
    rng: np.random.Generator,
    n_days: int = 10,
    n_per_day: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create data where treatment effect grows over time."""
    control_parts = []
    treatment_parts = []
    ts_ctrl_parts = []
    ts_trt_parts = []

    for day in range(n_days):
        ctrl_rate = 0.10
        # Effect grows from 0.10 (no effect) to 0.30
        trt_rate = 0.10 + 0.20 * (day / (n_days - 1))

        control_parts.append(rng.binomial(1, ctrl_rate, size=n_per_day))
        treatment_parts.append(rng.binomial(1, trt_rate, size=n_per_day))
        ts_ctrl_parts.append(np.full(n_per_day, day))
        ts_trt_parts.append(np.full(n_per_day, day))

    control = np.concatenate(control_parts)
    treatment = np.concatenate(treatment_parts)
    timestamps = np.concatenate(ts_ctrl_parts + ts_trt_parts)
    return control, treatment, timestamps


# ─── Construction / validation ───────────────────────────────────────


class TestEffectTimeSeriesInit:
    """Tests for EffectTimeSeries.__init__ validation."""

    def test_default_params(self) -> None:
        ets = EffectTimeSeries()
        assert ets._alpha == 0.05

    def test_custom_alpha(self) -> None:
        ets = EffectTimeSeries(alpha=0.01)
        assert ets._alpha == 0.01

    def test_alpha_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            EffectTimeSeries(alpha=1.5)

    def test_alpha_zero(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            EffectTimeSeries(alpha=0.0)


# ─── fit() validation ────────────────────────────────────────────────


class TestEffectTimeSeriesFitValidation:
    """Tests for EffectTimeSeries.fit input validation."""

    def test_timestamps_wrong_length(self) -> None:
        ets = EffectTimeSeries()
        ctrl = [0, 1, 0, 1]
        trt = [1, 1, 0, 0]
        ts = [1, 2, 3]  # Wrong length
        with pytest.raises(ValueError, match="timestamps"):
            ets.fit(ctrl, trt, ts)

    def test_control_too_short(self) -> None:
        ets = EffectTimeSeries()
        with pytest.raises(ValueError, match="control"):
            ets.fit([1], [0, 1], [1, 2, 3])

    def test_result_before_fit_raises(self) -> None:
        ets = EffectTimeSeries()
        with pytest.raises(RuntimeError, match="fit"):
            ets.result()


# ─── Stable effect ───────────────────────────────────────────────────


class TestEffectTimeSeriesStable:
    """Tests for stable treatment effect."""

    def test_stable_effect_is_stable(self) -> None:
        rng = np.random.default_rng(42)
        ctrl, trt, ts = _make_stable_ts_data(rng, n_days=10, n_per_day=300)
        ets = EffectTimeSeries().fit(ctrl, trt, ts)
        result = ets.result()
        assert isinstance(result, EffectTimeSeriesResult)
        assert result.is_stable is True

    def test_time_points_returned(self) -> None:
        rng = np.random.default_rng(42)
        ctrl, trt, ts = _make_stable_ts_data(rng, n_days=10, n_per_day=200)
        result = EffectTimeSeries().fit(ctrl, trt, ts).result()
        assert len(result.time_points) > 0

    def test_time_point_dict_keys(self) -> None:
        rng = np.random.default_rng(42)
        ctrl, trt, ts = _make_stable_ts_data(rng, n_days=10, n_per_day=200)
        result = EffectTimeSeries().fit(ctrl, trt, ts).result()
        expected_keys = {
            "timestamp",
            "cumulative_lift",
            "pvalue",
            "ci_lower",
            "ci_upper",
            "n_control",
            "n_treatment",
        }
        for tp in result.time_points:
            assert set(tp.keys()) == expected_keys

    def test_cumulative_n_increases(self) -> None:
        rng = np.random.default_rng(42)
        ctrl, trt, ts = _make_stable_ts_data(rng, n_days=10, n_per_day=200)
        result = EffectTimeSeries().fit(ctrl, trt, ts).result()
        n_controls = [tp["n_control"] for tp in result.time_points]
        # Cumulative counts should be non-decreasing
        for i in range(1, len(n_controls)):
            assert n_controls[i] >= n_controls[i - 1]

    def test_final_lift_matches_last_timepoint(self) -> None:
        rng = np.random.default_rng(42)
        ctrl, trt, ts = _make_stable_ts_data(rng, n_days=10, n_per_day=200)
        result = EffectTimeSeries().fit(ctrl, trt, ts).result()
        assert result.final_lift == result.time_points[-1]["cumulative_lift"]

    def test_final_pvalue_matches_last_timepoint(self) -> None:
        rng = np.random.default_rng(42)
        ctrl, trt, ts = _make_stable_ts_data(rng, n_days=10, n_per_day=200)
        result = EffectTimeSeries().fit(ctrl, trt, ts).result()
        assert result.final_pvalue == result.time_points[-1]["pvalue"]


# ─── Growing effect ─────────────────────────────────────────────────


class TestEffectTimeSeriesGrowing:
    """Tests for growing treatment effect."""

    def test_growing_effect_detected(self) -> None:
        rng = np.random.default_rng(42)
        ctrl, trt, ts = _make_growing_ts_data(rng, n_days=10, n_per_day=300)
        result = EffectTimeSeries().fit(ctrl, trt, ts).result()
        assert isinstance(result, EffectTimeSeriesResult)
        # Growing effect: final lift should be positive
        assert result.final_lift > 0

    def test_growing_last_timepoint_has_more_data(self) -> None:
        rng = np.random.default_rng(42)
        ctrl, trt, ts = _make_growing_ts_data(rng, n_days=10, n_per_day=200)
        result = EffectTimeSeries().fit(ctrl, trt, ts).result()
        first_n = result.time_points[0]["n_control"]
        last_n = result.time_points[-1]["n_control"]
        assert last_n > first_n


# ─── Correctness: compare against manual Experiment ──────────────────


class TestEffectTimeSeriesCorrectness:
    """Compare cumulative stats against manually running Experiment."""

    def test_final_matches_full_experiment(self) -> None:
        rng = np.random.default_rng(42)
        ctrl, trt, ts = _make_stable_ts_data(rng, n_days=5, n_per_day=200)
        result = EffectTimeSeries().fit(ctrl, trt, ts).result()

        # Run Experiment on all data
        full_exp = Experiment(ctrl, trt).run()

        # Final cumulative lift should match full Experiment lift
        assert abs(result.final_lift - full_exp.lift) < 1e-10

    def test_intermediate_point_matches_manual(self) -> None:
        rng = np.random.default_rng(42)
        n_days = 5
        n_per_day = 200
        ctrl, trt, ts = _make_stable_ts_data(
            rng, n_days=n_days, n_per_day=n_per_day
        )
        result = EffectTimeSeries().fit(ctrl, trt, ts).result()

        # Check the time point at day 2 (index 2 in unique timestamps)
        # This should use data from days 0, 1, 2
        tp_day2 = result.time_points[2]
        n_cumulative = 3 * n_per_day  # days 0, 1, 2
        assert tp_day2["n_control"] == n_cumulative
        assert tp_day2["n_treatment"] == n_cumulative

        # Manually run Experiment on first 3 days
        ctrl_3d = ctrl[: n_cumulative]
        trt_3d = trt[: n_cumulative]
        manual_exp = Experiment(ctrl_3d, trt_3d).run()
        assert abs(tp_day2["cumulative_lift"] - manual_exp.lift) < 1e-10


# ─── Edge cases ──────────────────────────────────────────────────────


class TestEffectTimeSeriesEdgeCases:
    """Edge cases and special behaviour."""

    def test_to_dict(self) -> None:
        rng = np.random.default_rng(42)
        ctrl, trt, ts = _make_stable_ts_data(rng, n_days=5, n_per_day=100)
        result = EffectTimeSeries().fit(ctrl, trt, ts).result()
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "time_points" in d
        assert "is_stable" in d
        assert "final_lift" in d

    def test_repr(self) -> None:
        rng = np.random.default_rng(42)
        ctrl, trt, ts = _make_stable_ts_data(rng, n_days=5, n_per_day=100)
        result = EffectTimeSeries().fit(ctrl, trt, ts).result()
        text = repr(result)
        assert "EffectTimeSeriesResult" in text

    def test_method_chaining(self) -> None:
        rng = np.random.default_rng(42)
        ctrl, trt, ts = _make_stable_ts_data(rng, n_days=5, n_per_day=100)
        result = EffectTimeSeries().fit(ctrl, trt, ts).result()
        assert isinstance(result, EffectTimeSeriesResult)

    def test_frozen_dataclass(self) -> None:
        rng = np.random.default_rng(42)
        ctrl, trt, ts = _make_stable_ts_data(rng, n_days=5, n_per_day=100)
        result = EffectTimeSeries().fit(ctrl, trt, ts).result()
        with pytest.raises(AttributeError):
            result.is_stable = False  # type: ignore[misc]

    def test_timestamps_as_floats(self) -> None:
        rng = np.random.default_rng(42)
        ctrl = rng.binomial(1, 0.10, size=200)
        trt = rng.binomial(1, 0.15, size=200)
        ts = np.concatenate([
            np.repeat([1.0, 2.0, 3.0, 4.0, 5.0], 40),
            np.repeat([1.0, 2.0, 3.0, 4.0, 5.0], 40),
        ])
        result = EffectTimeSeries().fit(ctrl, trt, ts).result()
        assert len(result.time_points) > 0
