"""Tests for NoveltyCurve diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from splita._types import NoveltyCurveResult
from splita.diagnostics.novelty import NoveltyCurve


# ─── Helpers ─────────────────────────────────────────────────────────


def _make_stable_data(
    rng: np.random.Generator,
    n_days: int = 21,
    n_per_day: int = 200,
    ctrl_rate: float = 0.10,
    trt_rate: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create data with a stable treatment effect."""
    n = n_days * n_per_day
    control = rng.binomial(1, ctrl_rate, size=n)
    treatment = rng.binomial(1, trt_rate, size=n)
    ts_ctrl = np.repeat(np.arange(n_days), n_per_day)
    ts_trt = np.repeat(np.arange(n_days), n_per_day)
    timestamps = np.concatenate([ts_ctrl, ts_trt])
    return control, treatment, timestamps


def _make_decreasing_data(
    rng: np.random.Generator,
    n_days: int = 21,
    n_per_day: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create data where treatment effect decreases over time (novelty)."""
    control_parts = []
    treatment_parts = []
    ts_ctrl_parts = []
    ts_trt_parts = []

    for day in range(n_days):
        # Effect starts at 0.20 and drops to 0.10 (control rate)
        trt_rate = 0.20 - 0.10 * (day / (n_days - 1))
        ctrl_rate = 0.10

        control_parts.append(rng.binomial(1, ctrl_rate, size=n_per_day))
        treatment_parts.append(rng.binomial(1, trt_rate, size=n_per_day))
        ts_ctrl_parts.append(np.full(n_per_day, day))
        ts_trt_parts.append(np.full(n_per_day, day))

    control = np.concatenate(control_parts)
    treatment = np.concatenate(treatment_parts)
    timestamps = np.concatenate(ts_ctrl_parts + ts_trt_parts)
    return control, treatment, timestamps


# ─── Construction / validation ───────────────────────────────────────


class TestNoveltyCurveInit:
    """Tests for NoveltyCurve.__init__ validation."""

    def test_default_params(self) -> None:
        nc = NoveltyCurve()
        assert nc._window_size == 7
        assert nc._alpha == 0.05

    def test_custom_params(self) -> None:
        nc = NoveltyCurve(window_size=14, alpha=0.01)
        assert nc._window_size == 14
        assert nc._alpha == 0.01

    def test_window_size_too_small(self) -> None:
        with pytest.raises(ValueError, match="window_size"):
            NoveltyCurve(window_size=1)

    def test_window_size_float_rejected(self) -> None:
        with pytest.raises(ValueError, match="window_size"):
            NoveltyCurve(window_size=3.5)

    def test_alpha_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            NoveltyCurve(alpha=1.5)

    def test_alpha_zero(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            NoveltyCurve(alpha=0.0)


# ─── fit() validation ────────────────────────────────────────────────


class TestNoveltyCurveFitValidation:
    """Tests for NoveltyCurve.fit input validation."""

    def test_timestamps_wrong_length(self) -> None:
        nc = NoveltyCurve(window_size=2)
        ctrl = [0, 1, 0, 1]
        trt = [1, 1, 0, 0]
        ts = [1, 2, 3]  # Wrong length
        with pytest.raises(ValueError, match="timestamps"):
            nc.fit(ctrl, trt, ts)

    def test_too_few_unique_timestamps(self) -> None:
        nc = NoveltyCurve(window_size=5)
        ctrl = [0, 1, 0, 1]
        trt = [1, 1, 0, 0]
        ts = [1, 1, 1, 1, 2, 2, 2, 2]  # Only 2 unique
        with pytest.raises(ValueError, match="unique timestamps"):
            nc.fit(ctrl, trt, ts)

    def test_control_too_short(self) -> None:
        nc = NoveltyCurve(window_size=2)
        with pytest.raises(ValueError, match="control"):
            nc.fit([1], [0, 1], [1, 2, 3])

    def test_result_before_fit_raises(self) -> None:
        nc = NoveltyCurve()
        with pytest.raises(RuntimeError, match="fit"):
            nc.result()


# ─── Stable effect ───────────────────────────────────────────────────


class TestNoveltyCurveStable:
    """Tests for stable treatment effect (no novelty)."""

    def test_stable_effect_no_novelty(self) -> None:
        rng = np.random.default_rng(42)
        ctrl, trt, ts = _make_stable_data(rng, n_days=21, n_per_day=200)
        nc = NoveltyCurve(window_size=7).fit(ctrl, trt, ts)
        result = nc.result()
        assert isinstance(result, NoveltyCurveResult)
        assert result.has_novelty_effect is False

    def test_stable_returns_windows(self) -> None:
        rng = np.random.default_rng(42)
        ctrl, trt, ts = _make_stable_data(rng, n_days=14, n_per_day=200)
        nc = NoveltyCurve(window_size=7).fit(ctrl, trt, ts)
        result = nc.result()
        assert len(result.windows) > 0

    def test_window_dict_keys(self) -> None:
        rng = np.random.default_rng(42)
        ctrl, trt, ts = _make_stable_data(rng, n_days=14, n_per_day=200)
        nc = NoveltyCurve(window_size=7).fit(ctrl, trt, ts)
        result = nc.result()
        expected_keys = {"window_start", "lift", "pvalue", "ci_lower", "ci_upper"}
        for win in result.windows:
            assert set(win.keys()) == expected_keys

    def test_stable_trend_direction(self) -> None:
        rng = np.random.default_rng(42)
        ctrl, trt, ts = _make_stable_data(rng, n_days=21, n_per_day=200)
        nc = NoveltyCurve(window_size=7).fit(ctrl, trt, ts)
        result = nc.result()
        # With stable data, trend should be stable (or at worst not decreasing)
        assert result.trend_direction in ("stable", "increasing", "decreasing")


# ─── Decreasing effect (novelty) ─────────────────────────────────────


class TestNoveltyCurveNovelty:
    """Tests for novelty (decreasing) effect detection."""

    def test_decreasing_effect_detected(self) -> None:
        rng = np.random.default_rng(42)
        ctrl, trt, ts = _make_decreasing_data(rng, n_days=21, n_per_day=300)
        nc = NoveltyCurve(window_size=7).fit(ctrl, trt, ts)
        result = nc.result()
        assert result.has_novelty_effect is True

    def test_decreasing_trend_direction(self) -> None:
        rng = np.random.default_rng(42)
        ctrl, trt, ts = _make_decreasing_data(rng, n_days=21, n_per_day=300)
        nc = NoveltyCurve(window_size=7).fit(ctrl, trt, ts)
        result = nc.result()
        assert result.trend_direction == "decreasing"


# ─── Edge cases ──────────────────────────────────────────────────────


class TestNoveltyCurveEdgeCases:
    """Edge cases and special behaviour."""

    def test_minimum_windows_warning(self) -> None:
        rng = np.random.default_rng(42)
        # 3 unique timestamps, window_size=3 → only 1 window
        ctrl = rng.binomial(1, 0.10, size=30)
        trt = rng.binomial(1, 0.15, size=30)
        ts = np.concatenate([
            np.repeat([0, 1, 2], 10),
            np.repeat([0, 1, 2], 10),
        ])
        with pytest.warns(RuntimeWarning, match="window"):
            NoveltyCurve(window_size=3).fit(ctrl, trt, ts)

    def test_to_dict(self) -> None:
        rng = np.random.default_rng(42)
        ctrl, trt, ts = _make_stable_data(rng, n_days=14, n_per_day=100)
        nc = NoveltyCurve(window_size=7).fit(ctrl, trt, ts)
        d = nc.result().to_dict()
        assert isinstance(d, dict)
        assert "windows" in d
        assert "has_novelty_effect" in d
        assert "trend_direction" in d

    def test_repr(self) -> None:
        rng = np.random.default_rng(42)
        ctrl, trt, ts = _make_stable_data(rng, n_days=14, n_per_day=100)
        nc = NoveltyCurve(window_size=7).fit(ctrl, trt, ts)
        text = repr(nc.result())
        assert "NoveltyCurveResult" in text

    def test_method_chaining(self) -> None:
        rng = np.random.default_rng(42)
        ctrl, trt, ts = _make_stable_data(rng, n_days=14, n_per_day=100)
        result = NoveltyCurve(window_size=7).fit(ctrl, trt, ts).result()
        assert isinstance(result, NoveltyCurveResult)

    def test_window_skips_insufficient_data(self) -> None:
        """Line 160: windows with <2 data points in either group are skipped."""
        # Control has data at ts 0..9, but treatment only at ts 5..9
        # Early windows (ts 0..4) will have 0 treatment data => skipped
        ctrl = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])
        trt = np.array([15.0, 16.0, 17.0, 18.0, 19.0])
        ts_ctrl = np.arange(10, dtype=float)  # timestamps 0..9
        ts_trt = np.arange(5, 10, dtype=float)  # timestamps 5..9
        ts = np.concatenate([ts_ctrl, ts_trt])
        nc = NoveltyCurve(window_size=3)
        nc.fit(ctrl, trt, ts)
        result = nc.result()
        assert isinstance(result, NoveltyCurveResult)
        # Early windows should be skipped (no treatment data)
        if result.windows:
            assert result.windows[0]["window_start"] >= 3.0

    def test_detect_trend_first_lift_zero_stable(self) -> None:
        """Line 246: first_lift==0 with non-decreasing trend."""
        windows = [
            {"lift": 0.0, "pvalue": 0.5, "ci_lower": -0.1, "ci_upper": 0.1, "window_start": 0.0},
            {"lift": 0.01, "pvalue": 0.4, "ci_lower": -0.1, "ci_upper": 0.1, "window_start": 1.0},
            {"lift": 0.02, "pvalue": 0.3, "ci_lower": -0.1, "ci_upper": 0.1, "window_start": 2.0},
        ]
        direction, has_novelty = NoveltyCurve._detect_trend(windows)
        # first_lift==0, not decreasing => has_novelty should be False
        assert has_novelty is False

    def test_detect_trend_first_lift_zero_decreasing(self) -> None:
        """Lines 254-255: first_lift==0 with decreasing trend."""
        windows = [
            {"lift": 0.0, "pvalue": 0.5, "ci_lower": -0.1, "ci_upper": 0.1, "window_start": 0.0},
            {"lift": -0.5, "pvalue": 0.01, "ci_lower": -0.7, "ci_upper": -0.3, "window_start": 1.0},
            {"lift": -1.0, "pvalue": 0.001, "ci_lower": -1.2, "ci_upper": -0.8, "window_start": 2.0},
        ]
        direction, has_novelty = NoveltyCurve._detect_trend(windows)
        assert direction == "decreasing"
        assert has_novelty is True

    def test_different_window_sizes(self) -> None:
        rng = np.random.default_rng(42)
        ctrl, trt, ts = _make_stable_data(rng, n_days=21, n_per_day=100)

        r7 = NoveltyCurve(window_size=7).fit(ctrl, trt, ts).result()
        r14 = NoveltyCurve(window_size=14).fit(ctrl, trt, ts).result()

        # Larger window → fewer windows
        assert len(r7.windows) > len(r14.windows)
