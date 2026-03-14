"""Tests for NonStationaryDetector (Tier 3, Item 11)."""

from __future__ import annotations

import numpy as np
import pytest

from splita._types import NonStationaryResult
from splita.diagnostics.nonstationarity import NonStationaryDetector


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestNonStationaryBasic:
    """Basic functionality tests."""

    def test_stable_data_is_stationary(self, rng):
        """Constant treatment effect should be detected as stationary."""
        n = 500
        timestamps = np.arange(n)
        # Use same base noise, just shift treatment by a constant
        base = rng.normal(10, 0.5, n)
        control = base.copy()
        treatment = base + 1.0  # exact constant +1 effect (no extra noise)

        detector = NonStationaryDetector(window_size=20)
        detector = detector.fit(control, treatment, timestamps)
        result = detector.result()

        assert isinstance(result, NonStationaryResult)
        assert result.is_stationary is True
        assert result.effect_trend == "stable"

    def test_increasing_trend_detected(self, rng):
        """Linearly increasing treatment effect should be detected."""
        n = 200
        timestamps = np.arange(n)
        control = rng.normal(10, 0.3, n)
        # Treatment effect grows from 0 to 10 over time
        effect = np.linspace(0, 10, n)
        treatment = control + effect + rng.normal(0, 0.3, n)

        detector = NonStationaryDetector(window_size=10)
        detector = detector.fit(control, treatment, timestamps)
        result = detector.result()

        assert result.effect_trend in ("increasing", "volatile")
        assert result.is_stationary is False

    def test_decreasing_trend_detected(self, rng):
        """Linearly decreasing treatment effect should be detected."""
        n = 200
        timestamps = np.arange(n)
        control = rng.normal(10, 0.3, n)
        effect = np.linspace(10, 0, n)
        treatment = control + effect + rng.normal(0, 0.3, n)

        detector = NonStationaryDetector(window_size=10)
        detector = detector.fit(control, treatment, timestamps)
        result = detector.result()

        assert result.effect_trend in ("decreasing", "volatile")
        assert result.is_stationary is False

    def test_volatile_data_detected(self, rng):
        """Highly volatile effect should be detected."""
        n = 200
        timestamps = np.arange(n)
        control = rng.normal(10, 0.3, n)
        # Effect alternates block-wise (each block > window_size)
        # so each window sees a different effect
        effect = np.where(np.arange(n) % 40 < 20, 8.0, -8.0)
        treatment = control + effect + rng.normal(0, 0.3, n)

        detector = NonStationaryDetector(window_size=10)
        detector = detector.fit(control, treatment, timestamps)
        result = detector.result()

        # Should detect non-stationarity
        assert result.is_stationary is False

    def test_change_point_detected(self, rng):
        """Abrupt change in treatment effect should yield change points."""
        n = 200
        timestamps = np.arange(n)
        control = rng.normal(10, 0.3, n)
        # Effect shifts from 0 to 5 at midpoint
        effect = np.where(np.arange(n) < 100, 0.0, 8.0)
        treatment = control + effect + rng.normal(0, 0.3, n)

        detector = NonStationaryDetector(window_size=10, threshold=0.05)
        detector = detector.fit(control, treatment, timestamps)
        result = detector.result()

        assert result.is_stationary is False
        # Should have at least one change point
        assert len(result.change_points) >= 1 or result.effect_trend != "stable"

    def test_window_effects_populated(self, rng):
        """window_effects should contain dictionaries with expected keys."""
        n = 50
        timestamps = np.arange(n)
        control = rng.normal(10, 1, n)
        treatment = rng.normal(11, 1, n)

        detector = NonStationaryDetector(window_size=10)
        detector = detector.fit(control, treatment, timestamps)
        result = detector.result()

        assert len(result.window_effects) > 0
        for w in result.window_effects:
            assert "start" in w
            assert "end" in w
            assert "effect" in w

    def test_custom_window_size(self, rng):
        """Different window sizes should produce different numbers of windows."""
        n = 100
        timestamps = np.arange(n)
        control = rng.normal(10, 1, n)
        treatment = rng.normal(11, 1, n)

        r1 = NonStationaryDetector(window_size=5).fit(control, treatment, timestamps).result()
        r2 = NonStationaryDetector(window_size=20).fit(control, treatment, timestamps).result()

        assert len(r1.window_effects) > len(r2.window_effects)

    def test_to_dict(self, rng):
        """Result should serialise to a dictionary."""
        n = 50
        timestamps = np.arange(n)
        control = rng.normal(10, 1, n)
        treatment = rng.normal(11, 1, n)

        result = NonStationaryDetector(window_size=10).fit(
            control, treatment, timestamps
        ).result()

        d = result.to_dict()
        assert isinstance(d, dict)
        assert "is_stationary" in d
        assert "effect_trend" in d

    def test_repr(self, rng):
        """Result __repr__ should be a string."""
        n = 50
        timestamps = np.arange(n)
        control = rng.normal(10, 1, n)
        treatment = rng.normal(11, 1, n)

        result = NonStationaryDetector(window_size=10).fit(
            control, treatment, timestamps
        ).result()

        assert "NonStationaryResult" in repr(result)

    def test_unsorted_timestamps(self, rng):
        """Data should be sorted by timestamps internally."""
        n = 50
        timestamps = rng.permutation(n).astype(float)
        control = rng.normal(10, 1, n)
        treatment = rng.normal(11, 1, n)

        # Should not raise
        result = NonStationaryDetector(window_size=10).fit(
            control, treatment, timestamps
        ).result()

        assert isinstance(result, NonStationaryResult)


class TestNonStationaryValidation:
    """Validation and error handling tests."""

    def test_window_size_too_large_raises(self, rng):
        """Window size larger than data should raise ValueError."""
        with pytest.raises(ValueError, match="at least"):
            NonStationaryDetector(window_size=100).fit(
                [1, 2, 3], [4, 5, 6], [0, 1, 2]
            )

    def test_mismatched_lengths_raises(self):
        """Mismatched array lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            NonStationaryDetector().fit([1, 2, 3], [4, 5], [0, 1, 2])

    def test_invalid_window_size_raises(self):
        """Window size < 2 should raise ValueError."""
        with pytest.raises(ValueError, match="window_size"):
            NonStationaryDetector(window_size=1)

    def test_result_before_fit_raises(self):
        """Calling result() before fit() should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="fitted"):
            NonStationaryDetector().result()
