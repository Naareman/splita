"""Tests for FlickerDetector."""

import numpy as np
import pytest

from splita._types import FlickerResult
from splita.diagnostics.flicker import FlickerDetector


# ─── No flickers ───────────────────────────────────────────────────


class TestNoFlickers:
    def test_stable_assignments(self):
        user_ids = [1, 2, 3, 1, 2, 3]
        variants = [0, 1, 0, 0, 1, 0]
        result = FlickerDetector().detect(user_ids, variants)
        assert result.flicker_rate == 0.0
        assert result.n_flickers == 0
        assert result.n_users == 3
        assert result.is_problematic is False
        assert result.flicker_users == []

    def test_single_observation_per_user(self):
        user_ids = [1, 2, 3, 4, 5]
        variants = [0, 1, 0, 1, 0]
        result = FlickerDetector().detect(user_ids, variants)
        assert result.flicker_rate == 0.0
        assert result.n_flickers == 0

    def test_numpy_arrays(self):
        result = FlickerDetector().detect(
            np.array([10, 20, 30, 10]),
            np.array([0, 1, 0, 0]),
        )
        assert result.flicker_rate == 0.0

    def test_result_is_frozen_dataclass(self):
        result = FlickerDetector().detect([1, 2], [0, 1])
        assert isinstance(result, FlickerResult)
        with pytest.raises(AttributeError):
            result.flicker_rate = 0.5  # type: ignore[misc]

    def test_to_dict(self):
        result = FlickerDetector().detect([1, 2], [0, 1])
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "flicker_rate" in d


# ─── Some flickers ────────────────────────────────────────────────


class TestSomeFlickers:
    def test_one_user_flickers(self):
        user_ids = [1, 2, 3, 1, 2, 3]
        variants = [0, 1, 0, 1, 1, 0]  # user 1 flickers: 0 -> 1
        result = FlickerDetector().detect(user_ids, variants)
        assert result.n_flickers == 1
        assert result.n_users == 3
        assert abs(result.flicker_rate - 1 / 3) < 1e-10
        assert 1 in result.flicker_users

    def test_multiple_users_flicker(self):
        user_ids = [1, 2, 3, 1, 2, 3]
        variants = [0, 1, 0, 1, 0, 0]  # users 1 and 2 flicker
        result = FlickerDetector().detect(user_ids, variants)
        assert result.n_flickers == 2
        assert 1 in result.flicker_users
        assert 2 in result.flicker_users


# ─── All flickers ──────────────────────────────────────────────────


class TestAllFlickers:
    def test_everyone_flickers(self):
        user_ids = [1, 2, 3, 1, 2, 3]
        variants = [0, 1, 0, 1, 0, 1]  # all 3 users switch
        result = FlickerDetector().detect(user_ids, variants)
        assert result.n_flickers == 3
        assert result.flicker_rate == 1.0
        assert result.is_problematic is True


# ─── Threshold ─────────────────────────────────────────────────────


class TestThreshold:
    def test_below_threshold_not_problematic(self):
        # 1 of 100 users flickers = 1%, threshold = 5%
        user_ids = list(range(100)) + [0]
        variants = [0] * 100 + [1]  # user 0 flickers
        result = FlickerDetector(threshold=0.05).detect(user_ids, variants)
        assert result.is_problematic is False

    def test_above_threshold_is_problematic(self):
        user_ids = [1, 2, 1, 2]
        variants = [0, 1, 1, 0]  # both flicker = 100%
        result = FlickerDetector(threshold=0.01).detect(user_ids, variants)
        assert result.is_problematic is True

    def test_custom_threshold(self):
        user_ids = list(range(10)) + [0]
        variants = [0] * 10 + [1]  # 1 of 10 = 10%
        result_low = FlickerDetector(threshold=0.05).detect(user_ids, variants)
        result_high = FlickerDetector(threshold=0.15).detect(user_ids, variants)
        assert result_low.is_problematic is True
        assert result_high.is_problematic is False


# ─── With timestamps ──────────────────────────────────────────────


class TestTimestamps:
    def test_timestamps_accepted(self):
        result = FlickerDetector().detect(
            user_ids=[1, 2, 1],
            variant_assignments=[0, 1, 0],
            timestamps=[100, 200, 300],
        )
        assert result.n_users == 2


# ─── Validation ────────────────────────────────────────────────────


class TestValidation:
    def test_invalid_threshold(self):
        with pytest.raises(ValueError, match="threshold"):
            FlickerDetector(threshold=1.5)

    def test_empty_user_ids(self):
        with pytest.raises(ValueError, match="user_ids"):
            FlickerDetector().detect([], [])

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            FlickerDetector().detect([1, 2, 3], [0, 1])

    def test_invalid_user_ids_type(self):
        with pytest.raises(TypeError, match="user_ids"):
            FlickerDetector().detect("not_array", [0, 1])

    def test_invalid_variant_type(self):
        with pytest.raises(TypeError, match="variant_assignments"):
            FlickerDetector().detect([1, 2], "not_array")

    def test_mismatched_timestamps(self):
        with pytest.raises(ValueError, match="timestamps"):
            FlickerDetector().detect([1, 2], [0, 1], timestamps=[100])

    def test_2d_user_ids(self):
        with pytest.raises(ValueError, match="1-D"):
            FlickerDetector().detect(np.array([[1, 2], [3, 4]]), [0, 1, 0, 1])


# ─── Message ──────────────────────────────────────────────────────


class TestMessage:
    def test_no_flicker_message(self):
        result = FlickerDetector().detect([1, 2], [0, 1])
        assert "No flickers" in result.message

    def test_flicker_message_includes_count(self):
        result = FlickerDetector().detect([1, 1], [0, 1])
        assert "1" in result.message
        assert "flickered" in result.message


# ─── Repr ──────────────────────────────────────────────────────────


class TestRepr:
    def test_repr_contains_key_info(self):
        result = FlickerDetector().detect([1, 2], [0, 1])
        text = repr(result)
        assert "FlickerResult" in text
        assert "flicker_rate" in text
