"""Tests for SampleSizeReestimation."""

from __future__ import annotations

import pytest

from splita.sequential.sample_size_reest import SampleSizeReestimation


class TestReestimate:
    def test_basic_reestimate(self):
        result = SampleSizeReestimation.reestimate(
            current_n=500,
            interim_effect=0.05,
            interim_se=0.04,
            target_power=0.80,
            alpha=0.05,
        )
        assert result.new_n_per_variant >= 500

    def test_returns_result_type(self):
        result = SampleSizeReestimation.reestimate(
            current_n=500,
            interim_effect=0.1,
            interim_se=0.04,
        )
        assert hasattr(result, "new_n_per_variant")
        assert hasattr(result, "conditional_power_current")
        assert hasattr(result, "conditional_power_new")
        assert hasattr(result, "increase_ratio")

    def test_power_increases(self):
        result = SampleSizeReestimation.reestimate(
            current_n=100,
            interim_effect=0.1,
            interim_se=0.1,
            target_power=0.80,
        )
        assert result.conditional_power_new >= result.conditional_power_current

    def test_large_effect_needs_less_increase(self):
        small = SampleSizeReestimation.reestimate(
            current_n=100, interim_effect=0.05, interim_se=0.1
        )
        large = SampleSizeReestimation.reestimate(
            current_n=100, interim_effect=0.5, interim_se=0.1
        )
        assert large.increase_ratio <= small.increase_ratio

    def test_already_powered(self):
        result = SampleSizeReestimation.reestimate(
            current_n=10000,
            interim_effect=0.5,
            interim_se=0.01,
            target_power=0.80,
        )
        assert result.new_n_per_variant == 10000
        assert result.conditional_power_current > 0.8

    def test_increase_ratio_is_correct(self):
        result = SampleSizeReestimation.reestimate(
            current_n=500,
            interim_effect=0.05,
            interim_se=0.04,
        )
        assert abs(result.increase_ratio - result.new_n_per_variant / 500) < 1e-10

    def test_zero_effect(self):
        result = SampleSizeReestimation.reestimate(
            current_n=500,
            interim_effect=0.0,
            interim_se=0.04,
        )
        assert result.new_n_per_variant > 500  # needs much more data

    def test_conditional_power_bounds(self):
        result = SampleSizeReestimation.reestimate(
            current_n=500,
            interim_effect=0.1,
            interim_se=0.04,
        )
        assert 0.0 <= result.conditional_power_current <= 1.0
        assert 0.0 <= result.conditional_power_new <= 1.0


class TestResultMethods:
    def test_to_dict(self):
        result = SampleSizeReestimation.reestimate(
            current_n=500, interim_effect=0.05, interim_se=0.04
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "new_n_per_variant" in d

    def test_repr(self):
        result = SampleSizeReestimation.reestimate(
            current_n=500, interim_effect=0.05, interim_se=0.04
        )
        assert "ReestimationResult" in repr(result)


class TestValidation:
    def test_invalid_current_n(self):
        with pytest.raises(ValueError, match="current_n"):
            SampleSizeReestimation.reestimate(
                current_n=1, interim_effect=0.05, interim_se=0.04
            )

    def test_negative_se(self):
        with pytest.raises(ValueError, match="interim_se"):
            SampleSizeReestimation.reestimate(
                current_n=500, interim_effect=0.05, interim_se=-0.04
            )

    def test_invalid_power(self):
        with pytest.raises(ValueError, match="target_power"):
            SampleSizeReestimation.reestimate(
                current_n=500,
                interim_effect=0.05,
                interim_se=0.04,
                target_power=1.5,
            )

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            SampleSizeReestimation.reestimate(
                current_n=500,
                interim_effect=0.05,
                interim_se=0.04,
                alpha=0.0,
            )
