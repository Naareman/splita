"""Tests for AATest diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from splita._types import AATestResult
from splita.diagnostics.aa_test import AATest


# ─── Construction / validation ───────────────────────────────────────


class TestAATestInit:
    """Tests for AATest.__init__ validation."""

    def test_default_params(self) -> None:
        aa = AATest()
        assert aa._n_simulations == 1000
        assert aa._alpha == 0.05

    def test_custom_params(self) -> None:
        aa = AATest(n_simulations=500, alpha=0.10, random_state=42)
        assert aa._n_simulations == 500
        assert aa._alpha == 0.10

    def test_n_simulations_too_small(self) -> None:
        with pytest.raises(ValueError, match="n_simulations"):
            AATest(n_simulations=5)

    def test_n_simulations_float_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_simulations"):
            AATest(n_simulations=100.5)

    def test_alpha_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            AATest(alpha=1.5)

    def test_alpha_zero(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            AATest(alpha=0.0)


# ─── Clean data → pass ──────────────────────────────────────────────


class TestAATestCleanData:
    """Tests with well-behaved data that should pass."""

    def test_conversion_data_passes(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.binomial(1, 0.10, size=2000)
        result = AATest(n_simulations=200, random_state=42).run(data)
        assert isinstance(result, AATestResult)
        assert result.passed is True

    def test_continuous_data_passes(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(100, 15, size=2000)
        result = AATest(n_simulations=200, random_state=42).run(data)
        assert result.passed is True

    def test_fp_rate_near_alpha(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.binomial(1, 0.10, size=2000)
        result = AATest(n_simulations=500, alpha=0.05, random_state=42).run(data)
        # FP rate should be near alpha (within reasonable bounds)
        assert 0.0 <= result.false_positive_rate <= 0.15

    def test_expected_rate_equals_alpha(self) -> None:
        result = AATest(n_simulations=100, alpha=0.05, random_state=42).run(
            np.random.default_rng(0).normal(0, 1, size=1000)
        )
        assert result.expected_rate == 0.05

    def test_n_simulations_in_result(self) -> None:
        result = AATest(n_simulations=100, random_state=42).run(
            np.random.default_rng(0).binomial(1, 0.5, size=500)
        )
        # n_simulations in result is the number of successful simulations
        assert result.n_simulations <= 100
        assert result.n_simulations > 0


# ─── P-value distribution ───────────────────────────────────────────


class TestAATestPValues:
    """Tests for p-value distribution properties."""

    def test_p_values_between_0_and_1(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.binomial(1, 0.10, size=2000)
        result = AATest(n_simulations=200, random_state=42).run(data)
        for p in result.p_values:
            assert 0.0 <= p <= 1.0

    def test_p_values_roughly_uniform(self) -> None:
        """P-values from A/A test should be roughly uniformly distributed."""
        rng = np.random.default_rng(42)
        data = rng.normal(50, 10, size=4000)
        result = AATest(n_simulations=500, random_state=42).run(data)

        pvals = np.array(result.p_values)
        # Check median is roughly 0.5 (within reasonable tolerance)
        median_p = float(np.median(pvals))
        assert 0.2 < median_p < 0.8

    def test_p_values_list_length(self) -> None:
        result = AATest(n_simulations=50, random_state=42).run(
            np.random.default_rng(0).binomial(1, 0.3, size=500)
        )
        assert len(result.p_values) > 0
        assert len(result.p_values) <= 50


# ─── Reproducibility ────────────────────────────────────────────────


class TestAATestEdgeCases:
    """Edge cases for uncovered lines."""

    def test_all_simulations_fail(self) -> None:
        """Lines 108-113: when all simulations fail, returns failed result."""
        # Using conversion metric with non-binary data causes Experiment to fail
        # every time (negative values fail the conversion z-test).
        data = np.array([-5.0, -3.0, -1.0, -2.0, -4.0, -6.0])
        result = AATest(n_simulations=10, random_state=42).run(data, metric="conversion")
        assert result.passed is False
        assert result.p_values == []
        assert result.false_positive_rate == 0.0
        assert "failed" in result.message.lower()

    def test_aa_test_fail_message(self) -> None:
        """Line 136: message when AA test fails (FP rate outside bounds)."""
        # Constant data produces 0% FP rate, which falls outside the
        # expected bounds when enough sims make the bounds tight
        data = np.zeros(40)
        result = AATest(n_simulations=200, alpha=0.05, random_state=0).run(data)
        assert result.passed is False
        assert "failed" in result.message.lower()


class TestAATestReproducibility:
    """Tests for reproducible results."""

    def test_same_seed_same_result(self) -> None:
        data = np.random.default_rng(0).binomial(1, 0.10, size=1000)
        r1 = AATest(n_simulations=100, random_state=42).run(data)
        r2 = AATest(n_simulations=100, random_state=42).run(data)
        assert r1.false_positive_rate == r2.false_positive_rate
        assert r1.p_values == r2.p_values

    def test_different_seed_different_result(self) -> None:
        data = np.random.default_rng(0).binomial(1, 0.10, size=1000)
        r1 = AATest(n_simulations=100, random_state=42).run(data)
        r2 = AATest(n_simulations=100, random_state=99).run(data)
        # Very unlikely to be identical
        assert r1.p_values != r2.p_values


# ─── Input validation ───────────────────────────────────────────────


class TestAATestInputValidation:
    """Tests for run() input validation."""

    def test_data_too_short(self) -> None:
        with pytest.raises(ValueError, match="data"):
            AATest(n_simulations=10, random_state=42).run([1, 0, 1])

    def test_non_array_data(self) -> None:
        with pytest.raises(TypeError, match="data"):
            AATest(n_simulations=10, random_state=42).run("not_an_array")


# ─── Result properties ──────────────────────────────────────────────


class TestAATestResultProperties:
    """Tests for AATestResult dataclass."""

    def test_to_dict(self) -> None:
        data = np.random.default_rng(0).binomial(1, 0.10, size=1000)
        result = AATest(n_simulations=50, random_state=42).run(data)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "false_positive_rate" in d
        assert "passed" in d
        assert "p_values" in d

    def test_repr(self) -> None:
        data = np.random.default_rng(0).binomial(1, 0.10, size=1000)
        result = AATest(n_simulations=50, random_state=42).run(data)
        text = repr(result)
        assert "AATestResult" in text
        assert "false_positive_rate" in text

    def test_message_on_pass(self) -> None:
        data = np.random.default_rng(42).binomial(1, 0.10, size=2000)
        result = AATest(n_simulations=200, random_state=42).run(data)
        if result.passed:
            assert "passed" in result.message

    def test_frozen_dataclass(self) -> None:
        data = np.random.default_rng(0).binomial(1, 0.10, size=1000)
        result = AATest(n_simulations=50, random_state=42).run(data)
        with pytest.raises(AttributeError):
            result.passed = False  # type: ignore[misc]
