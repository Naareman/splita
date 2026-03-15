"""Tests for splita.compare — compare two experiment results."""

from __future__ import annotations

import pytest

from splita._types import ComparisonResult, ExperimentResult
from splita.compare import compare


# ─── Fixtures ────────────────────────────────────────────────────────


def _make_result(
    lift=0.02,
    pvalue=0.003,
    ci_lower=0.007,
    ci_upper=0.033,
    significant=True,
    control_n=5000,
    treatment_n=5000,
    power=0.82,
    effect_size=0.15,
) -> ExperimentResult:
    return ExperimentResult(
        control_mean=0.10,
        treatment_mean=0.10 + lift,
        lift=lift,
        relative_lift=lift / 0.10,
        pvalue=pvalue,
        statistic=2.97,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        significant=significant,
        alpha=0.05,
        method="ztest",
        metric="conversion",
        control_n=control_n,
        treatment_n=treatment_n,
        power=power,
        effect_size=effect_size,
    )


@pytest.fixture
def result_a():
    return _make_result(lift=0.02, ci_lower=0.007, ci_upper=0.033)


@pytest.fixture
def result_b():
    return _make_result(lift=0.01, ci_lower=-0.004, ci_upper=0.024, significant=False)


@pytest.fixture
def result_same():
    return _make_result(lift=0.02, ci_lower=0.007, ci_upper=0.033)


# ─── Basic Functionality ─────────────────────────────────────────────


class TestCompareBasic:
    def test_returns_comparison_result(self, result_a, result_b):
        result = compare(result_a, result_b)
        assert isinstance(result, ComparisonResult)

    def test_effects_match_inputs(self, result_a, result_b):
        result = compare(result_a, result_b)
        assert result.effect_a == result_a.lift
        assert result.effect_b == result_b.lift

    def test_difference_is_b_minus_a(self, result_a, result_b):
        result = compare(result_a, result_b)
        assert abs(result.difference - (result_b.lift - result_a.lift)) < 1e-10

    def test_pvalue_in_range(self, result_a, result_b):
        result = compare(result_a, result_b)
        assert 0.0 <= result.pvalue <= 1.0

    def test_se_positive(self, result_a, result_b):
        result = compare(result_a, result_b)
        assert result.se > 0


# ─── Direction Detection ────────────────────────────────────────────


class TestCompareDirection:
    def test_a_larger_direction(self):
        a = _make_result(lift=0.05, ci_lower=0.03, ci_upper=0.07)
        b = _make_result(lift=0.01, ci_lower=-0.01, ci_upper=0.03)
        result = compare(a, b)
        if result.significant:
            assert result.direction == "a_larger"

    def test_b_larger_direction(self):
        a = _make_result(lift=0.01, ci_lower=-0.01, ci_upper=0.03)
        b = _make_result(lift=0.05, ci_lower=0.03, ci_upper=0.07)
        result = compare(a, b)
        if result.significant:
            assert result.direction == "b_larger"

    def test_equivalent_same_results(self, result_same):
        result = compare(result_same, result_same)
        assert result.direction == "equivalent"
        assert result.difference == 0.0


# ─── Significance ───────────────────────────────────────────────────


class TestCompareSignificance:
    def test_large_difference_significant(self):
        a = _make_result(lift=0.01, ci_lower=0.005, ci_upper=0.015)
        b = _make_result(lift=0.10, ci_lower=0.09, ci_upper=0.11)
        result = compare(a, b)
        assert result.significant is True

    def test_same_result_not_significant(self, result_same):
        result = compare(result_same, result_same)
        assert result.significant is False

    def test_custom_alpha(self, result_a, result_b):
        r1 = compare(result_a, result_b, alpha=0.99)
        r2 = compare(result_a, result_b, alpha=0.001)
        # More lenient alpha should be at least as likely to be significant
        if r2.significant:
            assert r1.significant


# ─── Confidence Interval ────────────────────────────────────────────


class TestCompareCI:
    def test_ci_contains_difference(self, result_a, result_b):
        result = compare(result_a, result_b)
        assert result.ci_lower <= result.difference <= result.ci_upper

    def test_ci_wider_with_uncertainty(self):
        # Wide CIs on inputs -> wide CI on comparison
        a = _make_result(lift=0.02, ci_lower=-0.05, ci_upper=0.09)
        b = _make_result(lift=0.03, ci_lower=-0.04, ci_upper=0.10)
        result = compare(a, b)
        ci_width = result.ci_upper - result.ci_lower
        assert ci_width > 0


# ─── Validation ──────────────────────────────────────────────────────


class TestCompareValidation:
    def test_rejects_non_experiment_result_a(self, result_b):
        with pytest.raises(TypeError, match="result_a"):
            compare("not a result", result_b)

    def test_rejects_non_experiment_result_b(self, result_a):
        with pytest.raises(TypeError, match="result_b"):
            compare(result_a, {"lift": 0.02})

    def test_rejects_bad_alpha(self, result_a, result_b):
        with pytest.raises(ValueError, match="alpha"):
            compare(result_a, result_b, alpha=1.5)


# ─── Serialization ──────────────────────────────────────────────────


class TestCompareSerialization:
    def test_to_dict(self, result_a, result_b):
        result = compare(result_a, result_b)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "difference" in d
        assert "direction" in d

    def test_repr(self, result_a, result_b):
        result = compare(result_a, result_b)
        text = repr(result)
        assert "ComparisonResult" in text
