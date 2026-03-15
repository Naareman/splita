"""Tests for splita.explain — plain-English result interpretation."""

from __future__ import annotations

import json

import pytest

from splita._types import (
    BayesianResult,
    ExperimentResult,
    SampleSizeResult,
    SRMResult,
)
from splita.explain import explain


# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def significant_result() -> ExperimentResult:
    return ExperimentResult(
        control_mean=0.10,
        treatment_mean=0.12,
        lift=0.02,
        relative_lift=0.20,
        pvalue=0.003,
        statistic=2.97,
        ci_lower=0.007,
        ci_upper=0.033,
        significant=True,
        alpha=0.05,
        method="ztest",
        metric="conversion",
        control_n=5000,
        treatment_n=5000,
        power=0.82,
        effect_size=0.15,
    )


@pytest.fixture
def nonsignificant_result() -> ExperimentResult:
    return ExperimentResult(
        control_mean=0.10,
        treatment_mean=0.101,
        lift=0.001,
        relative_lift=0.01,
        pvalue=0.42,
        statistic=0.81,
        ci_lower=-0.004,
        ci_upper=0.006,
        significant=False,
        alpha=0.05,
        method="ztest",
        metric="conversion",
        control_n=200,
        treatment_n=200,
        power=0.12,
        effect_size=0.01,
    )


@pytest.fixture
def underpowered_significant() -> ExperimentResult:
    return ExperimentResult(
        control_mean=0.10,
        treatment_mean=0.12,
        lift=0.02,
        relative_lift=0.20,
        pvalue=0.048,
        statistic=1.98,
        ci_lower=0.0001,
        ci_upper=0.04,
        significant=True,
        alpha=0.05,
        method="ztest",
        metric="conversion",
        control_n=500,
        treatment_n=500,
        power=0.55,
        effect_size=0.064,
    )


@pytest.fixture
def srm_failed() -> SRMResult:
    return SRMResult(
        observed=[4500, 5500],
        expected_counts=[5000.0, 5000.0],
        chi2_statistic=100.0,
        pvalue=0.001,
        passed=False,
        alpha=0.01,
        deviations_pct=[-10.0, 10.0],
        worst_variant=1,
        message="Sample ratio mismatch detected!",
    )


@pytest.fixture
def srm_passed() -> SRMResult:
    return SRMResult(
        observed=[4990, 5010],
        expected_counts=[5000.0, 5000.0],
        chi2_statistic=0.04,
        pvalue=0.84,
        passed=True,
        alpha=0.01,
        deviations_pct=[-0.2, 0.2],
        worst_variant=1,
        message="No sample ratio mismatch detected.",
    )


@pytest.fixture
def bayesian_strong() -> BayesianResult:
    return BayesianResult(
        prob_b_beats_a=0.983,
        expected_loss_a=0.015,
        expected_loss_b=0.0003,
        lift=0.02,
        relative_lift=0.20,
        ci_lower=0.005,
        ci_upper=0.035,
        credible_level=0.95,
        control_mean=0.10,
        treatment_mean=0.12,
        prob_in_rope=None,
        rope=None,
        metric="conversion",
        control_n=5000,
        treatment_n=5000,
    )


@pytest.fixture
def bayesian_weak() -> BayesianResult:
    return BayesianResult(
        prob_b_beats_a=0.62,
        expected_loss_a=0.005,
        expected_loss_b=0.003,
        lift=0.005,
        relative_lift=0.05,
        ci_lower=-0.002,
        ci_upper=0.012,
        credible_level=0.95,
        control_mean=0.10,
        treatment_mean=0.105,
        prob_in_rope=0.35,
        rope=(-0.005, 0.005),
        metric="conversion",
        control_n=2000,
        treatment_n=2000,
    )


@pytest.fixture
def bayesian_control_wins() -> BayesianResult:
    return BayesianResult(
        prob_b_beats_a=0.03,
        expected_loss_a=0.0001,
        expected_loss_b=0.02,
        lift=-0.03,
        relative_lift=-0.25,
        ci_lower=-0.05,
        ci_upper=-0.01,
        credible_level=0.95,
        control_mean=0.12,
        treatment_mean=0.09,
        prob_in_rope=None,
        rope=None,
        metric="conversion",
        control_n=5000,
        treatment_n=5000,
    )


@pytest.fixture
def sample_size_result() -> SampleSizeResult:
    return SampleSizeResult(
        n_per_variant=3843,
        n_total=7686,
        alpha=0.05,
        power=0.80,
        mde=0.02,
        relative_mde=0.20,
        baseline=0.10,
        metric="conversion",
        effect_size=0.064,
        days_needed=None,
    )


@pytest.fixture
def sample_size_with_days() -> SampleSizeResult:
    return SampleSizeResult(
        n_per_variant=3843,
        n_total=7686,
        alpha=0.05,
        power=0.80,
        mde=0.02,
        relative_mde=0.20,
        baseline=0.10,
        metric="conversion",
        effect_size=0.064,
        days_needed=8,
    )


# ─── Test: ExperimentResult explain ─────────────────────────────────


class TestExplainExperiment:
    def test_significant_mentions_significant(
        self, significant_result: ExperimentResult
    ) -> None:
        text = explain(significant_result)
        assert "statistically significant" in text
        assert "p=0.00" in text

    def test_significant_mentions_effect_size(
        self, significant_result: ExperimentResult
    ) -> None:
        text = explain(significant_result)
        assert "Cohen's d" in text
        assert "negligible" in text  # 0.15 < 0.2 threshold

    def test_significant_mentions_power(
        self, significant_result: ExperimentResult
    ) -> None:
        text = explain(significant_result)
        assert "power" in text.lower()
        assert "adequate" in text

    def test_nonsignificant_mentions_no_difference(
        self, nonsignificant_result: ExperimentResult
    ) -> None:
        text = explain(nonsignificant_result)
        assert "No statistically significant" in text
        assert "p=0.42" in text

    def test_nonsignificant_suggests_cuped(
        self, nonsignificant_result: ExperimentResult
    ) -> None:
        text = explain(nonsignificant_result)
        assert "CUPED" in text

    def test_underpowered_warns(
        self, underpowered_significant: ExperimentResult
    ) -> None:
        text = explain(underpowered_significant)
        assert "underpowered" in text.lower()

    def test_decreased_effect(self) -> None:
        result = ExperimentResult(
            control_mean=0.12,
            treatment_mean=0.10,
            lift=-0.02,
            relative_lift=-0.167,
            pvalue=0.003,
            statistic=-2.97,
            ci_lower=-0.033,
            ci_upper=-0.007,
            significant=True,
            alpha=0.05,
            method="ztest",
            metric="conversion",
            control_n=5000,
            treatment_n=5000,
            power=0.82,
            effect_size=0.15,
        )
        text = explain(result)
        assert "decreased" in text


# ─── Test: SRMResult explain ────────────────────────────────────────


class TestExplainSRM:
    def test_srm_failed_warning(self, srm_failed: SRMResult) -> None:
        text = explain(srm_failed)
        assert "WARNING" in text
        assert "Sample Ratio Mismatch" in text
        assert "invalid" in text

    def test_srm_failed_suggestions(self, srm_failed: SRMResult) -> None:
        text = explain(srm_failed)
        assert "randomization" in text.lower()
        assert "bot" in text.lower()

    def test_srm_passed(self, srm_passed: SRMResult) -> None:
        text = explain(srm_passed)
        assert "No Sample Ratio Mismatch" in text
        assert "WARNING" not in text


# ─── Test: BayesianResult explain ───────────────────────────────────


class TestExplainBayesian:
    def test_strong_treatment(self, bayesian_strong: BayesianResult) -> None:
        text = explain(bayesian_strong)
        assert "98.30%" in text
        assert "ship" in text.lower()

    def test_weak_evidence(self, bayesian_weak: BayesianResult) -> None:
        text = explain(bayesian_weak)
        assert "62.00%" in text
        assert "not yet decisive" in text

    def test_rope_mentioned(self, bayesian_weak: BayesianResult) -> None:
        text = explain(bayesian_weak)
        assert "ROPE" in text
        assert "negligible" in text

    def test_control_wins(
        self, bayesian_control_wins: BayesianResult
    ) -> None:
        text = explain(bayesian_control_wins)
        assert "3.00%" in text
        assert "keep the control" in text.lower()


# ─── Test: SampleSizeResult explain ─────────────────────────────────


class TestExplainSampleSize:
    def test_basic_output(self, sample_size_result: SampleSizeResult) -> None:
        text = explain(sample_size_result)
        assert "3,843" in text
        assert "7,686" in text
        assert "80.00%" in text

    def test_with_days(
        self, sample_size_with_days: SampleSizeResult
    ) -> None:
        text = explain(sample_size_with_days)
        assert "8 days" in text


# ─── Test: unsupported type ─────────────────────────────────────────


class TestExplainUnsupported:
    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(TypeError, match="does not support"):
            explain("not a result")

    def test_error_lists_supported_types(self) -> None:
        with pytest.raises(TypeError, match="ExperimentResult"):
            explain(42)


# ─── Test: serialization roundtrip ──────────────────────────────────


class TestSerialization:
    def test_experiment_to_json(
        self, significant_result: ExperimentResult
    ) -> None:
        json_str = significant_result.to_json()
        parsed = json.loads(json_str)
        assert parsed["control_mean"] == 0.10
        assert parsed["significant"] is True

    def test_experiment_roundtrip(
        self, significant_result: ExperimentResult
    ) -> None:
        d = significant_result.to_dict()
        restored = ExperimentResult.from_dict(d)
        assert restored.control_mean == significant_result.control_mean
        assert restored.pvalue == significant_result.pvalue
        assert restored.significant == significant_result.significant
        assert restored.method == significant_result.method

    def test_srm_roundtrip(self, srm_failed: SRMResult) -> None:
        d = srm_failed.to_dict()
        restored = SRMResult.from_dict(d)
        assert restored.passed == srm_failed.passed
        assert restored.pvalue == srm_failed.pvalue
        assert restored.observed == srm_failed.observed

    def test_bayesian_roundtrip(
        self, bayesian_strong: BayesianResult
    ) -> None:
        json_str = bayesian_strong.to_json()
        parsed = json.loads(json_str)
        restored = BayesianResult.from_dict(parsed)
        assert restored.prob_b_beats_a == bayesian_strong.prob_b_beats_a
        assert restored.metric == bayesian_strong.metric

    def test_sample_size_roundtrip(
        self, sample_size_result: SampleSizeResult
    ) -> None:
        d = sample_size_result.to_dict()
        restored = SampleSizeResult.from_dict(d)
        assert restored.n_per_variant == sample_size_result.n_per_variant
        assert restored.baseline == sample_size_result.baseline

    def test_from_dict_ignores_extra_keys(self) -> None:
        d = dict(
            control_mean=0.10,
            treatment_mean=0.12,
            lift=0.02,
            relative_lift=0.2,
            pvalue=0.03,
            statistic=2.1,
            ci_lower=0.002,
            ci_upper=0.038,
            significant=True,
            alpha=0.05,
            method="ztest",
            metric="conversion",
            control_n=1000,
            treatment_n=1000,
            power=0.65,
            effect_size=0.06,
            extra_field="should be ignored",
        )
        result = ExperimentResult.from_dict(d)
        assert result.control_mean == 0.10
        assert not hasattr(result, "extra_field")

    def test_to_json_indent(
        self, significant_result: ExperimentResult
    ) -> None:
        json_str = significant_result.to_json(indent=4)
        # 4-space indentation should produce lines with 4 leading spaces
        assert "    " in json_str

    def test_to_json_parses_as_valid_json(
        self, significant_result: ExperimentResult
    ) -> None:
        json_str = significant_result.to_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert len(parsed) == 16  # 16 fields in ExperimentResult

    def test_bayesian_roundtrip_with_rope(
        self, bayesian_weak: BayesianResult
    ) -> None:
        json_str = bayesian_weak.to_json()
        parsed = json.loads(json_str)
        # rope is a tuple, serialized as a list in JSON
        restored = BayesianResult.from_dict(
            {**parsed, "rope": tuple(parsed["rope"]) if parsed["rope"] else None}
        )
        assert restored.prob_in_rope == bayesian_weak.prob_in_rope
        assert restored.rope == bayesian_weak.rope
