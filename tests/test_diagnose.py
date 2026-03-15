"""Tests for splita.diagnose — structured next-steps diagnosis."""

from __future__ import annotations

import pytest

from splita._types import DiagnosisResult, ExperimentResult
from splita.diagnose import diagnose


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
def healthy_result():
    return _make_result(power=0.85, control_n=5000, treatment_n=5000)


@pytest.fixture
def underpowered_result():
    return _make_result(
        power=0.30,
        significant=False,
        pvalue=0.42,
        control_n=200,
        treatment_n=200,
    )


@pytest.fixture
def critical_result():
    return _make_result(
        power=0.10,
        significant=False,
        pvalue=0.80,
        control_n=20,
        treatment_n=20,
        ci_lower=-0.20,
        ci_upper=0.24,
    )


# ─── Basic Functionality ─────────────────────────────────────────────


class TestDiagnoseBasic:
    def test_returns_diagnosis_result(self, healthy_result):
        result = diagnose(healthy_result)
        assert isinstance(result, DiagnosisResult)

    def test_has_status(self, healthy_result):
        result = diagnose(healthy_result)
        assert result.status in ("healthy", "warning", "critical")

    def test_has_action_items(self, healthy_result):
        result = diagnose(healthy_result)
        assert isinstance(result.action_items, list)
        assert len(result.action_items) >= 1

    def test_has_next_steps(self, healthy_result):
        result = diagnose(healthy_result)
        assert isinstance(result.next_steps, list)
        assert len(result.next_steps) >= 1

    def test_has_confidence_level(self, healthy_result):
        result = diagnose(healthy_result)
        assert result.confidence_level in ("high", "medium", "low")


# ─── Status Assessment ──────────────────────────────────────────────


class TestDiagnoseStatus:
    def test_healthy_status(self, healthy_result):
        result = diagnose(healthy_result)
        assert result.status == "healthy"

    def test_warning_status(self, underpowered_result):
        result = diagnose(underpowered_result)
        assert result.status in ("warning", "critical")

    def test_critical_status(self, critical_result):
        result = diagnose(critical_result)
        assert result.status == "critical"


# ─── Confidence Level ───────────────────────────────────────────────


class TestDiagnoseConfidence:
    def test_high_confidence(self):
        r = _make_result(power=0.90, control_n=5000, treatment_n=5000)
        result = diagnose(r)
        assert result.confidence_level == "high"

    def test_low_confidence(self):
        r = _make_result(power=0.20, control_n=50, treatment_n=50)
        result = diagnose(r)
        assert result.confidence_level == "low"


# ─── Action Items ────────────────────────────────────────────────────


class TestDiagnoseActions:
    def test_underpowered_action(self, underpowered_result):
        result = diagnose(underpowered_result)
        power_items = [a for a in result.action_items if "power" in a.lower() or "underpowered" in a.lower()]
        assert len(power_items) >= 1

    def test_small_sample_action(self):
        r = _make_result(control_n=50, treatment_n=50, power=0.10)
        result = diagnose(r)
        sample_items = [a for a in result.action_items if "sample" in a.lower()]
        assert len(sample_items) >= 1

    def test_negligible_effect_action(self):
        r = _make_result(effect_size=0.01, power=0.90, control_n=50000, treatment_n=50000)
        result = diagnose(r)
        effect_items = [a for a in result.action_items if "negligible" in a.lower() or "effect size" in a.lower()]
        assert len(effect_items) >= 1

    def test_large_effect_action(self):
        r = _make_result(effect_size=1.5, power=0.99, control_n=5000, treatment_n=5000)
        result = diagnose(r)
        large_items = [a for a in result.action_items if "Large effect" in a or "bug" in a.lower()]
        assert len(large_items) >= 1


# ─── Next Steps ──────────────────────────────────────────────────────


class TestDiagnoseNextSteps:
    def test_significant_healthy_suggests_ship(self, healthy_result):
        result = diagnose(healthy_result)
        ship_steps = [s for s in result.next_steps if "ship" in s.lower()]
        assert len(ship_steps) >= 1

    def test_not_significant_suggests_more_data(self, underpowered_result):
        result = diagnose(underpowered_result)
        more_data_steps = [
            s for s in result.next_steps
            if "sample" in s.lower() or "power" in s.lower() or "not significant" in s.lower()
        ]
        assert len(more_data_steps) >= 1


# ─── Imbalanced Samples ─────────────────────────────────────────────


class TestDiagnoseImbalance:
    def test_imbalanced_samples_flagged(self):
        r = _make_result(control_n=1000, treatment_n=500)
        result = diagnose(r)
        imbalance_items = [a for a in result.action_items if "imbalanced" in a.lower() or "SRM" in a]
        assert len(imbalance_items) >= 1


# ─── Validation ──────────────────────────────────────────────────────


class TestDiagnoseValidation:
    def test_rejects_non_experiment_result(self):
        with pytest.raises(TypeError, match="ExperimentResult"):
            diagnose("not a result")

    def test_rejects_dict(self):
        with pytest.raises(TypeError):
            diagnose({"lift": 0.02})


# ─── Serialization ──────────────────────────────────────────────────


class TestDiagnoseSerialization:
    def test_to_dict(self, healthy_result):
        result = diagnose(healthy_result)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "status" in d
        assert "action_items" in d
        assert "next_steps" in d
        assert "confidence_level" in d

    def test_repr(self, healthy_result):
        result = diagnose(healthy_result)
        text = repr(result)
        assert "DiagnosisResult" in text
        assert "status=" in text
