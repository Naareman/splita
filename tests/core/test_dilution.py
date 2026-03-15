"""Tests for DilutionAnalysis."""

from __future__ import annotations

import pytest

from splita.core.dilution import DilutionAnalysis


class TestDilutionBasic:
    """Basic functionality tests."""

    def test_known_dilution(self):
        """Diluted ATE = triggered ATE * trigger rate."""
        r = DilutionAnalysis().dilute(
            triggered_effect=10.0, triggered_se=1.0, trigger_rate=0.3
        )
        assert abs(r.diluted_ate - 3.0) < 1e-10
        assert abs(r.diluted_se - 0.3) < 1e-10

    def test_full_trigger(self):
        """With trigger_rate=1.0, diluted = triggered."""
        r = DilutionAnalysis().dilute(
            triggered_effect=5.0, triggered_se=0.5, trigger_rate=1.0
        )
        assert abs(r.diluted_ate - 5.0) < 1e-10
        assert abs(r.diluted_se - 0.5) < 1e-10

    def test_small_trigger_rate(self):
        """Small trigger rate should shrink the effect proportionally."""
        r = DilutionAnalysis().dilute(
            triggered_effect=20.0, triggered_se=2.0, trigger_rate=0.05
        )
        assert abs(r.diluted_ate - 1.0) < 1e-10
        assert abs(r.diluted_se - 0.1) < 1e-10

    def test_negative_effect(self):
        """Negative triggered effect should remain negative after dilution."""
        r = DilutionAnalysis().dilute(
            triggered_effect=-4.0, triggered_se=1.0, trigger_rate=0.5
        )
        assert r.diluted_ate < 0
        assert abs(r.diluted_ate - (-2.0)) < 1e-10

    def test_triggered_ate_preserved(self):
        """triggered_ate should equal the input."""
        r = DilutionAnalysis().dilute(
            triggered_effect=7.5, triggered_se=1.0, trigger_rate=0.4
        )
        assert r.triggered_ate == 7.5
        assert r.trigger_rate == 0.4

    def test_pvalue_significant(self):
        """Large effect should yield small p-value."""
        r = DilutionAnalysis().dilute(
            triggered_effect=10.0, triggered_se=0.5, trigger_rate=0.5
        )
        assert r.diluted_pvalue < 0.05

    def test_pvalue_not_significant(self):
        """Small effect should not be significant."""
        r = DilutionAnalysis().dilute(
            triggered_effect=0.1, triggered_se=5.0, trigger_rate=0.1
        )
        assert r.diluted_pvalue > 0.05

    def test_zero_effect(self):
        """Zero effect should give p-value = 1."""
        r = DilutionAnalysis().dilute(
            triggered_effect=0.0, triggered_se=1.0, trigger_rate=0.5
        )
        assert r.diluted_ate == 0.0
        assert r.diluted_pvalue == 1.0

    def test_to_dict(self):
        r = DilutionAnalysis().dilute(
            triggered_effect=5.0, triggered_se=1.0, trigger_rate=0.3
        )
        d = r.to_dict()
        assert isinstance(d, dict)
        assert "diluted_ate" in d

    def test_repr(self):
        r = DilutionAnalysis().dilute(
            triggered_effect=5.0, triggered_se=1.0, trigger_rate=0.3
        )
        assert "DilutionResult" in repr(r)


class TestDilutionValidation:
    """Validation and error tests."""

    def test_alpha_out_of_range(self):
        with pytest.raises(ValueError, match="alpha"):
            DilutionAnalysis(alpha=0.0)

    def test_trigger_rate_zero(self):
        with pytest.raises(ValueError, match="trigger_rate"):
            DilutionAnalysis().dilute(
                triggered_effect=5.0, triggered_se=1.0, trigger_rate=0.0
            )

    def test_trigger_rate_negative(self):
        with pytest.raises(ValueError, match="trigger_rate"):
            DilutionAnalysis().dilute(
                triggered_effect=5.0, triggered_se=1.0, trigger_rate=-0.1
            )

    def test_trigger_rate_above_one(self):
        with pytest.raises(ValueError, match="trigger_rate"):
            DilutionAnalysis().dilute(
                triggered_effect=5.0, triggered_se=1.0, trigger_rate=1.5
            )

    def test_negative_se(self):
        with pytest.raises(ValueError, match="triggered_se"):
            DilutionAnalysis().dilute(
                triggered_effect=5.0, triggered_se=-1.0, trigger_rate=0.5
            )
