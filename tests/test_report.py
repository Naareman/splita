"""Tests for splita.report — experiment report generation."""

from __future__ import annotations

import pytest

from splita._types import BayesianResult, ExperimentResult, SampleSizeResult, SRMResult
from splita.report import report


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
def nonsig_result() -> ExperimentResult:
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
def srm_pass() -> SRMResult:
    return SRMResult(
        observed=[5000, 5000],
        expected_counts=[5000.0, 5000.0],
        chi2_statistic=0.0,
        pvalue=1.0,
        passed=True,
        alpha=0.01,
        deviations_pct=[0.0, 0.0],
        worst_variant=0,
        message="No SRM detected.",
    )


@pytest.fixture
def srm_fail() -> SRMResult:
    return SRMResult(
        observed=[4000, 6000],
        expected_counts=[5000.0, 5000.0],
        chi2_statistic=400.0,
        pvalue=1e-50,
        passed=False,
        alpha=0.01,
        deviations_pct=[-20.0, 20.0],
        worst_variant=1,
        message="SRM detected.",
    )


@pytest.fixture
def bayesian_result() -> BayesianResult:
    return BayesianResult(
        prob_b_beats_a=0.97,
        expected_loss_a=0.005,
        expected_loss_b=0.0002,
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
def sample_size_result() -> SampleSizeResult:
    return SampleSizeResult(
        n_per_variant=5000,
        n_total=10000,
        alpha=0.05,
        power=0.80,
        mde=0.02,
        relative_mde=0.20,
        baseline=0.10,
        metric="conversion",
        effect_size=0.07,
        days_needed=14,
    )


# ─── Basic HTML output ────────────────────────────────────────────────


class TestHTMLOutput:
    """Test HTML report generation."""

    def test_html_contains_title(self, significant_result):
        html = report(significant_result, title="My Test")
        assert "My Test" in html

    def test_html_is_self_contained(self, significant_result):
        html = report(significant_result)
        assert "<style>" in html
        assert "</html>" in html
        assert "<!DOCTYPE html>" in html

    def test_html_contains_result_fields(self, significant_result):
        html = report(significant_result)
        assert "control_mean" in html
        assert "pvalue" in html

    def test_html_with_srm_shows_badge(self, significant_result, srm_pass):
        html = report(significant_result, srm_pass)
        assert "PASS" in html
        assert "Data Quality" in html

    def test_html_with_failed_srm_shows_fail_badge(self, significant_result, srm_fail):
        html = report(significant_result, srm_fail)
        assert "FAIL" in html

    def test_html_includes_explanation(self, significant_result):
        html = report(significant_result)
        assert "significant" in html.lower()


# ─── Plain text output ────────────────────────────────────────────────


class TestTextOutput:
    """Test plain-text report generation."""

    def test_text_contains_title(self, significant_result):
        text = report(significant_result, format="text")
        assert "Experiment Report" in text

    def test_text_custom_title(self, significant_result):
        text = report(significant_result, title="Q1 Test", format="text")
        assert "Q1 Test" in text

    def test_text_contains_sections(self, significant_result, srm_pass):
        text = report(significant_result, srm_pass, format="text")
        assert "Summary" in text
        assert "Data Quality" in text
        assert "Primary Metrics" in text

    def test_text_shows_result_fields(self, significant_result):
        text = report(significant_result, format="text")
        assert "control_mean" in text


# ─── Multiple result types ───────────────────────────────────────────


class TestMultipleResults:
    """Test reports with multiple result types."""

    def test_all_four_types(
        self, significant_result, srm_pass, bayesian_result, sample_size_result
    ):
        html = report(
            significant_result, srm_pass, bayesian_result, sample_size_result
        )
        assert "Primary Metrics" in html
        assert "Data Quality" in html
        assert "Secondary Metrics" in html

    def test_multiple_primary_results(self, significant_result, nonsig_result):
        html = report(significant_result, nonsig_result)
        assert "Significant" in html
        assert "Not significant" in html

    def test_result_count_in_summary(
        self, significant_result, srm_pass, bayesian_result
    ):
        html = report(significant_result, srm_pass, bayesian_result)
        assert ">3<" in html  # The count in stat-card


# ─── Recommendations ─────────────────────────────────────────────────


class TestRecommendations:
    """Test recommendation generation."""

    def test_underpowered_recommendation(self, nonsig_result):
        html = report(nonsig_result)
        assert "underpowered" in html.lower() or "variance reduction" in html.lower()

    def test_srm_fail_recommendation(self, srm_fail, significant_result):
        html = report(significant_result, srm_fail)
        assert "Mismatch" in html

    def test_no_recommendations_for_clean_result(self, significant_result, srm_pass):
        text = report(significant_result, srm_pass, format="text")
        # A significant, well-powered result with passing SRM should not
        # have SRM-failure recommendations
        assert "Investigate randomization" not in text


# ─── Error handling ──────────────────────────────────────────────────


class TestErrorHandling:
    """Test validation and error paths."""

    def test_no_results_raises_valueerror(self):
        with pytest.raises(ValueError, match="at least one"):
            report()

    def test_invalid_format_raises_valueerror(self, significant_result):
        with pytest.raises(ValueError, match="must be"):
            report(significant_result, format="pdf")

    def test_non_dataclass_raises_typeerror(self):
        with pytest.raises(TypeError, match="not a splita result"):
            report({"not": "a result"})

    def test_string_input_raises_typeerror(self):
        with pytest.raises(TypeError, match="not a splita result"):
            report("not a result")
