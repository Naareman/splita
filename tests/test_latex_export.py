"""Tests for splita.export.latex — LaTeX table generation."""

from __future__ import annotations

import pytest

from splita._types import BayesianResult, ExperimentResult, SampleSizeResult, SRMResult
from splita.export.latex import to_latex_table, to_latex_tabular


@pytest.fixture
def experiment_result() -> ExperimentResult:
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
        days_needed=8,
    )


class TestToLatexTabular:
    def test_contains_tabular_env(self, experiment_result):
        tex = to_latex_tabular(experiment_result)
        assert r"\begin{tabular}" in tex
        assert r"\end{tabular}" in tex

    def test_contains_booktabs_rules(self, experiment_result):
        tex = to_latex_tabular(experiment_result)
        assert r"\toprule" in tex
        assert r"\midrule" in tex
        assert r"\bottomrule" in tex

    def test_contains_field_names(self, experiment_result):
        tex = to_latex_tabular(experiment_result)
        assert "control\\_mean" in tex
        assert "pvalue" in tex
        assert "significant" in tex

    def test_contains_field_values(self, experiment_result):
        tex = to_latex_tabular(experiment_result)
        assert "0.0030" in tex
        assert "ztest" in tex

    def test_escapes_underscores(self, experiment_result):
        tex = to_latex_tabular(experiment_result)
        # Field names with underscores should be escaped
        assert "control_mean" not in tex
        assert "control\\_mean" in tex


class TestToLatexTable:
    def test_contains_table_env(self, experiment_result):
        tex = to_latex_table(experiment_result)
        assert r"\begin{table}" in tex
        assert r"\end{table}" in tex

    def test_default_caption_is_class_name(self, experiment_result):
        tex = to_latex_table(experiment_result)
        assert r"\caption{ExperimentResult}" in tex

    def test_custom_caption(self, experiment_result):
        tex = to_latex_table(experiment_result, caption="A/B Test Results")
        assert r"\caption{A/B Test Results}" in tex

    def test_custom_label(self, experiment_result):
        tex = to_latex_table(experiment_result, label="tab:ab")
        assert r"\label{tab:ab}" in tex

    def test_no_label_when_none(self, experiment_result):
        tex = to_latex_table(experiment_result)
        assert r"\label" not in tex

    def test_centering(self, experiment_result):
        tex = to_latex_table(experiment_result)
        assert r"\centering" in tex

    def test_works_with_sample_size(self, sample_size_result):
        tex = to_latex_table(sample_size_result)
        assert r"\begin{table}" in tex
        assert "3843" in tex


class TestToLatexMethod:
    """Test the to_latex() method on _DictMixin."""

    def test_to_latex_method(self, experiment_result):
        tex = experiment_result.to_latex()
        assert r"\begin{tabular}" in tex
        assert "control\\_mean" in tex

    def test_to_latex_produces_same_as_standalone(self, experiment_result):
        method_result = experiment_result.to_latex()
        standalone_result = to_latex_tabular(experiment_result)
        assert method_result == standalone_result
