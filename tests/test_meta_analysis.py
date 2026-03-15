"""Tests for splita.meta_analysis."""

from __future__ import annotations

import math

import pytest

from splita.meta_analysis import meta_analysis


class TestMetaAnalysisFixed:
    """Fixed-effects meta-analysis."""

    def test_two_studies_fixed(self) -> None:
        r = meta_analysis([0.05, 0.03], [0.02, 0.01], method="fixed")
        assert r.method == "fixed"
        assert r.combined_effect > 0

    def test_weights_sum_to_one(self) -> None:
        r = meta_analysis([0.1, 0.2, 0.3], [0.05, 0.03, 0.04],
                          method="fixed")
        assert abs(sum(r.study_weights) - 1.0) < 1e-10

    def test_ci_contains_effect(self) -> None:
        r = meta_analysis([0.05, 0.03], [0.02, 0.01], method="fixed")
        assert r.ci_lower <= r.combined_effect <= r.ci_upper

    def test_smaller_se_gets_more_weight(self) -> None:
        r = meta_analysis([0.05, 0.03], [0.10, 0.01], method="fixed")
        # Study 2 has smaller SE, should get more weight
        assert r.study_weights[1] > r.study_weights[0]


class TestMetaAnalysisRandom:
    """Random-effects (DerSimonian-Laird) meta-analysis."""

    def test_random_is_default(self) -> None:
        r = meta_analysis([0.05, 0.03], [0.02, 0.01])
        assert r.method == "random"

    def test_i_squared_range(self) -> None:
        r = meta_analysis([0.05, 0.03, 0.07], [0.02, 0.01, 0.03])
        assert 0.0 <= r.i_squared <= 1.0

    def test_heterogeneity_pvalue_valid(self) -> None:
        r = meta_analysis([0.05, 0.03, 0.07], [0.02, 0.01, 0.03])
        assert 0.0 <= r.heterogeneity_pvalue <= 1.0

    def test_homogeneous_studies_low_i_squared(self) -> None:
        # Same effect, same SE — should have low heterogeneity
        r = meta_analysis([0.05, 0.05, 0.05], [0.01, 0.01, 0.01])
        assert r.i_squared < 0.1

    def test_heterogeneous_studies_high_i_squared(self) -> None:
        # Very different effects — high heterogeneity
        r = meta_analysis([0.01, 0.50, 0.01, 0.50], [0.01, 0.01, 0.01, 0.01])
        assert r.i_squared > 0.5


class TestMetaAnalysisLabels:
    """Label handling."""

    def test_labels_returned(self) -> None:
        labels = ["A", "B", "C"]
        r = meta_analysis([0.1, 0.2, 0.3], [0.05, 0.03, 0.04],
                          labels=labels)
        assert r.labels == labels

    def test_no_labels_default(self) -> None:
        r = meta_analysis([0.1, 0.2], [0.05, 0.03])
        assert r.labels is None


class TestMetaAnalysisValidation:
    """Input validation."""

    def test_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            meta_analysis([0.1, 0.2], [0.05])

    def test_single_study(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            meta_analysis([0.1], [0.05])

    def test_zero_se(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            meta_analysis([0.1, 0.2], [0.05, 0.0])

    def test_negative_se(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            meta_analysis([0.1, 0.2], [0.05, -0.01])

    def test_invalid_method(self) -> None:
        with pytest.raises(ValueError, match="method"):
            meta_analysis([0.1, 0.2], [0.05, 0.03], method="bayesian")

    def test_mismatched_labels(self) -> None:
        with pytest.raises(ValueError, match="labels"):
            meta_analysis([0.1, 0.2], [0.05, 0.03], labels=["A"])


class TestMetaAnalysisSerialization:
    """to_dict / to_json."""

    def test_to_dict(self) -> None:
        r = meta_analysis([0.05, 0.03], [0.02, 0.01])
        d = r.to_dict()
        assert isinstance(d, dict)
        assert "combined_effect" in d

    def test_to_json(self) -> None:
        r = meta_analysis([0.05, 0.03], [0.02, 0.01])
        j = r.to_json()
        assert isinstance(j, str)
        assert "combined_effect" in j

    def test_pvalue_between_0_and_1(self) -> None:
        r = meta_analysis([0.05, 0.03, 0.07], [0.02, 0.01, 0.03])
        assert 0.0 <= r.pvalue <= 1.0
