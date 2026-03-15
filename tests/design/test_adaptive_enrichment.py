"""Tests for AdaptiveEnrichment (Simon & Simon 2013)."""

from __future__ import annotations

import numpy as np
import pytest

from splita._types import EnrichmentResult
from splita.design.adaptive_enrichment import AdaptiveEnrichment


class TestAdaptiveEnrichmentBasic:
    """Basic functionality tests."""

    def test_returns_enrichment_result(self):
        """update() should return an EnrichmentResult."""
        enricher = AdaptiveEnrichment()
        result = enricher.update({
            "young": (2.0, 0.5),
            "old": (0.1, 0.8),
        })
        assert isinstance(result, EnrichmentResult)

    def test_selects_effective_subgroups(self):
        """Subgroups with high z-score should be selected."""
        enricher = AdaptiveEnrichment(futility_threshold=1.0)
        result = enricher.update({
            "A": (5.0, 1.0),   # z = 5.0 -> selected
            "B": (0.1, 1.0),   # z = 0.1 -> dropped
        })
        assert "A" in result.selected_subgroups
        assert "B" in result.dropped_subgroups

    def test_drops_ineffective_subgroups(self):
        """Subgroups with low z-score should be dropped."""
        enricher = AdaptiveEnrichment(futility_threshold=0.5)
        result = enricher.update({
            "X": (0.01, 1.0),  # z = 0.01 -> dropped
            "Y": (3.0, 0.5),   # z = 6.0 -> selected
        })
        assert "X" in result.dropped_subgroups
        assert "Y" in result.selected_subgroups

    def test_stage_increments(self):
        """Stage should increment with each update call."""
        enricher = AdaptiveEnrichment()
        r1 = enricher.update({"A": (1.0, 0.5)})
        assert r1.stage == 1
        r2 = enricher.update({"A": (1.5, 0.5)})
        assert r2.stage == 2

    def test_enrichment_ratios(self):
        """Enrichment ratios should be absolute z-scores."""
        enricher = AdaptiveEnrichment()
        result = enricher.update({
            "A": (3.0, 1.0),  # z = 3.0
            "B": (-2.0, 1.0),  # z = 2.0
        })
        assert abs(result.enrichment_ratios["A"] - 3.0) < 0.01
        assert abs(result.enrichment_ratios["B"] - 2.0) < 0.01

    def test_dropped_subgroups_persist(self):
        """Dropped subgroups should stay dropped across updates."""
        enricher = AdaptiveEnrichment(futility_threshold=1.0)
        enricher.update({
            "A": (5.0, 1.0),
            "B": (0.01, 1.0),
        })
        r2 = enricher.update({
            "A": (4.0, 1.0),
        })
        assert "B" in r2.dropped_subgroups

    def test_result_method(self):
        """result() should return the latest enrichment state."""
        enricher = AdaptiveEnrichment()
        enricher.update({"A": (2.0, 0.5)})
        result = enricher.result()
        assert isinstance(result, EnrichmentResult)
        assert result.stage == 1

    def test_to_dict(self):
        """to_dict() should return a plain dict."""
        enricher = AdaptiveEnrichment()
        result = enricher.update({"A": (2.0, 0.5), "B": (0.5, 1.0)})
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "selected_subgroups" in d


class TestAdaptiveEnrichmentValidation:
    """Validation and error handling tests."""

    def test_empty_subgroup_results(self):
        """Empty dict should raise ValueError."""
        enricher = AdaptiveEnrichment()
        with pytest.raises(ValueError, match="can't be empty"):
            enricher.update({})

    def test_non_dict_subgroup_results(self):
        """Non-dict input should raise TypeError."""
        enricher = AdaptiveEnrichment()
        with pytest.raises(TypeError, match="must be a dict"):
            enricher.update([("A", (1.0, 0.5))])  # type: ignore[arg-type]

    def test_invalid_tuple_length(self):
        """Tuple with wrong length should raise ValueError."""
        enricher = AdaptiveEnrichment()
        with pytest.raises(ValueError, match="\\(effect, se\\) tuple"):
            enricher.update({"A": (1.0,)})  # type: ignore[arg-type]

    def test_non_finite_effect(self):
        """Non-finite effect should raise ValueError."""
        enricher = AdaptiveEnrichment()
        with pytest.raises(ValueError, match="finite"):
            enricher.update({"A": (float("inf"), 0.5)})

    def test_non_positive_se(self):
        """Non-positive SE should raise ValueError."""
        enricher = AdaptiveEnrichment()
        with pytest.raises(ValueError, match="positive and finite"):
            enricher.update({"A": (1.0, 0.0)})

    def test_negative_futility_threshold(self):
        """Negative futility_threshold should raise ValueError."""
        with pytest.raises(ValueError, match="futility_threshold"):
            AdaptiveEnrichment(futility_threshold=-0.1)

    def test_invalid_alpha(self):
        """alpha outside (0, 1) should raise ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            AdaptiveEnrichment(alpha=1.5)

    def test_result_before_update(self):
        """result() before any update should raise ValueError."""
        enricher = AdaptiveEnrichment()
        with pytest.raises(ValueError, match="No interim analyses"):
            enricher.result()

    def test_repr(self):
        """__repr__ should produce a readable string."""
        enricher = AdaptiveEnrichment()
        result = enricher.update({"A": (2.0, 0.5)})
        s = repr(result)
        assert "EnrichmentResult" in s
        assert "stage" in s
