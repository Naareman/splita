"""Tests for MarketplaceExperiment (Bajari et al. 2023)."""

from __future__ import annotations

import numpy as np
import pytest

from splita._types import MarketplaceResult
from splita.causal.marketplace import MarketplaceExperiment


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestMarketplaceBasic:
    """Basic functionality tests."""

    def test_returns_marketplace_result(self, rng):
        """analyze() should return a MarketplaceResult."""
        n = 200
        outcomes = rng.normal(10, 1, n)
        treatments = np.repeat([0, 1], 100).astype(float)
        clusters = np.tile(np.arange(20), 10)
        result = MarketplaceExperiment().analyze(
            outcomes, treatments, side="buyer", clusters=clusters
        )
        assert isinstance(result, MarketplaceResult)

    def test_detects_treatment_effect(self, rng):
        """Large treatment effect should be significant."""
        n = 400
        treatments = np.repeat([0, 1], 200).astype(float)
        outcomes = rng.normal(10, 1, n) + treatments * 3.0
        clusters = np.tile(np.arange(20), 20)
        result = MarketplaceExperiment().analyze(
            outcomes, treatments, side="buyer", clusters=clusters
        )
        assert result.pvalue < 0.05
        assert result.ate > 0

    def test_no_effect_not_significant(self, rng):
        """No treatment effect should yield high p-value."""
        n = 200
        treatments = np.repeat([0, 1], 100).astype(float)
        outcomes = rng.normal(10, 1, n)
        clusters = np.tile(np.arange(20), 10)
        result = MarketplaceExperiment().analyze(
            outcomes, treatments, side="seller", clusters=clusters
        )
        assert result.pvalue > 0.01

    def test_design_effect_positive(self, rng):
        """Design effect should be >= 1."""
        n = 200
        treatments = np.repeat([0, 1], 100).astype(float)
        outcomes = rng.normal(10, 1, n)
        clusters = np.tile(np.arange(10), 20)
        result = MarketplaceExperiment().analyze(
            outcomes, treatments, side="buyer", clusters=clusters
        )
        assert result.design_effect >= 1.0

    def test_estimated_bias_nonnegative(self, rng):
        """Estimated bias should be non-negative."""
        n = 200
        treatments = np.repeat([0, 1], 100).astype(float)
        outcomes = rng.normal(10, 1, n)
        clusters = np.tile(np.arange(20), 10)
        result = MarketplaceExperiment().analyze(
            outcomes, treatments, side="buyer", clusters=clusters
        )
        assert result.estimated_bias >= 0

    def test_recommended_side_valid(self, rng):
        """Recommended side should be 'buyer' or 'seller'."""
        n = 200
        treatments = np.repeat([0, 1], 100).astype(float)
        outcomes = rng.normal(10, 1, n)
        clusters = np.tile(np.arange(20), 10)
        result = MarketplaceExperiment().analyze(
            outcomes, treatments, side="buyer", clusters=clusters
        )
        assert result.recommended_side in ("buyer", "seller")

    def test_to_dict(self, rng):
        """to_dict() should return a plain dict."""
        n = 200
        treatments = np.repeat([0, 1], 100).astype(float)
        outcomes = rng.normal(10, 1, n)
        clusters = np.tile(np.arange(20), 10)
        result = MarketplaceExperiment().analyze(
            outcomes, treatments, side="buyer", clusters=clusters
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "ate" in d


class TestMarketplaceValidation:
    """Validation and error handling tests."""

    def test_invalid_side(self, rng):
        """Invalid side should raise ValueError."""
        with pytest.raises(ValueError, match="'buyer' or 'seller'"):
            MarketplaceExperiment().analyze(
                rng.normal(0, 1, 10),
                np.ones(10),
                side="platform",  # type: ignore[arg-type]
                clusters=np.arange(10),
            )

    def test_invalid_alpha(self):
        """alpha outside (0, 1) should raise ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            MarketplaceExperiment(alpha=0.0)

    def test_non_binary_treatments(self, rng):
        """Non-binary treatments should raise ValueError."""
        with pytest.raises(ValueError, match="only 0 and 1"):
            MarketplaceExperiment().analyze(
                rng.normal(0, 1, 10),
                np.array([0, 1, 2, 0, 1, 0, 1, 0, 1, 0], dtype=float),
                side="buyer",
                clusters=np.arange(10),
            )

    def test_too_few_clusters(self, rng):
        """Fewer than 2 clusters should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2 clusters"):
            MarketplaceExperiment().analyze(
                rng.normal(0, 1, 10),
                np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=float),
                side="buyer",
                clusters=np.zeros(10, dtype=int),
            )

    def test_repr(self, rng):
        """__repr__ should produce a readable string."""
        n = 200
        treatments = np.repeat([0, 1], 100).astype(float)
        outcomes = rng.normal(10, 1, n)
        clusters = np.tile(np.arange(20), 10)
        result = MarketplaceExperiment().analyze(
            outcomes, treatments, side="buyer", clusters=clusters
        )
        s = repr(result)
        assert "MarketplaceResult" in s
        assert "ate" in s
