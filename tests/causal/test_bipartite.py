"""Tests for BipartiteExperiment (Harshaw et al. 2023)."""

from __future__ import annotations

import numpy as np
import pytest

from splita._types import BipartiteResult
from splita.causal.bipartite import BipartiteExperiment


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _make_bipartite_data(rng, n_buyers=100, n_sellers=30, effect=0.0, density=0.3):
    """Helper to generate bipartite experiment data."""
    buyer_treatments = rng.binomial(1, 0.5, n_buyers).astype(float)
    buyer_outcomes = rng.normal(10, 1, n_buyers) + buyer_treatments * effect
    graph = (rng.random((n_buyers, n_sellers)) < density).astype(float)

    # Seller outcomes depend on exposure to treated buyers
    treated_buyer_counts = graph.T @ buyer_treatments
    total_buyer_counts = graph.sum(axis=0)
    exposure = np.zeros(n_sellers)
    mask = total_buyer_counts > 0
    exposure[mask] = treated_buyer_counts[mask] / total_buyer_counts[mask]
    seller_outcomes = rng.normal(8, 1, n_sellers) + exposure * effect

    return buyer_outcomes, seller_outcomes, buyer_treatments, graph


class TestBipartiteBasic:
    """Basic functionality tests."""

    def test_returns_bipartite_result(self, rng):
        """fit() should return a BipartiteResult."""
        y_b, y_s, t_b, graph = _make_bipartite_data(rng)
        result = BipartiteExperiment().fit(y_b, y_s, t_b, graph)
        assert isinstance(result, BipartiteResult)

    def test_detects_buyer_side_effect(self, rng):
        """Large buyer-side effect should be detected."""
        y_b, y_s, t_b, graph = _make_bipartite_data(rng, effect=3.0, n_buyers=200)
        result = BipartiteExperiment().fit(y_b, y_s, t_b, graph)
        assert result.buyer_side_effect > 1.0

    def test_no_effect_small_buyer_effect(self, rng):
        """No treatment should yield near-zero buyer effect."""
        y_b, y_s, t_b, graph = _make_bipartite_data(rng, effect=0.0, n_buyers=200)
        result = BipartiteExperiment().fit(y_b, y_s, t_b, graph)
        assert abs(result.buyer_side_effect) < 1.0

    def test_n_exposed_sellers(self, rng):
        """n_exposed_sellers should be non-negative and <= n_sellers."""
        n_sellers = 30
        y_b, y_s, t_b, graph = _make_bipartite_data(
            rng, n_sellers=n_sellers, density=0.5
        )
        result = BipartiteExperiment().fit(y_b, y_s, t_b, graph)
        assert 0 <= result.n_exposed_sellers <= n_sellers

    def test_cross_side_pvalue_valid(self, rng):
        """Cross-side p-value should be in [0, 1]."""
        y_b, y_s, t_b, graph = _make_bipartite_data(rng, density=0.5)
        result = BipartiteExperiment().fit(y_b, y_s, t_b, graph)
        assert 0.0 <= result.cross_side_pvalue <= 1.0

    def test_exposure_threshold_affects_n_exposed(self, rng):
        """Higher threshold should reduce the number of exposed sellers."""
        y_b, y_s, t_b, graph = _make_bipartite_data(rng, density=0.4, n_buyers=100)
        r_low = BipartiteExperiment(exposure_threshold=0.3).fit(
            y_b, y_s, t_b, graph
        )
        r_high = BipartiteExperiment(exposure_threshold=0.8).fit(
            y_b, y_s, t_b, graph
        )
        assert r_low.n_exposed_sellers >= r_high.n_exposed_sellers

    def test_to_dict(self, rng):
        """to_dict() should return a plain dict."""
        y_b, y_s, t_b, graph = _make_bipartite_data(rng)
        d = BipartiteExperiment().fit(y_b, y_s, t_b, graph).to_dict()
        assert isinstance(d, dict)
        assert "buyer_side_effect" in d
        assert "n_exposed_sellers" in d


class TestBipartiteValidation:
    """Validation and error handling tests."""

    def test_invalid_exposure_threshold(self):
        """exposure_threshold outside (0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="exposure_threshold"):
            BipartiteExperiment(exposure_threshold=0.0)

    def test_invalid_alpha(self):
        """alpha outside (0, 1) should raise ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            BipartiteExperiment(alpha=1.5)

    def test_non_binary_treatments(self, rng):
        """Non-binary treatments should raise ValueError."""
        with pytest.raises(ValueError, match="only 0 and 1"):
            BipartiteExperiment().fit(
                rng.normal(0, 1, 10),
                rng.normal(0, 1, 5),
                np.array([0, 1, 2, 0, 1, 0, 1, 0, 1, 0], dtype=float),
                np.ones((10, 5)),
            )

    def test_wrong_graph_shape(self, rng):
        """Wrong graph shape should raise ValueError."""
        with pytest.raises(ValueError, match="shape"):
            BipartiteExperiment().fit(
                rng.normal(0, 1, 10),
                rng.normal(0, 1, 5),
                rng.binomial(1, 0.5, 10).astype(float),
                np.ones((5, 10)),  # transposed
            )

    def test_graph_not_ndarray(self, rng):
        """Non-ndarray graph should raise TypeError."""
        with pytest.raises(TypeError, match="numpy ndarray"):
            BipartiteExperiment().fit(
                rng.normal(0, 1, 10),
                rng.normal(0, 1, 5),
                rng.binomial(1, 0.5, 10).astype(float),
                [[1, 0], [0, 1]],  # type: ignore[arg-type]
            )

    def test_graph_wrong_ndim(self, rng):
        """3-D graph should raise ValueError."""
        with pytest.raises(ValueError, match="2-D array"):
            BipartiteExperiment().fit(
                rng.normal(0, 1, 10),
                rng.normal(0, 1, 5),
                rng.binomial(1, 0.5, 10).astype(float),
                np.ones((10, 5, 2)),
            )

    def test_repr(self, rng):
        """__repr__ should produce a readable string."""
        y_b, y_s, t_b, graph = _make_bipartite_data(rng)
        s = repr(BipartiteExperiment().fit(y_b, y_s, t_b, graph))
        assert "BipartiteResult" in s
        assert "buyer_side_effect" in s
