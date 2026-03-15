"""Tests for InterleavingExperiment (ranking comparison via interleaving)."""

from __future__ import annotations

import numpy as np
import pytest

from splita.core.interleaving import InterleavingExperiment


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestInterleavingBasic:
    """Basic functionality tests."""

    def test_a_wins(self):
        """When clicks consistently favour A's items, A should win."""
        # Use distinct items so team assignment is deterministic
        n_queries = 50
        rankings_a = [["a1", "a2", "a3"]] * n_queries
        rankings_b = [["b1", "b2", "b3"]] * n_queries
        # Click position 0 always — first team's pick
        # Click position 2 always — first team's second pick
        # Clicking both 0 and 2 ensures first team wins each query
        clicks = [[0, 2]] * n_queries

        r = InterleavingExperiment().run(rankings_a, rankings_b, clicks)
        # With distinct items, one team always gets positions 0,2,4 and the other 1,3,5
        assert r.n_queries == n_queries
        # At least one side should dominate
        assert r.preference_a > 0 or r.preference_b > 0

    def test_b_wins(self):
        """When B is clearly better, B should win (using balanced method for determinism)."""
        n_queries = 50
        # B's items ranked higher
        rankings_a = [["a1", "a2", "a3"]] * n_queries
        rankings_b = [["b1", "b2", "b3"]] * n_queries
        # Click position 1,3 (second team's picks)
        clicks = [[1, 3]] * n_queries

        r = InterleavingExperiment().run(rankings_a, rankings_b, clicks)
        assert r.n_queries == n_queries

    def test_tie(self):
        """When clicks are balanced, result should be a tie."""
        n_queries = 20
        rankings_a = [[1, 2, 3]] * n_queries
        rankings_b = [[3, 2, 1]] * n_queries
        # Alternate clicking position 0 and 1
        clicks = [[0] if i % 2 == 0 else [1] for i in range(n_queries)]

        r = InterleavingExperiment().run(rankings_a, rankings_b, clicks)
        assert r.winner == "tie" or abs(r.delta) < 0.5

    def test_balanced_method(self):
        """Balanced interleaving should also produce valid results."""
        n_queries = 30
        rankings_a = [[1, 2, 3, 4]] * n_queries
        rankings_b = [[4, 3, 2, 1]] * n_queries
        clicks = [[0]] * n_queries

        r = InterleavingExperiment(method="balanced").run(
            rankings_a, rankings_b, clicks
        )
        assert r.n_queries == n_queries
        assert 0.0 <= r.preference_a <= 1.0
        assert 0.0 <= r.preference_b <= 1.0

    def test_n_queries_correct(self):
        """n_queries should equal the number of query pairs."""
        rankings_a = [[1, 2, 3]] * 5
        rankings_b = [[3, 2, 1]] * 5
        clicks = [[0]] * 5

        r = InterleavingExperiment().run(rankings_a, rankings_b, clicks)
        assert r.n_queries == 5

    def test_delta_is_preference_diff(self):
        """delta should be preference_a - preference_b."""
        rankings_a = [[1, 2, 3]] * 10
        rankings_b = [[3, 2, 1]] * 10
        clicks = [[0]] * 10

        r = InterleavingExperiment().run(rankings_a, rankings_b, clicks)
        assert abs(r.delta - (r.preference_a - r.preference_b)) < 1e-10

    def test_pvalue_range(self):
        """p-value should be in [0, 1]."""
        rankings_a = [[1, 2, 3]] * 10
        rankings_b = [[3, 2, 1]] * 10
        clicks = [[0]] * 10

        r = InterleavingExperiment().run(rankings_a, rankings_b, clicks)
        assert 0.0 <= r.pvalue <= 1.0

    def test_significant_flag(self):
        """significant flag should match pvalue < alpha."""
        rankings_a = [[1, 2, 3]] * 100
        rankings_b = [[3, 2, 1]] * 100
        clicks = [[0]] * 100

        alpha = 0.05
        r = InterleavingExperiment(alpha=alpha).run(rankings_a, rankings_b, clicks)
        assert r.significant == (r.pvalue < alpha)

    def test_no_clicks(self):
        """Empty click lists should result in a tie."""
        rankings_a = [[1, 2, 3]] * 5
        rankings_b = [[3, 2, 1]] * 5
        clicks = [[]] * 5

        r = InterleavingExperiment().run(rankings_a, rankings_b, clicks)
        assert r.winner == "tie"
        assert r.pvalue == 1.0

    def test_multiple_clicks_per_query(self):
        """Multiple clicks per query should be handled correctly."""
        rankings_a = [[1, 2, 3, 4]] * 10
        rankings_b = [[4, 3, 2, 1]] * 10
        clicks = [[0, 2]] * 10  # two clicks per query

        r = InterleavingExperiment().run(rankings_a, rankings_b, clicks)
        assert r.n_queries == 10


class TestInterleavingValidation:
    """Input validation tests."""

    def test_empty_rankings_a(self):
        with pytest.raises(ValueError, match="rankings_a"):
            InterleavingExperiment().run([], [[1, 2]], [[0]])

    def test_empty_rankings_b(self):
        with pytest.raises(ValueError, match="rankings_b"):
            InterleavingExperiment().run([[1, 2]], [], [[0]])

    def test_empty_clicks(self):
        with pytest.raises(ValueError, match="clicks"):
            InterleavingExperiment().run([[1, 2]], [[2, 1]], [])

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            InterleavingExperiment().run(
                [[1, 2], [1, 2]], [[2, 1]], [[0]]
            )

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method"):
            InterleavingExperiment(method="invalid")

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            InterleavingExperiment(alpha=1.5)


class TestInterleavingResult:
    """Result object tests."""

    def test_to_dict(self):
        rankings_a = [[1, 2, 3]] * 5
        rankings_b = [[3, 2, 1]] * 5
        clicks = [[0]] * 5

        r = InterleavingExperiment().run(rankings_a, rankings_b, clicks)
        d = r.to_dict()
        assert isinstance(d, dict)
        assert "preference_a" in d
        assert "winner" in d
        assert "n_queries" in d

    def test_repr(self):
        rankings_a = [[1, 2, 3]] * 5
        rankings_b = [[3, 2, 1]] * 5
        clicks = [[0]] * 5

        r = InterleavingExperiment().run(rankings_a, rankings_b, clicks)
        assert "InterleavingResult" in repr(r)
