"""Tests for FractionalFactorialDesign."""

from __future__ import annotations

import numpy as np
import pytest

from splita.design.factorial import FractionalFactorialDesign


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestFactorialGenerate:
    """Tests for design matrix generation."""

    def test_2_factors_full(self):
        """2 factors should produce a full 2^2 = 4-run design."""
        ffd = FractionalFactorialDesign()
        matrix = ffd.generate(2)

        assert matrix.shape == (4, 2)
        assert set(np.unique(matrix)) == {-1.0, 1.0}

    def test_3_factors(self):
        """3 factors should produce a valid design."""
        matrix = FractionalFactorialDesign().generate(3)

        assert matrix.shape[1] == 3
        assert matrix.shape[0] >= 4  # At least 2^2 = 4 runs

    def test_4_factors_resolution_3(self):
        """4 factors at resolution III should have fewer than 16 runs."""
        matrix = FractionalFactorialDesign().generate(4, resolution=3)

        assert matrix.shape[1] == 4
        assert matrix.shape[0] <= 16

    def test_values_are_plus_minus_one(self):
        """Design matrix should only contain -1 and +1."""
        matrix = FractionalFactorialDesign().generate(5, resolution=3)

        unique_vals = set(np.unique(matrix))
        assert unique_vals == {-1.0, 1.0}

    def test_n_factors_minimum(self):
        """n_factors < 2 should raise ValueError."""
        with pytest.raises(ValueError, match="n_factors"):
            FractionalFactorialDesign().generate(1)

    def test_resolution_too_high(self):
        """Resolution > 5 should raise ValueError."""
        with pytest.raises(ValueError, match="resolution"):
            FractionalFactorialDesign().generate(4, resolution=6)

    def test_higher_resolution_more_runs(self):
        """Higher resolution should generally mean more runs."""
        ffd = FractionalFactorialDesign()
        m3 = ffd.generate(6, resolution=3)
        m5 = ffd.generate(6, resolution=5)

        assert m5.shape[0] >= m3.shape[0]


class TestFactorialAnalyze:
    """Tests for outcome analysis."""

    def test_detect_significant_factor(self, rng):
        """Should detect a factor with a large effect."""
        ffd = FractionalFactorialDesign()
        matrix = ffd.generate(3)
        n_runs = matrix.shape[0]

        # Factor 1 has a large effect, others don't
        outcomes = 5.0 * matrix[:, 0] + rng.normal(0, 0.5, n_runs)

        r = ffd.analyze(outcomes, matrix)

        assert "X1" in r.significant_factors
        assert abs(r.main_effects["X1"]) > abs(r.main_effects["X2"])

    def test_no_significant_factors(self, rng):
        """When outcomes are pure noise, no factors should be significant."""
        ffd = FractionalFactorialDesign()
        matrix = ffd.generate(3)
        outcomes = rng.normal(0, 1, matrix.shape[0])

        r = ffd.analyze(outcomes, matrix)

        # With noise, usually no factors are significant
        # (could occasionally be one by chance)
        assert len(r.significant_factors) <= 1

    def test_custom_factor_names(self, rng):
        """Should use custom factor names when provided."""
        ffd = FractionalFactorialDesign()
        matrix = ffd.generate(3)
        outcomes = 3.0 * matrix[:, 0] + rng.normal(0, 0.5, matrix.shape[0])
        names = ["color", "size", "price"]

        r = ffd.analyze(outcomes, matrix, factor_names=names)

        assert "color" in r.main_effects
        assert "size" in r.main_effects
        assert "price" in r.main_effects

    def test_interactions_computed(self, rng):
        """Should compute two-factor interactions."""
        ffd = FractionalFactorialDesign()
        matrix = ffd.generate(3)
        outcomes = rng.normal(0, 1, matrix.shape[0])

        r = ffd.analyze(outcomes, matrix)

        assert len(r.interactions) > 0
        assert "X1:X2" in r.interactions

    def test_result_fields(self, rng):
        """Result should have all expected fields."""
        ffd = FractionalFactorialDesign()
        matrix = ffd.generate(4)
        outcomes = rng.normal(0, 1, matrix.shape[0])

        r = ffd.analyze(outcomes, matrix)

        assert r.n_factors == 4
        assert r.n_runs == matrix.shape[0]
        assert r.resolution >= 3

    def test_wrong_outcome_length(self, rng):
        """Outcomes length must match design matrix rows."""
        ffd = FractionalFactorialDesign()
        matrix = ffd.generate(3)

        with pytest.raises(ValueError, match="same length"):
            ffd.analyze(rng.normal(0, 1, 100), matrix)

    def test_wrong_factor_names_length(self, rng):
        """factor_names length must match n_factors."""
        ffd = FractionalFactorialDesign()
        matrix = ffd.generate(3)

        with pytest.raises(ValueError, match="one name per factor"):
            ffd.analyze(rng.normal(0, 1, matrix.shape[0]), matrix,
                        factor_names=["a", "b"])

    def test_effect_sizes_computed(self, rng):
        """Effect sizes should be computed for all factors."""
        ffd = FractionalFactorialDesign()
        matrix = ffd.generate(3)
        outcomes = 2.0 * matrix[:, 0] + rng.normal(0, 1, matrix.shape[0])

        r = ffd.analyze(outcomes, matrix)

        assert len(r.effect_sizes) == 3
        assert abs(r.effect_sizes["X1"]) > abs(r.effect_sizes["X2"])

    def test_invalid_alpha(self):
        """Alpha outside (0, 1) should raise ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            FractionalFactorialDesign(alpha=0.0)

    def test_to_dict(self, rng):
        """to_dict should return a plain dictionary."""
        ffd = FractionalFactorialDesign()
        matrix = ffd.generate(3)
        outcomes = rng.normal(0, 1, matrix.shape[0])
        r = ffd.analyze(outcomes, matrix)

        d = r.to_dict()
        assert isinstance(d, dict)
        assert "main_effects" in d

    def test_repr(self, rng):
        """repr should return a formatted string."""
        ffd = FractionalFactorialDesign()
        matrix = ffd.generate(3)
        r = ffd.analyze(rng.normal(0, 1, matrix.shape[0]), matrix)
        assert "FactorialResult" in repr(r)
